# This file is based on https://github.com/Yaoyi-Li/GCA-Matting

import cv2
import numpy as np
import random
import glob
from   torch.utils.data import Dataset
from   torch.nn import functional as F
from   torchvision import transforms
from scipy.ndimage import grey_dilation, grey_erosion
import torch
from   torch.utils.data import DataLoader
import numpy
import timeit
import math

class ToTensorTrain(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        image, alpha = sample['image'][:,:,::-1], sample['alpha']
        image = image.astype(np.float32).transpose((2, 0, 1))
        fg, bg = sample['fg'][:,:,::-1], sample['bg'][:,:,::-1]
        fg, bg = fg.astype(np.float32).transpose((2, 0, 1)), bg.astype(np.float32).transpose((2, 0, 1))
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        image /= 255.
        fg /= 255.
        bg /= 255.
        sample['image'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(alpha)
        sample['fg'], sample['bg'] = torch.from_numpy(fg), torch.from_numpy(bg)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        sample['fg'] = sample['fg'].sub_(self.mean).div_(self.std)
        sample['bg'] = sample['bg'].sub_(self.mean).div_(self.std)
        return sample

class ToTensorTest(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __call__(self, sample):
        image = sample['image'][:,:,::-1].astype(np.float32).transpose((2, 0, 1))
        alpha = np.expand_dims(sample['alpha'].astype(np.float32), axis=0)
        image /= 255.
        sample['image'], sample['alpha'] = torch.from_numpy(image), torch.from_numpy(alpha)
        sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        return sample

class RandomAffine(object):
    def __init__(self, degrees=30, shear=10, flip=0.5):
        self.degrees = (-degrees, degrees)
        self.shear = (-shear, shear)
        self.flip = flip

    @staticmethod
    def get_params(degrees, shears, flip):
        angle = random.uniform(degrees[0], degrees[1])
        translations = (0, 0)
        scale = (1.0, 1.0)
        shear = random.uniform(shears[0], shears[1])

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        H, W, _ = sample['fg'].shape
        if np.maximum(W, H) < 1024:
            params = self.get_params((0, 0), self.shear, self.flip)
        else:
            params = self.get_params(self.degrees, self.shear, self.flip)

        center = (W * 0.5 + 0.5, H * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        sample['fg'] = cv2.warpAffine(sample['fg'], M, (W, H),
                            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        sample['alpha'] = cv2.warpAffine(sample['alpha'], M, (W, H),
                               flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
        return sample

    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

class RandFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = cv2.flip(sample['image'], 1)
            sample['fg'] = cv2.flip(sample['fg'], 1)
            sample['bg'] = cv2.flip(sample['bg'], 1)
            sample['alpha'] = cv2.flip(sample['alpha'], 1)
        return sample

class RandCrop(object):
    def __init__(self, L):
        self.L = L

    def __call__(self, sample):
        H, W = sample['alpha'].shape

        h = np.random.randint(0, H-self.L+1)
        w = np.random.randint(0, W-self.L+1)
        sample['fg'] = sample['fg'][h:h+self.L, w:w+self.L]
        sample['alpha'] = sample['alpha'][h:h+self.L, w:w+self.L]

        bg = cv2.resize(sample['bg'], (W, H))
        h = np.random.randint(0, H-self.L+1)
        w = np.random.randint(0, W-self.L+1)
        sample['bg'] = bg[h:h+self.L, w:w+self.L]

        return sample

class RandJitter(object):
    def __call__(self, sample):
        alpha = sample['alpha']
        if not np.all(alpha==0):
            fg = sample['fg']
            # convert to HSV space, convert to float32 image to keep precision during space conversion.
            fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
            # Hue noise
            hue_jitter = np.random.randint(-40, 40)
            fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
            # Saturation noise
            sat_bar = fg[:, :, 1][alpha > 0].mean()
            sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
            sat = fg[:, :, 1]
            sat = np.abs(sat + sat_jitter)
            sat[sat>1] = 2 - sat[sat>1]
            fg[:, :, 1] = sat
            # Value noise
            val_bar = fg[:, :, 2][alpha > 0].mean()
            val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
            val = fg[:, :, 2]
            val = np.abs(val + val_jitter)
            val[val>1] = 2 - val[val>1]
            fg[:, :, 2] = val
            # convert back to BGR space
            fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)*255
            sample['fg'] = fg
        return sample

def check_value(v, vm, vM):
    v[v < vm ] = vm
    v[v > vM] = vM
    return v

class Composite(object):
    def __call__(self, sample):
        fg = check_value(sample['fg'], 0, 255)
        bg = check_value(sample['bg'], 0, 255)
        alpha = check_value(sample['alpha'], 0, 1)
        sample['image'] = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        return sample

def get_new_hw(H, W, L, min_s, max_s):
    hs = min_s + (max_s - min_s) * random.random()
    ws = min_s + (max_s - min_s) * random.random()
    s = max([L / H, L / W])
    h = round(H * s * hs)
    w = round(W * s * ws)
    return h, w

class RandRescale(object):
    def __init__(self, L, min_s, max_s):
        self.L = L
        self.min_s = min_s
        self.max_s = max_s

    def __call__(self, sample):
        h, w = get_new_hw(sample['fg'].shape[0], sample['fg'].shape[1], self.L, self.min_s, self.max_s)
        sample['fg'] = cv2.resize(sample['fg'], (w, h))
        sample['alpha'] = cv2.resize(sample['alpha'], (w, h))
        h, w = get_new_hw(sample['bg'].shape[0], sample['bg'].shape[1], self.L, self.min_s, self.max_s)
        sample['bg'] = cv2.resize(sample['bg'], (w, h))        
        return sample

def alpha_to_trimap(alpha):
    mask = (alpha > 1e-4).float().numpy()
    L = alpha.shape[0] + alpha.shape[1]
    L = round(L * 0.025)
    mask1 = grey_dilation(mask, size=(L, L))
    mask2 = grey_erosion(mask, size=(L, L))
    trimap = np.zeros_like(mask, dtype=np.float32)
    trimap[alpha > 0.5] = 1
    trimap[np.where(mask1 - mask2 != 0)] = 0.5
    return torch.tensor(trimap, dtype=torch.float32).unsqueeze(0)

class Generate_Trimap():    
    def __call__(self, sample):
        sample['trimap'] = alpha_to_trimap(sample['alpha'][0])
        return sample

class ResizeImage():
    def __init__(self, L):
        self.L = L

    def __call__(self, sample):
        sample['image'] = F.interpolate(sample['image'].unsqueeze(0), size=(self.L, self.L), mode='area')[0]
        sample['alpha'] = F.interpolate(sample['alpha'].unsqueeze(0), size=(self.L, self.L), mode='area')[0]
        return sample

class TrainDataset(Dataset):
    def __init__(self, fg_root, alpha_root, bg_root):
        self.fg_paths = glob.glob(fg_root + '/*')
        self.alpha_paths = glob.glob(alpha_root + '/*')
        self.fg_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.alpha_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.bg_paths = glob.glob(bg_root + '/*')
        self.transforms = transforms.Compose([ RandomAffine(),
                                                RandRescale(512, 1, 1.5),
                                                RandCrop(512),
                                                RandJitter(),
                                                Composite(),
                                                RandFlip(),
                                                ToTensorTrain(),
                                                Generate_Trimap()
                                            ])

    def __getitem__(self, idx):
        fg = cv2.imread(self.fg_paths[idx])
        fg = fg.astype(np.float32)
        bg = cv2.imread(random.choice(self.bg_paths), 1)
        alpha = cv2.imread(self.alpha_paths[idx], 0)
        alpha = alpha.astype(np.float32)/255
        sample = {'fg': fg, 'bg': bg, 'alpha': alpha}
        sample = self.composite_fg(sample)
        return self.transforms(sample)
                
    def __len__(self):
        return len(self.fg_paths)

    def composite_fg(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        if random.random() < 0.5:
            idx_other = np.random.randint(len(self.fg_paths))
            fg_other = cv2.imread(self.fg_paths[idx_other]).astype(np.float32)
            alpha_other = cv2.imread(self.alpha_paths[idx_other], 0).astype(np.float32)/255

            fg_other = cv2.resize(fg_other, (alpha.shape[1], alpha.shape[0]))
            alpha_other = cv2.resize(alpha_other, (alpha.shape[1], alpha.shape[0]))

            sample['alpha'] = 1 - (1 - alpha) * (1 - alpha_other)
            alpha_add = alpha + 1e-6
            alpha_other_add = alpha_other + 1e-6
            sample['fg'] = (fg * np.expand_dims(alpha_add, 2) + fg_other * np.expand_dims(alpha_other_add, 2)) / (np.expand_dims(alpha_add, 2) + np.expand_dims(alpha_other_add, 2))
        return sample

class TestDataset(Dataset):
    def __init__(self, image_root, alpha_root):
        self.image_paths = glob.glob(image_root + '/*')
        self.alpha_paths = glob.glob(alpha_root + '/*')
        self.image_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.alpha_paths.sort(key=lambda path: path.split('/')[-1].split('\\')[-1].split('.')[0])
        self.transforms = transforms.Compose([ToTensorTest(), ResizeImage(512)])

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = image.astype(np.float32)
        alpha = cv2.imread(self.alpha_paths[idx], 0)
        alpha = alpha.astype(np.float32)/255
        name = self.alpha_paths[idx].split('/')[-1].split('\\')[-1].split('.')[0]
        sample = {'image': image, 'alpha': alpha, 'name': name, 'H': image.shape[0], 'W': image.shape[1]}
        return self.transforms(sample)

    def __len__(self):
        return len(self.image_paths)

def worker_init_fn(worker):
    seed = int(worker * 100000 + timeit.default_timer() % 1000)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def get_train_dataloader(fg_root, alpha_root, bg_root, batch_size=16):
    return DataLoader(TrainDataset(fg_root, alpha_root, bg_root),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            sampler=None,
            drop_last=True)

def get_test_dataloader(image_root, alpha_root):
    return DataLoader(TestDataset(image_root, alpha_root),
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            drop_last=False)
