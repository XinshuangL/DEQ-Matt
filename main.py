import torch
import torch.nn.utils as nn_utils
import os
import json
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import tqdm
from dataset import get_train_dataloader, get_test_dataloader
from models.deqmatt_trainer import compute_loss, model_predict
from models.deqmatt import build_model
from models.mdeq_model import get_pretrained_model
from lib.InSPyReNet import InSPyReNet_SwinB

class ClipGrad:
    def __init__(self):
        self.moving_max_grad = None
        self.max_grad = None
        self.moment = 0.999

    def __call__(self, model):
        if self.moving_max_grad == None:
            self.moving_max_grad = nn_utils.clip_grad_norm_(model.parameters(), 1e+2)
            self.max_grad = self.moving_max_grad
        else:
            self.max_grad = nn_utils.clip_grad_norm_(model.parameters(), 2 * self.moving_max_grad)
            self.moving_max_grad = self.moving_max_grad * self.moment + self.max_grad * (1 - self.moment)

class SOD:
    def __init__(self):
        self.sod = InSPyReNet_SwinB(64, False, [384, 384])
        self.sod.load_state_dict(torch.load(('pretrained/Plus_Ultra_LR.pth'), map_location=torch.device('cpu')), strict=True)
        self.sod = self.sod.cuda()
        self.sod.eval()

    def __call__(self, image):
        with torch.no_grad():
            N, C, H, W = image.shape
            image = F.interpolate(image, (384, 384))
            pred = self.sod({'image': image})['pred']
            return F.interpolate(pred, (H, W))

def test_model(model, dataloader, sod, save_folder):
    save_root = f'vis/{save_folder}/'
    os.makedirs(save_root, exist_ok=True)
    MSE = 0
    MAD = 0
    count = 0
    for data in tqdm.tqdm(dataloader):
        data['image'] = data['image'].cuda()
        data['alpha'] = data['alpha'].cuda()
        pred = model_predict(model, data['image'], sod(data['image']))
        N = data['image'].shape[0]
        dif = data['alpha'] - pred
        MSE += torch.mean(dif * dif) * N
        MAD += torch.mean(torch.abs(dif)) * N
        count += N

        for i in range(N):
            cur_name = data['name'][i]
            cur_pred = pred[i][0].cpu().numpy() * 255
            cur_pred[cur_pred > 255] = 255
            cur_pred[cur_pred < 0] = 0
            cv2.imwrite(f'{save_root}/{cur_name}.png', cur_pred.astype(np.uint8))

    return float(MSE / count), float(MAD / count)

def compute_dist_loss(model, model_pretrain, data, train_step=None):
    image, sod_map = data['image'], torch.zeros_like(data['sod_map'], dtype=torch.float32).cuda()

    with torch.no_grad():
        logits_pretrain = model_pretrain(image)

    y = model(image, sod_map, train_step=train_step, return_fearure=True)
    logits = model_pretrain(y, given_y=True)
    return F.mse_loss(logits, logits_pretrain)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str)
    opt = parser.parse_args()
    dataset = opt.dataset

    # load model
    model = build_model()
    model = model.cuda()
    model.train()
    sod = SOD()
    model_pretrain = get_pretrained_model().cuda()
    model_pretrain.eval()

    # load data
    train_fg_root = f'../{dataset}/train/fg/'
    train_alpha_root = f'../{dataset}/train/alpha/'
    train_bg_root = f'../coco/'
    test_image_root = f'../{dataset}/test/merged/'
    test_alpha_root = f'../{dataset}/test/alpha_copy/'
    dataloader = get_train_dataloader(train_fg_root, train_alpha_root, train_bg_root)

    # training settings
    epochs = 100
    warmup_epoch = 10
    init_lr = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - warmup_epoch ) *len(dataloader))
    clip_grad = ClipGrad()

    # train
    for epoch in tqdm.tqdm(range(epochs)):
        for i, data in enumerate(dataloader):
            data['image'] = data['image'].cuda()
            data['alpha'] = data['alpha'].cuda()
            data['trimap'] = data['trimap'].cuda()
            data['sod_map'] = sod(data['image'])

            step = epoch * len(dataloader) + i + 1
            if epoch < warmup_epoch:
                cur_lr = init_lr * step / warmup_epoch / len(dataloader)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_lr
            else:
                lr_scheduler.step()

            optimizer.zero_grad()
            loss = compute_loss(model, data, epoch=epoch, train_step=step)
            loss.backward()
            dist_loss = compute_dist_loss(model, model_pretrain, data, train_step=step)
            dist_loss.backward()
            clip_grad(model)
            optimizer.step()
    
    # save model
    os.makedirs(f'checkpoints/', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/{dataset}_{epochs}.ckpt')

    # test
    test_dataloader = get_test_dataloader(test_image_root, test_alpha_root)
    with torch.no_grad():
        model.eval()
        MSE, MAD = test_model(model, test_dataloader, sod, dataset)
    result = {
        'MSE': MSE,
        'MAD': MAD
    }

    # save result
    os.makedirs(f'results/', exist_ok=True)
    with open(f'results/{dataset}.json', 'w') as f:
        json.dump(result, f, indent=2)
