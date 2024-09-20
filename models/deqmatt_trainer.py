import torch
import torch.nn.functional as F
import cv2
import numpy as np
import random

Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,60)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640,640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0/255.0] = 0
    uncertain_area[pred > 1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = np.array(weight, dtype=np.float32)
    weight = torch.from_numpy(weight).cuda()
    weight = F.interpolate(weight, size=(H,W), mode='nearest')
    return weight

def regression_loss_(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    return torch.mean(regression_loss_(logit, target, loss_type, weight))

def lap_loss(logit, target, gauss_filter, loss_type='l1', weight=None):
    '''
    Based on FBA Matting implementation:
    https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    '''

    def conv_gauss(x, kernel):
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        return x

    def downsample(x):
        return x[:, :, ::2, ::2]

    def upsample(x, kernel):
        N, C, H, W = x.shape
        cc = torch.cat([x, torch.zeros(N, C, H, W).cuda()], dim=3)
        cc = cc.view(N, C, H * 2, W)
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(N, C, W, H * 2).cuda()], dim=3)
        cc = cc.view(N, C, W * 2, H * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return conv_gauss(x_up, kernel=4 * gauss_filter)

    def lap_pyramid(x, kernel, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down, kernel)
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

    def weight_pyramid(x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            down = downsample(current)
            pyr.append(current)
            current = down
        return pyr

    pyr_logit = lap_pyramid(x=logit, kernel=gauss_filter, max_levels=5)
    pyr_target = lap_pyramid(x=target, kernel=gauss_filter, max_levels=5)
    if weight is not None:
        pyr_weight = weight_pyramid(x=weight, max_levels=5)
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2 ** i) for i, A in
                    enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
    else:
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2 ** i) for i, A in
                    enumerate(zip(pyr_logit, pyr_target)))

gauss_filter = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]]).cuda()
gauss_filter /= 256.
gauss_filter = gauss_filter.repeat(1, 1, 1, 1)

def composition_loss(alpha, fg, bg, image, weight, loss_type='l1'):
    merged = fg * alpha + bg * (1 - alpha)
    return regression_loss(merged, image, loss_type=loss_type, weight=weight)

def merge_alpha(alpha_A, alpha_B, weight_A):
    N, C, H, W = alpha_A.shape
    alpha_merge = torch.zeros((N, C, H, W), dtype=torch.float32).cuda()
    alpha_merge += alpha_A
    alpha_merge[weight_A == 0] -= alpha_A[weight_A == 0]
    alpha_merge[weight_A == 0] += alpha_B[weight_A == 0]
    return alpha_merge

def loss_calculation(out, gt, fg, bg, image, epoch, warmup_epoch):
    N, C, H, W = gt.shape
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8, alpha_pred_os16 = out['alpha_os1'], out['alpha_os4'], out['alpha_os8'], out['alpha_os16']

    if epoch < warmup_epoch:
        weight_os16 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
        weight_os8 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
        weight_os4 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
        weight_os1 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()

    elif epoch < warmup_epoch * 3:
        if random.randint(0, 1) == 0:
            weight_os16 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
            weight_os8 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
            weight_os4 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
            weight_os1 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
        else:
            weight_os16 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
            weight_os8 = get_unknown_tensor_from_pred(alpha_pred_os16, rand_width=45, train_mode=True)
            alpha_pred_os8 = merge_alpha(alpha_pred_os8, alpha_pred_os16, weight_os8)
            weight_os4 = get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=30, train_mode=True)
            alpha_pred_os4 = merge_alpha(alpha_pred_os4, alpha_pred_os8, weight_os4)
            weight_os1 = get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=15, train_mode=True)
            alpha_pred_os1 = merge_alpha(alpha_pred_os1, alpha_pred_os4, weight_os1)
    else:
        weight_os16 = torch.ones((N, 1, H, W), dtype=torch.float32).cuda()
        weight_os8 = get_unknown_tensor_from_pred(alpha_pred_os16, rand_width=45, train_mode=True)
        alpha_pred_os8 = merge_alpha(alpha_pred_os8, alpha_pred_os16, weight_os8)
        weight_os4 = get_unknown_tensor_from_pred(alpha_pred_os8, rand_width=30, train_mode=True)
        alpha_pred_os4 = merge_alpha(alpha_pred_os4, alpha_pred_os8, weight_os4)
        weight_os1 = get_unknown_tensor_from_pred(alpha_pred_os4, rand_width=15, train_mode=True)
        alpha_pred_os1 = merge_alpha(alpha_pred_os1, alpha_pred_os4, weight_os1)

    """===== Calculate Loss ====="""
    loss = (regression_loss(alpha_pred_os1, gt, loss_type='l1', weight=weight_os1) * 2 + \
                                regression_loss(alpha_pred_os4, gt, loss_type='l1', weight=weight_os4) * 1 + \
                                regression_loss(alpha_pred_os8, gt, loss_type='l1', weight=weight_os8) * 1 + \
                                regression_loss(alpha_pred_os16, gt, loss_type='l1', weight=weight_os16) * 1) / 5.0

    loss += (lap_loss(logit=alpha_pred_os1, target=gt, gauss_filter=gauss_filter, loss_type='l1', weight=weight_os1) * 2 + \
                                lap_loss(logit=alpha_pred_os4, target=gt, gauss_filter=gauss_filter, loss_type='l1', weight=weight_os4) * 1 + \
                                lap_loss(logit=alpha_pred_os8, target=gt, gauss_filter=gauss_filter, loss_type='l1', weight=weight_os8) * 1 + \
                                lap_loss(logit=alpha_pred_os16, target=gt, gauss_filter=gauss_filter, loss_type='l1', weight=weight_os16) * 1) / 5.0
    
    loss += (composition_loss(alpha_pred_os1, fg, bg, image, weight=weight_os1) * 2 +\
            composition_loss(alpha_pred_os4, fg, bg, image, weight=weight_os4) * 1 +\
            composition_loss(alpha_pred_os8, fg, bg, image, weight=weight_os8) * 1 +\
            composition_loss(alpha_pred_os16, fg, bg, image, weight=weight_os16) * 1) / 5.0

    return loss

def compute_loss(model, data, epoch=100, warmup_epoch=10, train_step=None):
    image, gt, sod_map = data['image'], data['alpha'], data['sod_map']
    fg, bg = data['fg'].cuda(), data['bg'].cuda()
    image = image.cuda()
    sod_map = sod_map.cuda()
    gt = gt.cuda()
    out, jac_loss, sradius = model(image, sod_map, train_step=train_step)
    loss = loss_calculation(out, gt, fg, bg, image, epoch, warmup_epoch)
    return loss

def model_predict(model, image, sod_map):
    image = image.cuda()
    sod_map = sod_map.cuda()
    out, jac_loss, sradius = model(image, sod_map, compute_jac_loss=False, train_step=-1)
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8, alpha_pred_os16 = out['alpha_os1'], out['alpha_os4'], out['alpha_os8'], out['alpha_os16']
    alpha_pred = alpha_pred_os16.clone().detach()
    weight_os8 = get_unknown_tensor_from_pred(alpha_pred, rand_width=45, train_mode=False)
    alpha_pred[weight_os8>0] = alpha_pred_os8[weight_os8>0]
    weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=30, train_mode=False)
    alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
    weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=15, train_mode=False)
    alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
    return alpha_pred
