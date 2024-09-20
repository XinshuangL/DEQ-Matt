import os
import sys
import logging
from termcolor import colored
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from   models.ops import GuidedCxtAtten

sys.path.append("lib/models")
from lib.models.mdeq_core import MDEQNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

from lib.config import config
from lib.config import update_config
update_config(config, 'models/seg_mdeq_SMALL.yaml')

class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)
        
    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dIBNormRelu(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
        self.conv2_wo_activation = Conv2dIBNormRelu(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2, with_relu=False)
        if in_channel == out_channel:
            self.upsample = None
        else:
            self.upsample = Conv2dIBNormRelu(in_channel, out_channel, 1, stride=1, padding=0, with_relu=False, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2_wo_activation(out)
        if not self.upsample == None:
            identity = self.upsample(identity)
        out += identity
        out = F.leaky_relu(out)
        return out

def make_layer(in_channel, out_channel, kernel_size, block_num):
    layers = [BasicBlock(in_channel, out_channel, kernel_size)]
    for _ in range(1, block_num):
        layers.append(BasicBlock(out_channel, out_channel, kernel_size))
    return nn.Sequential(*layers)

class Layer1(nn.Module):
    def __init__(self, channels, img_channel=3):
        super(Layer1, self).__init__()

        self.se_block = SEBlock(channels[16], channels[16], reduction=2)
        self.conv16 = make_layer(channels[16] + img_channel, channels[16], 5, 2)
        self.out = nn.Sequential(
            Conv2dIBNormRelu(channels[16], 32, kernel_size=3, stride=1, padding=1),
            Conv2dIBNormRelu(32, 1, kernel_size=3, stride=1, padding=1, with_ibn=False, with_relu=False),
            nn.Sigmoid()
        )

    def forward(self, img, x16):
        img16 = F.interpolate(img, scale_factor=1/16, mode='bilinear', align_corners=False)
        f16 = self.se_block(x16)
        f16 = torch.cat([f16, img16], dim=1)
        f16 = self.conv16(f16)
        pred16 = self.out(f16)
        return pred16, f16

class Layer2(nn.Module):
    def __init__(self, channels, img_channel=3):
        super(Layer2, self).__init__()
        self.conv16 = Conv2dIBNormRelu(channels[16], channels[8], 3, stride=1, padding=1)

        self.conv8 = make_layer(channels[8] + img_channel, channels[8], 3, 3)

        self.guidance_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(img_channel, 16, kernel_size=3, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(16),
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, channels[4], kernel_size=3, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(channels[4]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[4], channels[8], kernel_size=3, padding=0, stride=2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(channels[8])
        )
        self.gca = GuidedCxtAtten(channels[8], channels[8])
        
        self.out = nn.Sequential(
            Conv2dIBNormRelu(channels[8], 1, kernel_size=3, stride=1, padding=1, with_ibn=False, with_relu=False),
            nn.Sigmoid()
        )

    def forward(self, img, x8, f16_1):
        img8 = F.interpolate(img, scale_factor=1/8, mode='bilinear', align_corners=False)
        f16 = self.conv16(f16_1)
        f8 = F.interpolate(f16, scale_factor=2, mode='bilinear', align_corners=False)
        f8 = torch.cat([f8 + x8, img8], dim=1)
        f8 = self.conv8(f8)

        img_fea = self.guidance_head(img)
        f8 = self.gca(img_fea, f8)

        pred8 = self.out(f8)
        return pred8, f8

class Layer3(nn.Module):
    def __init__(self, channels, img_channel=3):
        super(Layer3, self).__init__()

        self.conv8 = Conv2dIBNormRelu(channels[8], channels[4], 3, stride=1, padding=1)

        self.conv4 = make_layer(channels[4] + img_channel, channels[4], 3, 3)

        self.out = nn.Sequential(
            Conv2dIBNormRelu(channels[4], 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
            nn.Sigmoid()
        )

    def forward(self, img, x4, f8_2):
        img4 = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)

        f8 = self.conv8(f8_2)
        f4 = F.interpolate(f8, scale_factor=2, mode='bilinear', align_corners=False)
        f4 = torch.cat([f4 + x4, img4], dim=1)
        f4 = self.conv4(f4)
        pred4 = self.out(f4)
        return pred4, f4

class Layer4(nn.Module):
    def __init__(self, channels, img_channel=3):
        super(Layer4, self).__init__()
        
        self.conv4 = Conv2dIBNormRelu(channels[4], channels[2], 3, stride=1, padding=1)

        self.conv2 = make_layer(channels[2] + img_channel, channels[2], 3, 2)

        self.out = nn.Sequential(
            Conv2dIBNormRelu(channels[2] + img_channel, 32, 3, stride=1, padding=1),
            Conv2dIBNormRelu(32, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
            nn.Sigmoid()
        )

    def forward(self, img, x2, f4_3):
        img2 = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)

        f4 = self.conv4(f4_3)
        f2 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        f2 = torch.cat([f2 + x2, img2], dim=1)
        f2 = self.conv2(f2)
        f1 = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=False)
        f1 = torch.cat([f1, img], dim=1)
        return self.out(f1)

class DEQMatt(MDEQNet):
    def __init__(self, cfg, **kwargs):
        global BN_MOMENTUM
        super(DEQMatt, self).__init__(cfg, BN_MOMENTUM=BN_MOMENTUM, **kwargs)

        channels = {
            2: 32,
            4: 64,
            8: 128,
            16: 256
        }

        self.layer1 = Layer1(channels, 4)
        self.layer2 = Layer2(channels, 4)
        self.layer3 = Layer3(channels, 3)
        self.layer4 = Layer4(channels, 3)

    def forward(self, img, sod_map, train_step=0, return_fearure=False, **kwargs):
        img_sod = torch.cat([img, sod_map], dim=1)
        y, jac_loss, sradius = self._forward(img_sod, train_step, **kwargs)

        x2 = y[0]
        x4 = y[1]
        x8 = y[2]
        x16 = y[3]

        if return_fearure:
            return y

        pred16, f16 = self.layer1(img_sod, x16)
        pred8, f8 = self.layer2(img_sod, x8, f16)
        pred4, f4 = self.layer3(img, x4, f8)
        pred = self.layer4(img, x2, f4)

        pred16 = F.interpolate(pred16, scale_factor=16, mode='bilinear', align_corners=False)
        pred8 = F.interpolate(pred8, scale_factor=8, mode='bilinear', align_corners=False)
        pred4 = F.interpolate(pred4, scale_factor=4, mode='bilinear', align_corners=False)
        
        return {'alpha_os16': pred16, 'alpha_os8': pred8, 'alpha_os4': pred4, 'alpha_os1': pred}, jac_loss, sradius
    
    def init_weights(self, pretrained=''):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info(f'=> init weights from normal distribution. PRETRAINED={pretrained}')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            # remove model.
            pretrained_dict_tmp = {}
            for k, v in pretrained_dict.items():
                if 'model.' in k:
                    k = k[6:]
                pretrained_dict_tmp[k] = v
            pretrained_dict = pretrained_dict_tmp

            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            
            # Just verification...
            diff_modules = set()
            for k in pretrained_dict.keys():
                if k not in model_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In ImageNet MDEQ but not Cityscapes MDEQ: {sorted(list(diff_modules))}", "red"))
            diff_modules = set()
            for k in model_dict.keys():
                if k not in pretrained_dict.keys():
                    diff_modules.add(k.split(".")[0])
            print(colored(f"In Cityscapes MDEQ but not ImageNet MDEQ: {sorted(list(diff_modules))}", "green"))
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            tmp_weight = torch.zeros((32, 4, 3, 3), dtype=torch.float32)
            tmp_weight[:, :3] = pretrained_dict['downsample.0.weight']
            pretrained_dict['downsample.0.weight'] = tmp_weight
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print(pretrained)
            raise NotImplementedError
        
def build_model():
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = DEQMatt(config)
    model.init_weights(config.MODEL.PRETRAINED)
    return model
