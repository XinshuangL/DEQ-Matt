import os
import sys
import logging
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

sys.path.append("lib/models")
from lib.models.mdeq_core import MDEQNet
sys.path.append("../")

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

from lib.config import config
from lib.config import update_config
update_config(config, 'models/seg_mdeq_SMALL.yaml') #SMALL

class MDEQSegNet(MDEQNet):
    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ Segmentation model with the given hyperparameters
        """
        global BN_MOMENTUM
        super(MDEQSegNet, self).__init__(cfg, img_channel=3, BN_MOMENTUM=BN_MOMENTUM, **kwargs)
        
        # Last layer
        last_inp_channels = np.int32(np.sum(self.num_channels))
        self.last_layer = nn.Sequential(nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1),
                                        nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(last_inp_channels, cfg.DATASET.NUM_CLASSES, cfg.MODEL.EXTRA.FINAL_CONV_KERNEL, 
                                                  stride=1, padding=0))
    
    def segment(self, y):
        """
        Given outputs at multiple resolutions, segment the feature map by predicting the class of each pixel
        """
        # Segmentation Head
        y0_h, y0_w = y[0].size(2), y[0].size(3)
        all_res = [y[0]]
        for i in range(1, self.num_branches):
            all_res.append(F.interpolate(y[i], size=(y0_h, y0_w), mode='bilinear', align_corners=True))

        y = torch.cat(all_res, dim=1)
        all_res = None
        y = self.last_layer(y)
        return y

    def forward(self, x, train_step=-1, given_y=False, **kwargs):
        if given_y:
            return self.segment(x)
        else:
            y, jac_loss, sradius = self._forward(x, train_step, **kwargs)
            return self.segment(y)

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
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

def get_pretrained_model():
    global BN_MOMENTUM
    BN_MOMENTUM = 0.01
    model = MDEQSegNet(config)
    model.init_weights(config.MODEL.PRETRAINED)
    return model
