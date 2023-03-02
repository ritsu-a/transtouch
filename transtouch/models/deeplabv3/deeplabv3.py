# credit to https://github.com/fregu856/deeplabv3/tree/master/model

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from active_zero2.models.deeplabv3.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from active_zero2.models.deeplabv3.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
        )

        self.num_classes = 2
        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, data_batch):
        # (x has shape (batch_size, 3, h, w))

        # TODO : remove the stupid conv
        x = self.conv1(data_batch["img_l"])
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        pred_dict = {}
        pred_dict["error_segmentation"] = output
        return pred_dict

