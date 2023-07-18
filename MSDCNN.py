import os
import torch.nn as nn
import torch
from torch.nn.functional import interpolate

import matplotlib.pyplot as plt


class MSDCNNmodel(nn.Module):
    def __init__(self, scale, **kwargs):
        super(MSDCNNmodel, self).__init__()
        self.mslr_mean = kwargs.get('mslr_mean')
        self.mslr_std = kwargs.get('mslr_std')
        self.pan_mean = kwargs.get('pan_mean')
        self.pan_std = kwargs.get('pan_std')

        self.conv_1 = nn.Conv2d(in_channels=5,
                                out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(
            in_channels=32, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        # (input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False)
        self.interpolate = interpolate

        self.scale = scale

    def forward(self, pan, mslr):
        # channel-wise normalization
        '''pan = (pan - self.pan_mean) / self.pan_std
        mslr = (mslr - self.mslr_mean) / self.mslr_std'''

        # torch.nn.functional.interpolate(x, scale_factor=cfg.scale, mode='bicubic', align_corners=True)#
        mslr = self.interpolate(mslr, scale_factor=self.scale, mode='bicubic')
        mssr = torch.cat([mslr, pan], -3)
        mssr = self.relu(self.conv_1(mssr))
        mssr = self.relu(self.conv_2(mssr))
        mssr = self.conv_3(mssr)

        # channel-wise denormalization
        '''mssr = mssr * self.mslr_std + self.mslr_mean'''

        return mssr


if __name__ == "__main__":
    pan = torch.randn(1, 1, 256, 256)
    lr = torch.randn(1, 4, 64, 64)
    pnn = PNNmodel(4)
    print(pnn(pan, lr).shape)
