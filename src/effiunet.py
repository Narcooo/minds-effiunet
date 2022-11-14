# -*- coding: utf-8 -*-
"""
@author Majx
@date 2022年11月03日 11:00:53

@describe TODO
"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from src.efficientnet import efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b0, efficientnet_b4,\
efficientnet_b5,efficientnet_b6, efficientnet_b7
from src.unet import UNet

size_dict = {'efficientnet-b0': [432, 296, 152, 80, 35, 32], 'efficientnet-b1': [432, 296, 152, 80, 35, 32],
             'efficientnet-b2': [472, 304, 152, 80, 35, 32], 'efficientnet-b3': [520, 304, 160, 88, 35, 32],
             'efficientnet-b4': [608, 312, 160, 88, 35, 32], 'efficientnet-b5': [688, 320, 168, 88, 35, 32],
             'efficientnet-b6': [776, 328, 168, 96, 35, 32], 'efficientnet-b7': [864, 336, 176, 96, 35, 32]}
encoder_dict = {'efficientnet-b0': efficientnet_b0, 'efficientnet-b1': efficientnet_b1,
             'efficientnet-b2': efficientnet_b2, 'efficientnet-b3': efficientnet_b3,
             'efficientnet-b4': efficientnet_b4, 'efficientnet-b5': efficientnet_b5,
             'efficientnet-b6': efficientnet_b6, 'efficientnet-b7': efficientnet_b7}
# def get_effi(encoder='b0'):

def double_conv(in_ch, out_ch):
    return nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU(),
                              nn.Conv2d(out_ch, out_ch, 3),
                              nn.BatchNorm2d(out_ch), nn.ReLU())

class EfficientUnet(nn.Cell):
    def __init__(self, encoder, num_classes=2):
        super().__init__()

        self.size = size_dict[encoder]
        # self.encoder = efficientnet_b0()
        self.encoder = encoder_dict[encoder]()

        # self.decoder = UNet(n_channels_dict['efficientnet-b1'])
        self.concat1 = P.Concat(axis=1)
        self.concat2 = P.Concat(axis=1)
        self.concat3 = P.Concat(axis=1)
        self.concat4 = P.Concat(axis=1)
        self.upsample1 = nn.ResizeBilinear()
        self.double_conv6 = double_conv(self.size[0], 256)
        self.upsample2 = nn.ResizeBilinear()
        self.double_conv7 = double_conv(self.size[1], 128)
        self.upsample3 = nn.ResizeBilinear()
        self.double_conv8 = double_conv(self.size[2], 64)
        self.upsample4 = nn.ResizeBilinear()
        self.double_conv9 = double_conv(self.size[3], 32)
        self.upsample5 = nn.ResizeBilinear()
        self.final = nn.Conv2d(32, num_classes, 1)
    def construct(self, x):
        x = self.encoder(x)
        feature1 = x[0]
        feature2 = x[1]
        feature3 = x[2]
        feature4 = x[3]
        feature5 = x[4]
        up_feature1 = self.upsample1(feature5, scale_factor=2)
        tmp = self.concat1((feature4, up_feature1))
        tmp = self.double_conv6(tmp)

        up_feature2 = self.upsample2(tmp, scale_factor=2)
        tmp = self.concat2((feature3, up_feature2))
        tmp = self.double_conv7(tmp)

        up_feature3 = self.upsample3(tmp, scale_factor=2)
        tmp = self.concat3((feature2, up_feature3))
        tmp = self.double_conv8(tmp)

        up_feature4 = self.upsample4(tmp, scale_factor=2)
        tmp = self.concat4((feature1, up_feature4))
        tmp = self.double_conv9(tmp)

        up_feature5 = self.upsample5(tmp, scale_factor=2)
        # output = self.final(tmp)
        # return [up_feature3, feature5, feature4, feature3, feature2, feature1, tmp]

        output = self.final(up_feature5)
        return output

if __name__ == '__main__':
    t = Tensor(np.ones([1, 3, 1312, 2080]), ms.float32)
    model = EfficientUnet(encoder='efficientnet-b7', num_classes=1)
    a = model(t)
    print(a)