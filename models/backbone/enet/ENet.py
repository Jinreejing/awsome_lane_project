##################################################################
# Reproducing the paper                                          #
# ENet - Real Time Semantic Segmentation                         #
# Paper: https://arxiv.org/pdf/1606.02147.pdf                    #
#                                                                #
# Copyright (c) 2019                                             #
# Authors: @iArunava <iarunavaofficial@gmail.com>                #
#          @AvivSham <mista2311@gmail.com>                       #
#                                                                #
# License: BSD License 3.0                                       #
#                                                                #
# The Code in this file is distributed for free                  #
# usage and modification with proper credits                     #
# directing back to this repository.                             #
##################################################################

import torch
import torch.nn as nn
from models.backbone.enet.InitialBlock import InitialBlock
from models.backbone.enet.RDDNeck import RDDNeck
from models.backbone.enet.UBNeck import UBNeck
from models.backbone.enet.ASNeck import ASNeck


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ENet(nn.Module):
    def __init__(self, heads):
        super().__init__()

        # The initial block
        self.init = InitialBlock()

        # The first bottleneck
        self.b10 = RDDNeck(dilation=1,
                           in_channels=16,
                           out_channels=64,
                           down_flag=True,
                           p=0.01)

        self.b11 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        # self.b12 = RDDNeck(dilation=1,
        #                    in_channels=64,
        #                    out_channels=64,
        #                    down_flag=False,
        #                    p=0.01)
        #
        # self.b13 = RDDNeck(dilation=1,
        #                    in_channels=64,
        #                    out_channels=64,
        #                    down_flag=False,
        #                    p=0.01)

        self.b14 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           p=0.01)

        # The second bottleneck
        self.b20 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=128,
                           down_flag=True)

        self.b21 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b22 = RDDNeck(dilation=2,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b23_As = ASNeck(in_channels=128,
                          out_channels=128)

        self.b24 = RDDNeck(dilation=4,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b25 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b26 = RDDNeck(dilation=8,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b27_As = ASNeck(in_channels=128,
                          out_channels=128)

        self.b28 = RDDNeck(dilation=16,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        # The third bottleneck
        self.b31 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b32 = RDDNeck(dilation=2,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b33_As = ASNeck(in_channels=128,
                          out_channels=128)

        self.b34 = RDDNeck(dilation=4,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b35 = RDDNeck(dilation=1,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b36 = RDDNeck(dilation=8,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        self.b37_As = ASNeck(in_channels=128,
                          out_channels=128)

        self.b38 = RDDNeck(dilation=16,
                           in_channels=128,
                           out_channels=128,
                           down_flag=False)

        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=128,
                          out_channels=64,
                          relu=True)

        self.b41 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           relu=True)

        self.b42 = RDDNeck(dilation=1,
                           in_channels=64,
                           out_channels=64,
                           down_flag=False,
                           relu=True)

        # # The fifth bottleneck
        # self.b50 = UBNeck(in_channels=64, 
        #                   out_channels=16, 
        #                   relu=True)

        # self.b50 = RDDNeck(dilation=1, 
        #                    in_channels=64, 
        #                    out_channels=16, 
        #                    down_flag=False, 
        #                    relu=True)

        # self.b51 = RDDNeck(dilation=1, 
        #                    in_channels=16, 
        #                    out_channels=16, 
        #                    down_flag=False, 
        #                    relu=True)

        # # Final ConvTranspose Layer
        # self.fullconv = nn.ConvTranspose2d(in_channels=16, 
        #                                    out_channels=self.C, 
        #                                    kernel_size=3, 
        #                                    stride=2, 
        #                                    padding=1, 
        #                                    output_padding=1,
        #                                    bias=False)

        # new additions for LaneAF
        self.heads = heads
        final_kernel = 1
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                RDDNeck(dilation=1,
                        in_channels=64,
                        out_channels=64,
                        down_flag=False,
                        relu=True),
                RDDNeck(dilation=1,
                        in_channels=64,
                        out_channels=64,
                        down_flag=False,
                        relu=True),
                nn.Conv2d(64, classes,
                          kernel_size=final_kernel, stride=1,
                          padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # 上采样
        self.unpool = nn.Upsample(scale_factor=2)

    def forward(self, x):

        # The initial block
        x = self.init(x)

        # The first bottleneck
        x, i1 = self.b10(x) # down
        x = self.b11(x) # 正常res 无变化
        # x = self.b12(x) # 无变化
        # x = self.b13(x) # 无变化
        x = self.b14(x) # 正常res 无变化

        # The second bottleneck
        x, i2 = self.b20(x) # 1x128x45x80
        x = self.b21(x) # 无变化 RDDNeck
        x = self.b22(x) # 无变化 RDDNeck dilation = 2
        x = self.b23_As(x) # 无变化 ASNeck 1x5 5x1
        x = self.b24(x) # 无变化 RDDNeck dilation = 4
        x = self.b25(x) # 无变化 RDDNeck
        x = self.b26(x) # 无变化 RDDNeck dilation = 8
        x = self.b27_As(x) # 无变化 ASNeck 1x5 5x1
        x = self.b28(x) # 无变化 RDDNeck dilation = 16

        # The third bottleneck
        x = self.b31(x)# 无变化 RDDNeck
        x = self.b32(x)# 无变化 RDDNeck dilation = 2
        x = self.b33_As(x)# 无变化 ASNeck
        x = self.b34(x)# 无变化  RDDNeck dilation = 4
        x = self.b35(x)# 无变化  RDDNeck dilation = 1
        x = self.b36(x)# 无变化  RDDNeck dilation = 8
        x = self.b37_As(x)# 无变化 ASNeck
        x = self.b38(x)# 无变化  RDDNeck dilation = 16

        # The fourth bottleneck
        x = self.unpool(x)
        # x = self.b40(x, i2) # unpool和b40选择一个就好
        # x = self.b41(x) # RDDNeck 无变化
        out = self.b42(x) # RDDNeck 无变化

        # LaneAF heads
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(out)
        return [z]


if __name__ == '__main__':
    tensor = torch.randn((1, 3, 360, 640), dtype=torch.float32).cuda()
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = ENet(heads=heads).cuda()
    out = model(tensor)


