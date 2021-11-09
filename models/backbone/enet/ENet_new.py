import torch
import torch.nn as nn


class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    padding=0)

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)

        self.prelu = nn.PReLU(16)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)
        side = self.maxpool(x)
        x = torch.cat((main, side), dim=1)
        x = self.prelu(x)
        return x


class RDDNeck(nn.Module):

    def __init__(self, dilation, in_channels, out_channels,
                 down_flag, relu=False, projection_ratio=4, p=0.1):
        super().__init__()

        # Define class variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2,
                                    stride=2,
                                    padding=0, return_indices=True)
        # self.maxpool = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dropout = nn.Dropout2d(p=p)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.reduced_depth,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False,
                               dilation=1)

        self.prelu1 = activation

        self.conv2 = nn.Conv2d(in_channels=self.reduced_depth,
                               out_channels=self.reduced_depth,
                               kernel_size=3,
                               stride=self.stride,
                               padding=self.dilation,
                               bias=True,
                               dilation=self.dilation)

        self.prelu2 = activation

        self.conv3 = nn.Conv2d(in_channels=self.reduced_depth,
                               out_channels=self.out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False,
                               dilation=1)

        self.prelu3 = activation

        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):

        bs = x.size()[0]
        x_copy = x

        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        x = self.dropout(x)

        # Main Branch
        if self.down_flag:
            x_copy, indices = self.maxpool(x_copy)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                extras = extras.cuda()
            x_copy = torch.cat((x_copy, extras), dim=1)

        # Sum of main and side branches
        x = x + x_copy
        x = self.prelu3(x)

        if self.down_flag:
            return x, indices
        else:
            return x


class UBNeck(nn.Module):
    def __init__(self, in_channels, out_channels, relu=False, projection_ratio=4):
        super().__init__()
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # self.unpool = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.unpool = nn.Upsample(scale_factor=2)
        # self.unpool = nn.MaxUnpool2d(kernel_size = 2,
        #                              stride = 2)
        # self.unpool = nn.MaxUnpool2d(kernel_size=2,
        #                              stride=2)

        self.main_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=1)

        self.dropout = nn.Dropout2d(p=0.1)

        self.convt1 = nn.ConvTranspose2d(in_channels=self.in_channels,
                                         out_channels=self.reduced_depth,
                                         kernel_size=1,
                                         padding=0,
                                         bias=False)

        self.prelu1 = activation

        self.convt2 = nn.ConvTranspose2d(in_channels=self.reduced_depth,
                                         out_channels=self.reduced_depth,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1,
                                         bias=False)

        self.prelu2 = activation

        self.convt3 = nn.ConvTranspose2d(in_channels=self.reduced_depth,
                                         out_channels=self.out_channels,
                                         kernel_size=1,
                                         padding=0,
                                         bias=False)

        self.prelu3 = activation

        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x, indices):
        x_copy = x
        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)
        x = self.convt2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)
        x = self.convt3(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        # Main Branch
        x_copy = self.main_conv(x_copy)
        # x_copy = self.unpool(x_copy, indices, output_size=x.size())
        # x_copy = self.unpool(x_copy)
        x_copy = self.unpool(x_copy)
        # Concat
        x = x + x_copy
        x = self.prelu3(x)
        return x


