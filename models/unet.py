import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            DoubleConvBlock(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.downsampling(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConvBlock(in_channels=in_channels, out_channels=out_channels, mid_channels=in_channels//2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x_merged = torch.cat((x2, x1), dim=1)
        return self.conv(x_merged)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ndims=8):
        super().__init__()
        # Encoder (Downsampling with Strided Convolution)
        self.inc = DoubleConvBlock(in_channels=in_channels, out_channels=ndims)

        self.downsampling1 = DownSampling(in_channels=ndims, out_channels=4*ndims)
        self.downsampling2 = DownSampling(in_channels=4*ndims, out_channels=8*ndims)
        self.downsampling3 = DownSampling(in_channels=8*ndims, out_channels=16*ndims)
        self.downsampling4 = DownSampling(in_channels=16*ndims, out_channels=16*ndims)

        # Decoder
        self.upsampling4 = UpSampling(in_channels=32*ndims, out_channels=8*ndims)
        self.upsampling3 = UpSampling(in_channels=16*ndims, out_channels=4*ndims)
        self.upsampling2 = UpSampling(in_channels=8*ndims, out_channels=ndims)
        self.upsampling1 = UpSampling(in_channels=2*ndims, out_channels=ndims)

        # Output
        self.output = nn.Conv2d(in_channels=ndims, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)                   # H x W x ndims
        x2 = self.downsampling1(x1)        # H/2 x W/2 x (4*ndims)
        x3 = self.downsampling2(x2)        # H/4 x W/4 x (8*ndims)
        x4 = self.downsampling3(x3)        # H/8 x W/8 x (16*ndims)
        x5 = self.downsampling4(x4)        # H/16 x W/16 x (16*ndims)

        #Decoder
        x = self.upsampling4(x5, x4)       # H/8 x W/8 x (8*ndims)
        x = self.upsampling3(x, x3)        # H/4 x W/4 x (4*ndims)
        x = self.upsampling2(x, x2)        # H/2 x W/2 x (ndims)
        x = self.upsampling1(x, x1)        # H x W x ndims

        # # Output
        x = self.output(x)                 # H x H x 3
        x = torch.tanh(x)                  # H x H x 3
        return x