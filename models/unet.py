import torch
import torch.nn as nn

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.downsampling(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.upsampling(x)
    
class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ndims=16):
        super().__init__()
        # Encoder (Downsampling with Strided Convolution)
        self.encode1 = DoubleConvBlock(in_channels=in_channels, out_channels=ndims)
        self.downsampling1 = DownSampling(in_channels=ndims, out_channels=ndims)

        self.encode2 = DoubleConvBlock(in_channels=ndims, out_channels=ndims*2)
        self.downsampling2 = DownSampling(in_channels=ndims*2, out_channels=ndims*2)

        self.encode3 = DoubleConvBlock(in_channels=ndims*2, out_channels=ndims*4)
        self.downsampling3 = DownSampling(in_channels=ndims*4, out_channels=ndims*4)

        self.encode4 = DoubleConvBlock(in_channels=ndims*4, out_channels=ndims*8)
        self.downsampling4 = DownSampling(in_channels=ndims*8, out_channels=ndims*8)

        # Bottleneck
        self.bottle_neck = DoubleConvBlock(in_channels=ndims*8, out_channels=ndims*16)

        # Decoder (Upsampling with Strided Deconvolution)
        self.upsampling4 = UpSampling(in_channels=ndims*16, out_channels=ndims*8)
        self.decode4 = DoubleConvBlock(in_channels=ndims*16, out_channels=ndims*8)

        self.upsampling3 = UpSampling(in_channels=ndims*8, out_channels=ndims*4)
        self.decode3 = DoubleConvBlock(in_channels=ndims*8, out_channels=ndims*4)

        self.upsampling2 = UpSampling(in_channels=ndims*4, out_channels=ndims*2)
        self.decode2 = DoubleConvBlock(in_channels=ndims*4, out_channels=ndims*2)

        self.upsampling1 = UpSampling(in_channels=ndims*2, out_channels=ndims)
        self.decode1 = DoubleConvBlock(in_channels=ndims*2, out_channels=ndims)

        self.output = nn.Conv2d(in_channels=ndims, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encode1(x)             # 256 x 256 x 16
        x = self.downsampling1(enc1)       # 128 x 128 x 16

        enc2 = self.encode2(x)             # 128 x 128 x 32
        x = self.downsampling2(enc2)       # 64 x 64 x 32

        enc3 = self.encode3(x)             # 64 x 64 x 64
        x = self.downsampling3(enc3)       # 32 x 32 x 64

        enc4 = self.encode4(x)             # 32 x 32 x 128
        x = self.downsampling4(enc4)       # 16 x 16 x 128

        # Bottleneck
        x = self.bottle_neck(x)            # 16 x 16 x 256

        # Decoder
        x = self.upsampling4(x)            # 32 x 32 x 128
        x = torch.cat((x, enc4), dim=1)    # 32 x 32 x 256
        x = self.decode4(x)                # 32 x 32 x 128

        x = self.upsampling3(x)            # 64 x 64 x 64
        x = torch.cat((x, enc3), dim=1)    # 64 x 64 x 128
        x = self.decode3(x)                # 64 x 64 x 64

        x = self.upsampling2(x)            # 128 x 128 x 32
        x = torch.cat((x, enc2), dim=1)    # 128 x 128 x 64
        x = self.decode2(x)                # 128 x 128 x 32

        x = self.upsampling1(x)            # 256 x 256 x 16
        x = torch.cat((x, enc1), dim=1)    # 256 x 256 x 32
        x = self.decode1(x)                # 256 x 256 x 16

        # Output
        x = self.output(x)                 # 256 x 256 x 3
        x = torch.tanh(x)                  # 256 x 256 x 3
        return x