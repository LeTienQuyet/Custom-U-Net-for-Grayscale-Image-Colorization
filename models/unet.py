import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockAttention(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        num_heads = max(1, in_channels//64)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        # Self-attention block with Pre-LN
        normalized = self.norm1(x_flat)
        attn_out, _ = self.attn(normalized, normalized, normalized)
        x = x_flat + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.residual = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

    def forward(self, x):
        out = self.double_conv(x)
        out = out + self.residual(x)
        out = self.bn_act(out)
        return out

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            DoubleConvBlock(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.downsampling(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up =  nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        mid_channels = (in_channels + out_channels)//2
        self.conv = DoubleConvBlock(in_channels=in_channels+out_channels, out_channels=out_channels, mid_channels=mid_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x_merged = torch.cat((x2, x1), dim=1)
        return self.conv(x_merged)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ndims=8):
        super().__init__()
        # Encoder (Downsampling with Strided Convolution)
        self.inc = DoubleConvBlock(in_channels=in_channels, out_channels=ndims)

        self.downsampling1 = DownSampling(in_channels=ndims, out_channels=2*ndims)
        self.downsampling2 = DownSampling(in_channels=2*ndims, out_channels=4*ndims)
        self.downsampling3 = DownSampling(in_channels=4*ndims, out_channels=8*ndims)
        self.downsampling4 = DownSampling(in_channels=8*ndims, out_channels=16*ndims)

        self.bottle_neck = BlockAttention(in_channels=16*ndims)

        # Decoder
        self.upsampling4 = UpSampling(in_channels=16*ndims, out_channels=8*ndims)
        self.upsampling3 = UpSampling(in_channels=8*ndims, out_channels=4*ndims)
        self.upsampling2 = UpSampling(in_channels=4*ndims, out_channels=2*ndims)
        self.upsampling1 = UpSampling(in_channels=2*ndims, out_channels=ndims)

        # Output
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=ndims, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)                   # H x W x ndims
        x2 = self.downsampling1(x1)        # H/2 x W/2 x (2*ndims)
        x3 = self.downsampling2(x2)        # H/4 x W/4 x (4*ndims)
        x4 = self.downsampling3(x3)        # H/8 x W/8 x (8*ndims)
        x5 = self.downsampling4(x4)        # H/16 x W/16 x (16*ndims)

        x5 = self.bottle_neck(x5)          # H/16 x W/16 x (16*ndims)

        #Decoder
        x = self.upsampling4(x5, x4)       # H/8 x W/8 x (8*ndims)
        x = self.upsampling3(x, x3)        # H/4 x W/4 x (4*ndims)
        x = self.upsampling2(x, x2)        # H/2 x W/2 x (2*ndims)
        x = self.upsampling1(x, x1)        # H x W x ndims

        # # Output
        x = self.output(x)                 # H x H x 3
        return x