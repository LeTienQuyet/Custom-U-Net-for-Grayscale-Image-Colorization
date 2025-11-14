import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        num_heads = max(1, in_channels//64)

        self.proj_in = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.proj_out = nn.Linear(in_features=in_channels, out_features=in_channels)

        self.alpha = nn.Parameter(torch.tensor(1.0))

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

        # In projection
        x_flat = self.proj_in(x_flat)

        # Self-attention block with Pre-LN
        normalized = self.norm1(x_flat)
        attn_out, _ = self.attn(normalized, normalized, normalized)
        x_flat = x_flat + attn_out * self.alpha
        ff_out = self.ff(self.norm2(x_flat))
        x_flat = x_flat + ff_out * self.alpha

        # Out projection
        x_flat = self.proj_out(x_flat)

        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x

class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()

        t = int(abs((math.log2(in_channels) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling → [B, C, 1, 1]
        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        return x * y.expand_as(x)

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
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )

        self.eca = ECA(in_channels=out_channels)

    def forward(self, x):
        out = self.double_conv(x)
        out = self.eca(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.encode = DoubleConvBlock(in_channels=num_features, out_channels=2*num_features)

    def forward(self, x):
        enc = self.encode(x)
        x = self.downsampling(enc)
        return enc, x

class DecoderBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=3, stride=1, padding=1)
        )
        self.decode = DoubleConvBlock(in_channels=2*num_features, out_channels=num_features//2, mid_channels=num_features)

    def forward(self, x, enc):
        x = self.upsampling(x)
        x_merged = torch.cat((x, enc), dim=1)
        return self.decode(x_merged)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, ndims=8):
        super().__init__()

        # Pre-process
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ndims, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        # Encoder (Downsampling with Strided Convolution)
        self.encode1 = EncoderBlock(num_features=ndims)

        self.encode2 = EncoderBlock(num_features=2*ndims)

        self.encode3 = EncoderBlock(num_features=4*ndims)

        self.encode4 = EncoderBlock(num_features=8*ndims)

        # Bottle Neck
        self.bottle_neck = BottleNeck(in_channels=16*ndims)

        # Decoder
        self.decode4 = DecoderBlock(num_features=16*ndims)

        self.decode3 = DecoderBlock(num_features=8*ndims)

        self.decode2 = DecoderBlock(num_features=4*ndims)

        self.decode1 = DecoderBlock(num_features=2*ndims)

        # Post-process
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels=ndims, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Pre-process
        x = self.pre_process(x)         # H x W x ndims

        # Encoder
        enc1, x = self.encode1(x)       # H x W x (2*ndims)      || H/2 x W/2 x (2*ndims)

        enc2, x = self.encode2(x)       # H/2 x W/2 x (4*ndims)  || H/4 x W/4 x (4*ndims)

        enc3, x = self.encode3(x)       # H/4 x W/4 x (8*ndims)  || H/8 x W/8 x (8*ndims)

        enc4, x = self.encode4(x)       # H/8 x W/8 x (16*ndims) || H/16 x W/16 x (16*ndims)

        # Bottle Neck
        x = self.bottle_neck(x)         # H/16 x W/16 x (16*ndims)

        # Decoder
        x = self.decode4(x, enc4)       # H/8 x W/8 x (8*ndims)

        x = self.decode3(x, enc3)       # H/4 x W/4 x (4*ndims)

        x = self.decode2(x, enc2)       # H/2 x W/2 x (2*ndims)

        x = self.decode1(x, enc1)       # H x W x ndims

        # Post-process
        x = self.post_process(x)        # H x W x 3

        return x