import torch
import torch.nn as nn

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.downsampling(x)
    
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.upsampling(x)

class BottleneckAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)

        # Self-attention block with Pre-LN
        normalized = self.norm1(x_flat)
        attn_out, _ = self.attn(normalized, normalized, normalized)
        x = x_flat + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

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
    def __init__(self, in_channels=1, out_channels=3, ndims=8, dropout=0.1):
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
        self.bottle_neck = nn.Sequential(
            DoubleConvBlock(in_channels=8*ndims, out_channels=16*ndims),
            BottleneckAttention(in_channels=16*ndims, num_heads=ndims//2, dropout=dropout)
        )

        # Decoder (Upsampling with Strided Deconvolution)
        self.upsampling4 = UpSampling(in_channels=ndims*16, out_channels=ndims*8)
        self.decode4 = nn.Sequential(
            DoubleConvBlock(in_channels=ndims*16, out_channels=ndims*8),
            nn.Dropout(dropout)
        )

        self.upsampling3 = UpSampling(in_channels=ndims*8, out_channels=ndims*4)
        self.decode3 = nn.Sequential(
            DoubleConvBlock(in_channels=ndims*8, out_channels=ndims*4),
            nn.Dropout(dropout)
        )

        self.upsampling2 = UpSampling(in_channels=ndims*4, out_channels=ndims*2)
        self.decode2 = nn.Sequential(
            DoubleConvBlock(in_channels=ndims*4, out_channels=ndims*2),
            nn.Dropout(dropout)
        )

        self.upsampling1 = UpSampling(in_channels=ndims*2, out_channels=ndims)
        self.decode1 = DoubleConvBlock(in_channels=ndims*2, out_channels=ndims)

        self.output = nn.Conv2d(in_channels=ndims, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encode1(x)             # H x W x ndims
        x = self.downsampling1(enc1)       # H/2 x W/2 x ndims

        enc2 = self.encode2(x)             # H/2 x H/2 x (ndims * 2)
        x = self.downsampling2(enc2)       # H/4 x H/4 x (ndims * 2)

        enc3 = self.encode3(x)             # H/4 x H/4 x (ndims * 4)
        x = self.downsampling3(enc3)       # H/8 x H/8 x (ndims * 4)

        enc4 = self.encode4(x)             # H/8 x H/8 x (ndims * 8)
        x = self.downsampling4(enc4)       # H/16 x H/16 x (ndims * 8)

        # Bottleneck
        x = self.bottle_neck(x)            # H/16 x H/16 x (ndims * 16)

        # Decoder
        x = self.upsampling4(x)            # H/8 x H/8 x (ndims * 8)
        x = torch.cat((x, enc4), dim=1)    # H/8 x H/8 x (ndims * 16)
        x = self.decode4(x)                # H/8 x H/8 x (ndims * 8)

        x = self.upsampling3(x)            # H/4 x H/4 x (ndims * 4)
        x = torch.cat((x, enc3), dim=1)    # H/4 x H/4 x (ndims * 8)
        x = self.decode3(x)                # H/4 x H/4 x (ndims * 4)

        x = self.upsampling2(x)            # H/2 x H/2 x (ndims * 2)
        x = torch.cat((x, enc2), dim=1)    # H/2  x H/2  x (ndims * 4)
        x = self.decode2(x)                # H/2  x H/2  x (ndims * 2)

        x = self.upsampling1(x)            # H x H x ndims
        x = torch.cat((x, enc1), dim=1)    # H x H x (ndims * 2)
        x = self.decode1(x)                # H x H x ndims

        # Output
        x = self.output(x)                 # H x H x 3
        x = torch.tanh(x)                  # H x H x 3
        return x