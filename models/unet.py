import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def _build(self, n, dim, device):
        pos = torch.arange(n, dtype=torch.float, device=device).unsqueeze(1) #[n, 1]
        div = torch.exp(
            torch.arange(0, dim, 2, device=device).float() *
            (-math.log(10000.0) / dim)
        ) # [dim / 2]
        pe =  torch.zeros(n, dim, device=device) # [n, dim]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        return pe

    def forward(self, h, w, device):
        half_dim = self.in_channels // 2

        # Embed for height
        pe_h = self._build(h, half_dim, device)

        # Embed for width
        pe_w = self._build(w, half_dim, device)

        # Combine both
        pe_h = pe_h.unsqueeze(1).expand(-1, w, -1)
        pe_w = pe_w.unsqueeze(0).expand(h, -1, -1)

        pe = torch.cat([pe_h, pe_w], dim=-1)
        return pe.reshape(h * w, self.in_channels).unsqueeze(0)

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        num_heads = max(1, in_channels//64)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, 4 * in_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * in_channels, in_channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        # Self-attention block with Pre-LN
        normalized = self.norm1(x)
        attn_out, _ = self.attn(normalized, normalized, normalized)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x

class BottleNeck(nn.Module):
    def __init__(self, in_channels, num_blocks=1):
        super().__init__()

        self.proj_in = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.proj_out = nn.Linear(in_features=in_channels, out_features=in_channels)

        self.pos_enc = PositionalEncoding(in_channels=in_channels)

        self.blocks = nn.ModuleList([
            TransformerEncoder(in_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)

        # Add poistion encoding
        pe = self.pos_enc(H, W, x.device)
        x_flat = x_flat + pe

        # In projection
        x_flat = self.proj_in(x_flat)

        # N x Self-attention block with Pre-LN
        for block in self.blocks:
            x_flat = block(x_flat)

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
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True)
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
    def __init__(self, in_channels=1, out_channels=3, ndims=16, num_blocks=1):
        super().__init__()

        # Pre-process
        self.pre_process = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ndims, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True)
        )

        # Encoder (Downsampling with Strided Convolution)
        self.encode1 = EncoderBlock(num_features=ndims)

        self.encode2 = EncoderBlock(num_features=2*ndims)

        self.encode3 = EncoderBlock(num_features=4*ndims)

        self.encode4 = EncoderBlock(num_features=8*ndims)

        # Bottle Neck
        self.bottle_neck = BottleNeck(in_channels=16*ndims, num_blocks=num_blocks)

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