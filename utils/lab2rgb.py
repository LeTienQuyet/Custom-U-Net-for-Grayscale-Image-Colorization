import torch
import kornia.color as K

def lab2rgb(L, ab):
    # Lab from [0, 1] to [-128, 127]
    L = L * 100.0         # L ∈ [0, 100]
    ab = ab * 255.0 - 128.0 # a,b ∈ [-128, 127]

    # Concat L & ab
    lab = torch.cat([L, ab], dim=1)  # [B,3,H,W]

    # Convert to RGB for calculate perceptual loss
    rgb = K.lab_to_rgb(lab)  # [B,3,H,W] in range [0, 1]

    return rgb