import torch

def ColorfulnessMetric(images, reduction="mean"):
    """
    :param images: torch.Tensor [B, 3, H, W] range [0, 1]
    :return: colorfulness score (scale or batch)
    """
    # Convert to [0, 255]
    images = images * 255.0

    R, G, B = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]

    rg = R - G
    yb = 0.5 * (R + G) - B

    rg_mean = rg.mean(dim=(1, 2))
    rg_std = rg.std(dim=(1, 2))
    yb_mean = yb.mean(dim=(1, 2))
    yb_std = yb.std(dim=(1, 2))

    mean_root = torch.sqrt(rg_mean**2 + yb_mean**2)
    std_root = torch.sqrt(rg_std**2 + yb_std**2)

    colorfulness = 0.3 * mean_root + std_root
    if reduction == "mean":
        return colorfulness.mean()
    return colorfulness.sum()