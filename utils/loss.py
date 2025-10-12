import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15]):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),
            vgg[4:9].eval(),
            vgg[9:16].eval()
        ])
        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0

        with torch.no_grad():
            x, y = pred.detach(), target.detach()
            for block in self.blocks:
                x = block(x)
                y = block(y)
                loss += self.mse(x, y)

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.percept = PerceptualLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        percept_loss = self.percept(pred, target)
        total_loss = mse_loss + self.alpha * percept_loss
        return mse_loss, percept_loss, total_loss