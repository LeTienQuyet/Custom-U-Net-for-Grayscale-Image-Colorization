from facenet_pytorch import InceptionResnetV1

import torch
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        inception_restnet = InceptionResnetV1(pretrained="vggface2")
        self.blocks = nn.ModuleList([
            inception_restnet.conv2d_1a.eval(),
            inception_restnet.conv2d_2a.eval(),
            inception_restnet.conv2d_2b.eval(),
            inception_restnet.conv2d_3b.eval(),
            inception_restnet.conv2d_4a.eval(),
            inception_restnet.conv2d_4b.eval()
        ])

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss = 0.0

        x, y = pred, target.detach()
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
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