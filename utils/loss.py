from facenet_pytorch import InceptionResnetV1

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        inception_restnet = InceptionResnetV1(pretrained="vggface2")
        self.blocks = nn.ModuleList([
            inception_restnet.conv2d_1a.eval(),
            inception_restnet.conv2d_2a.eval(),
            inception_restnet.conv2d_2b.eval(),
            inception_restnet.maxpool_3a.eval(),
            inception_restnet.conv2d_3b.eval(),
            inception_restnet.conv2d_4a.eval(),
            inception_restnet.conv2d_4b.eval(),
        ])

        for block in self.blocks:
            for p in block.parameters():
                p.requires_grad = False

        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = F.interpolate(pred, size=(160, 160), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(160, 160), mode="bilinear", align_corners=False)

        loss = 0.0
        x, y = pred, target.detach()
        for block in self.blocks:
            x = block(x)
            with torch.no_grad():
                y = block(y)
            loss += self.mse(x, y)

        return loss