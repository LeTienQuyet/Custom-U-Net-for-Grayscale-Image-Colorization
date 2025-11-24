import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, ndf=24):
        super().__init__()

        # ( 128 x 128 x in_channels => 64 x 64 x ndf )
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=True
        )

        # ( 64 x 64 x ndf => 32 x 32 x (2 * ndf) )
        self.conv2 = nn.Conv2d(
            in_channels=ndf, out_channels=2*ndf, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(num_features=2*ndf)

        # ( 32 x 32 x (2 * ndf) => 32 x 32 x (4 * ndf) )
        self.conv3 = nn.Conv2d(
            in_channels=2*ndf, out_channels=4*ndf, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm3 = nn.BatchNorm2d(num_features=4*ndf)

        # ( 32 x 32 x (4 * ndf) => 16 x 16 x (8 * ndf) )
        self.conv4 = nn.Conv2d(
            in_channels=4*ndf, out_channels=8*ndf, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.norm4 = nn.BatchNorm2d(num_features=8*ndf)

        # ( 16 x 16 x (8 * ndf) => 16 x 16 x 1 )
        self.conv5 = nn.Conv2d(
            in_channels=8*ndf, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.norm3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.norm4(self.conv4(x)), 0.2, True)
        x = self.conv5(x)
        return x