import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class PBRMapGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 64, 1),
            ConvBlock(64, 128, 3),
            nn.MaxPool2d(2),
            ConvBlock(128, 256, 3),
            nn.MaxPool2d(2),
            ConvBlock(256, 512, 5),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(512, 256, 5),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128, 3),
            ConvBlock(128, 64, 3),
        )

        self.out_normaldx     = nn.Conv2d(64, 3, 1)
        self.out_roughness    = nn.Conv2d(64, 3, 1)
        self.out_displacement = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return {
            "normaldx"     : torch.sigmoid(self.out_normaldx(x)),
            "roughness"    : torch.sigmoid(self.out_roughness(x)),
            "displacement" : torch.sigmoid(self.out_displacement(x))
        }