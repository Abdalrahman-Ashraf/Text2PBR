import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class PBRMapGeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, 2)
        self.dec4 = ConvBlock(base_channels*8 * 2, base_channels*8)
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, 2)
        self.dec3 = ConvBlock(base_channels*4 * 2, base_channels*4)
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, 2)
        self.dec2 = ConvBlock(base_channels*2 * 2, base_channels*2)
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, 2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return torch.sigmoid(self.output(d1))
