import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class PBRMapGeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels*2, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels*2, base_channels*4, kernel_size=3)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels*4, base_channels*8, kernel_size=3)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = ConvBlock(base_channels*8, base_channels*16, kernel_size=3)

        # Mid
        self.mid = ConvBlock(base_channels*16, base_channels*32, kernel_size=3)

        # Decoder
        self.upconv5 = nn.ConvTranspose2d(base_channels*32, base_channels*16, kernel_size=1, stride=1)
        self.dec5 = ConvBlock(base_channels*32, base_channels*16, kernel_size=3)

        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels*16, base_channels*8, kernel_size=3)

        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels*8, base_channels*4, kernel_size=3)

        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels*4, base_channels*2, kernel_size=3)

        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels*2, base_channels, kernel_size=3)

        self.pred_map = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        e5 = self.enc5(p4)

        # Mid block
        b = self.mid(e5)

        # Decoder
        up5 = self.upconv5(b)
        cat5 = torch.cat([up5, e5], dim=1)
        d5 = self.dec5(cat5)

        up4 = self.upconv4(d5)
        cat4 = torch.cat([up4, e4], dim=1)
        d4 = self.dec4(cat4)

        up3 = self.upconv3(d4)
        cat3 = torch.cat([up3, e3], dim=1)
        d3 = self.dec3(cat3)

        up2 = self.upconv2(d3)
        cat2 = torch.cat([up2, e2], dim=1)
        d2 = self.dec2(cat2)

        up1 = self.upconv1(d2)
        cat1 = torch.cat([up1, e1], dim=1)
        d1 = self.dec1(cat1)

        return torch.sigmoid(self.pred_map(d1))
