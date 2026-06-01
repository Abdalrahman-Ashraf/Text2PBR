import torch
import torch.nn as nn
import torch.nn.utils as utils

class PBRPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def conv_block(in_c, out_c, kernel_size=3, stride=2, normalize=True):
            layers = [
                utils.spectral_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=1)
                )
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_channels, base_channels, normalize=False),
            conv_block(base_channels, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 4),
            conv_block(base_channels * 4, base_channels * 8, stride=1),
            utils.spectral_norm(
                nn.Conv2d(base_channels * 8, 1, kernel_size=3, stride=1, padding=1)
            ) 
        )

    def forward(self, x):
        return self.model(x)
