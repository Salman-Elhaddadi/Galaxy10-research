import torch
from torch import nn


def conv_bn(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(conv_bn(c, c), conv_bn(c, c))

    def forward(self, x):
        return x + self.block(x)


class GalAxy10(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            conv_bn(32, 64),
            conv_bn(64, 128),
            nn.MaxPool2d(2),
            Residual(128),
            conv_bn(128, 256),
            nn.MaxPool2d(2),
            Residual(256),
            conv_bn(256, 512),
            nn.MaxPool2d(2),
            Residual(512),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        return self.classifier(x)

    