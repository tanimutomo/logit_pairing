import sys
import torch
from torch import nn


class LeNet(nn.Module):
    """LeNet model (same architecture as logit pairing implementation)
    """
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.filters = 32

        self.features = nn.Sequential(
            *convblock(in_channels=1, out_channels=self.filters,
                       kernel_size=3, padding=1, use_bn=False),
            nn.MaxPool2d(kernel_size=2),
            *convblock(in_channels=self.filters, out_channels=self.filters * 2,
                       kernel_size=3, padding=1, use_bn=False),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.filters*2*7*7, 1024),
            nn.ReLU(True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def convblock(in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
    """
    Returns convolution block
    """
    if use_bn:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ]
    else:
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(True)
        ]
