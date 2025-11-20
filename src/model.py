import torch
import torch.nn as nn


class PairRangeCNN(nn.Module):
    def __init__(self):
        super().__init__()  # initialize nn.Module base class

        # ----- feature extractor: small 2D CNN -----
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            # 2 input channels (two range images),
            # 32 output feature maps, 3x3 conv, keep H,W size with padding

            nn.BatchNorm2d(32),
            # normalize per-channel; helps training stability

            nn.ReLU(inplace=True),
            # non-linearity

            nn.MaxPool2d(2),
            # downsample: H, W -> H/2, W/2  (64x1024 -> 32x512)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # 32 -> 64 channels, still with 3x3 conv

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
            # downsample again: 32x512 -> 16x256

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # a bit deeper: 64 -> 128 channels

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            # global average pooling over H,W
            # result shape: (B, 128, 1, 1)
        )

        # ----- classifier on top of the global feature -----
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            # 128-dim global feature -> 64-d hidden layer

            nn.ReLU(inplace=True),

            nn.Linear(64, 1),
            # final scalar (logit) for "same place" probability

            nn.Sigmoid(),
            # we keep a Sigmoid here because your training loop uses BCELoss
        )

    def forward(self, x):
        # x: (B, 2, H, W)   (two range images stacked as channels)
        x = self.features(x)        # -> (B, 128, 1, 1)
        x = x.view(x.size(0), -1)   # -> (B, 128)
        x = self.classifier(x)      # -> (B, 1) in [0,1]
        return x
