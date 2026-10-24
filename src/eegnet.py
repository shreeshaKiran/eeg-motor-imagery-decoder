"""EEGNet-8,2 (Lawhern et al., 2018), adapted to 160 Hz input.

Input: (batch, 1, channels, time). Settings are fixed in CLAUDE.md before any LOSO result.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):

    def __init__(
        self,
        channels,
        time_points,
        num_classes=2,
        F1=8,
        D=2,
        F2=16,
        kernel_length=80,       # half the sampling rate (160 Hz), as in the paper (64 at 128 Hz)
        dropout=0.25,           # cross-subject setting from the paper
    ):

        super().__init__()

        # Block 1: temporal conv -> depthwise spatial conv -> pool
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(((kernel_length - 1) // 2, kernel_length // 2, 0, 0)),   # "same" length
            nn.Conv2d(1, F1, (1, kernel_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv (depthwise temporal + pointwise) -> pool
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),                                         # "same" length
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Determine classifier size automatically
        with torch.no_grad():
            flattened = self.forward_features(
                torch.zeros(1, 1, channels, time_points)
            ).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(flattened, num_classes)

    def forward_features(self, x):
        return self.block2(self.block1(x))

    def forward(self, x):
        return self.classifier(self.forward_features(x).flatten(start_dim=1))

    @torch.no_grad()
    def apply_max_norm(self):
        """Max-norm constraints from the paper: spatial filters 1.0, dense layer 0.25."""
        spatial = self.block1[3].weight
        spatial.data = torch.renorm(spatial.data, p=2, dim=0, maxnorm=1.0)
        self.classifier.weight.data = torch.renorm(
            self.classifier.weight.data, p=2, dim=0, maxnorm=0.25
        )
