import torch
import torch.nn as nn


class EEGNet(nn.Module):

    def __init__(
        self,
        channels,
        time_points,
        num_classes=2
    ):

        super().__init__()

        # -----------------------------------------
        # Temporal convolution
        # -----------------------------------------

        self.temporal = nn.Conv2d(
            1,
            16,
            kernel_size=(1, 32),
            padding=(0, 16),
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(
            16
        )

        # -----------------------------------------
        # Spatial convolution
        # -----------------------------------------

        self.spatial = nn.Conv2d(
            16,
            32,
            kernel_size=(channels, 1),
            groups=16,
            bias=False
        )

        self.batch_norm2 = nn.BatchNorm2d(
            32
        )

        self.activation = nn.ELU()

        # -----------------------------------------
        # Pooling and dropout
        # -----------------------------------------

        self.pool = nn.AvgPool2d(
            kernel_size=(1, 4)
        )

        self.dropout = nn.Dropout(
            0.5
        )

        # -----------------------------------------
        # Determine classifier size automatically
        # -----------------------------------------

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                channels,
                time_points
            )

            output = self.forward_features(
                dummy
            )

            flattened = output.view(
                1,
                -1
            ).shape[1]

        self.classifier = nn.Linear(
            flattened,
            num_classes
        )

    def forward_features(self, x):

        x = self.temporal(x)

        x = self.batch_norm1(x)

        x = self.spatial(x)

        x = self.batch_norm2(x)

        x = self.activation(x)

        x = self.pool(x)

        x = self.dropout(x)

        return x

    def forward(self, x):

        x = self.forward_features(x)

        x = x.flatten(
            start_dim=1
        )

        x = self.classifier(x)

        return x