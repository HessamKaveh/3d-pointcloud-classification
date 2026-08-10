import torch
import torch.nn as nn


class PointNet(nn.Module):
    """
    PointNet-style classifier for unordered XYZ point clouds.

    Input:
        x: (B, N, 3)

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, num_classes=40):
        super().__init__()

        # Shared point-wise MLP implemented as 1x1 convolutions.
        self.features = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(
                f"Expected input shape (B, N, 3), got {tuple(x.shape)}"
            )

        # Conv1d expects (B, C, N), so XYZ becomes the channel dimension.
        x = x.transpose(1, 2).contiguous()  # (B, 3, N)
        x = self.features(x)                # (B, 1024, N)

        # Symmetric aggregation makes the representation permutation-invariant.
        x = torch.max(x, dim=2).values      # (B, 1024)

        return self.classifier(x)           # (B, num_classes)
