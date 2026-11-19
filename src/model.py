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


def knn(x, k):
    """x: (B, C, N) -> indices (B, N, k) of the k nearest neighbours in feature space."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def edge_features(x, k, idx=None):
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k)
    base = torch.arange(B, device=x.device).view(-1, 1, 1) * N
    x_t = x.transpose(2, 1).contiguous()                       # (B, N, C)
    nb = x_t.view(B * N, C)[(idx + base).view(-1)].view(B, N, k, C)
    ctr = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([nb - ctr, ctr], 3).permute(0, 3, 1, 2).contiguous()   # (B, 2C, N, k)


class DGCNN(nn.Module):
    """Dynamic Graph CNN (Wang et al., 2019): EdgeConv layers on k-NN graphs recomputed in feature space,
    so every point feature is built from its *local* neighbourhood, unlike PointNet's per-point MLP."""

    def __init__(self, num_classes=40, k=20, emb=1024):
        super().__init__()
        self.k = k

        def ec(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.c1, self.c2, self.c3, self.c4 = ec(6, 64), ec(128, 64), ec(128, 128), ec(256, 256)
        self.c5 = nn.Sequential(nn.Conv1d(512, emb, 1, bias=False), nn.BatchNorm1d(emb), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Linear(emb * 2, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()                     # (B, 3, N)
        x1 = self.c1(edge_features(x, self.k)).max(-1)[0]
        x2 = self.c2(edge_features(x1, self.k)).max(-1)[0]
        x3 = self.c3(edge_features(x2, self.k)).max(-1)[0]
        x4 = self.c4(edge_features(x3, self.k)).max(-1)[0]
        x = self.c5(torch.cat([x1, x2, x3, x4], 1))
        x = torch.cat([x.max(-1)[0], x.mean(-1)], 1)
        return self.head(x)


def build(name, num_classes=40):
    return {"pointnet": PointNet, "dgcnn": DGCNN}[name](num_classes=num_classes)
