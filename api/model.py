import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ChessNet() -> nn.Module:
    model: nn.Module = ChessResNet()
    return model


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(channels)
        self.conv2: nn.Conv2d = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        identity: Tensor = x
        out: Tensor = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


class ChessResNet(nn.Module):
    """
    Input: (B, 18, 8, 8)
    Output: logits for 2 classes (lose / win for side-to-move)
    """

    def __init__(
        self,
        channels: int = 128,
        num_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.blocks: nn.Sequential = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        self.head: nn.Sequential = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.drop: nn.Dropout = nn.Dropout(dropout)
        self.fc: nn.Linear = nn.Linear(32 * 8 * 8, 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.flatten(1)
        x = self.drop(x)
        return self.fc(x)
