import torch
import torch.nn as nn


class DQN(nn.Module):
    """Compact DQN for grid+scalar Bomberman states."""

    def __init__(self, in_channels: int, grid_size: tuple[int, int], scalar_size: int, n_actions: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size[0], grid_size[1])
            cnn_out_dim = int(self.pool(self.cnn(dummy)).flatten(1).shape[1])

        combined_dim = cnn_out_dim + 16

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, n_actions)

    def forward(self, grid: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        x = self.cnn(grid)
        x = self.pool(x).flatten(1)

        s = self.scalar_fc(scalar)

        x = torch.cat([x, s], dim=1)
        x = self.shared(x)

        value = self.value(x)
        advantage = self.advantage(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)
