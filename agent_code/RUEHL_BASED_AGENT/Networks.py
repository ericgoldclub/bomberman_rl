import torch
from torch import nn


class DQN(nn.Module):
    """Small dueling DQN with a 17 x 17 receptive field."""

    def __init__(self, board_channels, board_size, scalar_size, n_actions):
        super().__init__()
        self.board = nn.Sequential(
            nn.Conv2d(board_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            board_size = self.board(
                torch.zeros(1, board_channels, *board_size)
            ).shape[1]

        self.scalar = nn.Sequential(nn.Linear(scalar_size, 32), nn.ReLU())
        self.common = nn.Sequential(
            nn.Linear(board_size + 32, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, n_actions)

    def forward(self, board, scalar):
        x = torch.cat((self.board(board), self.scalar(scalar)), dim=1)
        x = self.common(x)
        advantage = self.advantage(x)
        return self.value(x) + advantage - advantage.mean(dim=1, keepdim=True)


# Keep the old class name convenient for notebooks and experiments.
DQN_prev = DQN
