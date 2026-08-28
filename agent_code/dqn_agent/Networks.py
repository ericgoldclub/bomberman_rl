import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_size: tuple[int, int],
        scalar_size: int,
        n_actions: int
    ):
        super().__init__()

        # CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Determine CNN output dimension automatically
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                in_channels,
                grid_size[0],
                grid_size[1]
            )

            cnn_out_dim = self.cnn(dummy).flatten(1).shape[1]

        # Scalar MLP
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # CNN + scalar features
        combined_dim = cnn_out_dim + 32

        # Shared network
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Dueling heads
        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, n_actions)

    def forward(self, grid, scalar):

        # CNN
        x = self.cnn(grid)
        x = torch.flatten(x, 1)

        # Scalar features
        s = self.scalar_fc(scalar)

        # Combine
        x = torch.cat([x, s], dim=1)

        # Shared network
        x = self.shared(x)

        # Dueling heads
        value = self.value(x)
        advantage = self.advantage(x)

        Q = value + advantage - advantage.mean(
            dim=1,
            keepdim=True
        )

        return Q


    def forward(self, grid : torch.Tensor, scalar : torch.Tensor) -> torch.Tensor:
        # grid: (batch_size, in_channels, H, W)
        # scalar: (batch_size, scalar_size)

        x = self.cnn(grid)  # (batch_size, 128, H', W')

        x = torch.flatten(x, 1)

        s = self.scalar_fc(scalar)  # (batch_size, 32)

        x = torch.cat([x,s], dim=1)  # (batch_size, combined_dim)

        x = self.shared(x)

        value = self.value(x)  # (batch_size, 1)
        advantage = self.advantage(x)  # (batch_size, n_actions)

        Q = value + advantage - advantage.mean(dim=1, keepdim=True)  # (batch_size, n_actions)

        return Q



class DQN_deep(nn.Module):
    '''
    The model combines a small CNN for grid channels with a MLP for scalar features,
    such as bomb availability, target distances, and remaining time.
    The outputs are Q-values for each action.
    '''
    def __init__(self, in_channels : int, grid_size: tuple[int, int] , scalar_size : int, n_actions : int):
        super().__init__()
        # CNN for grid channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_size[0], grid_size[1])
            cnn_out_dim = int(self.cnn(dummy).flatten(1).shape[1])


        # MLP for scalar features
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Final MLP to combine CNN and scalar features
        combined_dim = cnn_out_dim + 32

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64), # output Q-values for each action
            nn.ReLU(),
        )

        self.value = nn.Linear(64, 1)  # V(s)
        self.advantage = nn.Linear(64, n_actions)  # A(s,a)


    def forward(self, grid : torch.Tensor, scalar : torch.Tensor) -> torch.Tensor:
        # grid: (batch_size, in_channels, H, W)
        # scalar: (batch_size, scalar_size)

        x = self.cnn(grid)  # (batch_size, 128, H', W')

        x = torch.flatten(x, 1)

        s = self.scalar_fc(scalar)  # (batch_size, 32)

        x = torch.cat([x,s], dim=1)  # (batch_size, combined_dim)

        x = self.shared(x)

        value = self.value(x)  # (batch_size, 1)
        advantage = self.advantage(x)  # (batch_size, n_actions)

        Q = value + advantage - advantage.mean(dim=1, keepdim=True)  # (batch_size, n_actions)

        return Q
