import torch
import torch.nn as nn

import torch
import torch.nn as nn


class DQNResidualBlock(nn.Module):
    """Residual spatial-processing block that preserves grid resolution."""

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
    ):
        super().__init__()

        padding = dilation

        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            ),
        )

        self.activation = nn.ReLU()

    def forward(
        self,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        residual = grid
        grid = self.layers(grid)
        grid = grid + residual
        return self.activation(grid)


class DQN_improved(nn.Module):
    """
    Full-resolution residual DQN for Bomberman.

    The network preserves exact 17x17 spatial positions while using residual
    and dilated convolutions to reason over larger parts of the board.

    Grid branch:
        input -> 32 channels
        local residual block
        dilated residual block
        local residual block
        32 -> 8 channel compression
        flatten

    Scalar branch:
        scalar_size -> 64 -> 64

    Fusion:
        spatial + scalar -> 256 -> 128

    Dueling output:
        separate value and advantage streams
    """

    def __init__(
        self,
        in_channels: int,
        grid_size: tuple[int, int],
        scalar_size: int,
        n_actions: int,
    ):
        super().__init__()

        self.n_actions = n_actions

        # Initial local feature extraction.
        self.grid_stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        # Local patterns such as walls, crates, bombs and escape tiles.
        self.local_block_1 = DQNResidualBlock(
            channels=32,
            dilation=1,
        )

        # Medium-range relations across corridors and blast areas.
        self.dilated_block = DQNResidualBlock(
            channels=32,
            dilation=2,
        )

        # Refine the combined local and medium-range information.
        self.local_block_2 = DQNResidualBlock(
            channels=32,
            dilation=1,
        )

        # Reduce the flattened representation without losing grid positions.
        self.grid_compression = nn.Sequential(
            nn.Conv2d(
                32,
                8,
                kernel_size=1,
            ),
            nn.ReLU(),
        )

        # Determine the flattened grid size automatically.
        with torch.no_grad():
            dummy_grid = torch.zeros(
                1,
                in_channels,
                grid_size[0],
                grid_size[1],
            )
            dummy_grid = self._forward_grid(dummy_grid)
            grid_output_size = int(
                dummy_grid.flatten(start_dim=1).shape[1]
            )

        # Process global scalar information separately.
        self.scalar_network = nn.Sequential(
            nn.Linear(scalar_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_size = grid_output_size + 64

        # Combine spatial and scalar information.
        self.shared_network = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Estimate the general quality of the state.
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

        # Estimate the relative advantage of each action.
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_actions),
        )

        self._initialize_weights()

    def _forward_grid(
        self,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        grid = self.grid_stem(grid)
        grid = self.local_block_1(grid)
        grid = self.dilated_block(grid)
        grid = self.local_block_2(grid)
        grid = self.grid_compression(grid)
        return grid

    def _initialize_weights(self) -> None:
        """Initialize ReLU layers with Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # The final output layers have no ReLU after them. Smaller initial
        # weights keep initial Q-values close to zero.
        nn.init.uniform_(
            self.value_stream[-1].weight,
            -1e-3,
            1e-3,
        )
        nn.init.zeros_(
            self.value_stream[-1].bias
        )

        nn.init.uniform_(
            self.advantage_stream[-1].weight,
            -1e-3,
            1e-3,
        )
        nn.init.zeros_(
            self.advantage_stream[-1].bias
        )

    def forward(
        self,
        grid: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        # Spatial branch.
        grid_features = self._forward_grid(grid)
        grid_features = torch.flatten(
            grid_features,
            start_dim=1,
        )

        # Scalar branch.
        scalar_features = self.scalar_network(scalar)

        # Combine both representations.
        combined = torch.cat(
            (grid_features, scalar_features),
            dim=1,
        )
        shared = self.shared_network(combined)

        # Dueling DQN streams.
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)

        # Subtracting the mean makes the value/advantage decomposition
        # identifiable while preserving one Q-value per action.
        q_values = (
            value
            + advantage
            - advantage.mean(dim=1, keepdim=True)
        )

        return q_values

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

            nn.AdaptiveAvgPool2d((3, 3)),
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


class DQN_prev(nn.Module):
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
