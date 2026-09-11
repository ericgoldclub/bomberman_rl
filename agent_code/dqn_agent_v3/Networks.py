import torch
import torch.nn as nn


class DQN_prev(nn.Module):
    """Compact full-board dueling DQN.

    This keeps the original DQN_prev structure: a small CNN processes the
    board, a small MLP processes scalar features, and a dueling head predicts
    Q-values. The only material CNN change is a dilated/strided convolution
    schedule whose deepest features have a 17x17 receptive field, matching the
    complete tournament arena.
    """

    # (kernel_size, stride, dilation). With these four layers the receptive
    # field evolves as 1 -> 3 -> 7 -> 9 -> 17 pixels.
    CONVOLUTION_SPEC = (
        (3, 1, 1),
        (3, 1, 2),
        (3, 2, 1),
        (3, 1, 2),
    )

    def __init__(
        self,
        in_channels: int,
        grid_size: tuple[int, int],
        scalar_size: int,
        n_actions: int,
    ):
        super().__init__()

        if len(grid_size) != 2 or min(grid_size) <= 0:
            raise ValueError(
                "grid_size must contain two positive dimensions, "
                f"got {grid_size!r}"
            )

        self.grid_size = tuple(grid_size)
        self.n_actions = int(n_actions)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                in_channels,
                self.grid_size[0],
                self.grid_size[1],
            )
            cnn_out_dim = int(self.cnn(dummy).flatten(1).shape[1])

        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, self.n_actions)

        self._initialize_weights()

    @classmethod
    def receptive_field_size(cls) -> int:
        """Return the side length visible to a deepest convolution feature."""
        receptive_field = 1
        input_jump = 1
        for kernel_size, stride, dilation in cls.CONVOLUTION_SPEC:
            receptive_field += (
                (kernel_size - 1) * dilation * input_jump
            )
            input_jump *= stride
        return receptive_field

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small initial Q-values avoid large arbitrary preferences before any
        # replay update while leaving the hidden layers well conditioned.
        nn.init.uniform_(self.value.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.advantage.weight, -1e-3, 1e-3)

    def forward(
        self,
        grid: torch.Tensor,
        scalar: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(grid.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"Expected board size {self.grid_size}, "
                f"got {tuple(grid.shape[-2:])}."
            )

        board_features = self.cnn(grid).flatten(1)
        scalar_features = self.scalar_fc(scalar)
        shared_features = self.shared(
            torch.cat((board_features, scalar_features), dim=1)
        )

        value = self.value(shared_features)
        advantage = self.advantage(shared_features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
