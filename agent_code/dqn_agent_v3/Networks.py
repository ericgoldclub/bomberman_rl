import torch
import torch.nn as nn


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


class ResidualBlock(nn.Module):
    """Full-resolution dilated residual block with channel attention."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(16, channels),
            nn.SiLU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(16, channels),
        )
        reduced_channels = max(16, channels // 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers(x)
        features = features * self.channel_gate(features)
        return self.activation(x + features)


class AxialSpatialTransformerBlock(nn.Module):
    """Efficient global attention across board rows and columns."""

    def __init__(
        self,
        feature_size: int,
        heads: int,
        grid_size: tuple[int, int],
        expansion: int = 4,
    ):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.row_norm = nn.LayerNorm(feature_size)
        self.row_attention = nn.MultiheadAttention(
            feature_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.column_norm = nn.LayerNorm(feature_size)
        self.column_attention = nn.MultiheadAttention(
            feature_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(feature_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, expansion * feature_size),
            nn.SiLU(),
            nn.Linear(expansion * feature_size, feature_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, feature_size = tokens.shape
        height, width = self.grid_size
        if token_count != height * width:
            raise ValueError(
                f"Expected {height * width} spatial tokens, got {token_count}."
            )

        spatial = tokens.reshape(batch_size, height, width, feature_size)

        rows = spatial.reshape(batch_size * height, width, feature_size)
        normalized_rows = self.row_norm(rows)
        attended_rows, _ = self.row_attention(
            normalized_rows,
            normalized_rows,
            normalized_rows,
            need_weights=False,
        )
        spatial = (rows + attended_rows).reshape(
            batch_size,
            height,
            width,
            feature_size,
        )

        columns = spatial.permute(0, 2, 1, 3).reshape(
            batch_size * width,
            height,
            feature_size,
        )
        normalized_columns = self.column_norm(columns)
        attended_columns, _ = self.column_attention(
            normalized_columns,
            normalized_columns,
            normalized_columns,
            need_weights=False,
        )
        spatial = (columns + attended_columns).reshape(
            batch_size,
            width,
            height,
            feature_size,
        ).permute(0, 2, 1, 3)

        tokens = spatial.reshape(batch_size, token_count, feature_size)
        return tokens + self.feed_forward(
            self.feed_forward_norm(tokens)
        )


class HybridDQN(nn.Module):
    """Large full-resolution spatial-attention dueling DQN.

    The convolutional trunk never downsamples the board. Dilated residual
    blocks learn local movement, corridors and blast geometry, while global
    self-attention relates the player to distant coins, enemies and bombs.
    Six learned action queries then extract different evidence for each action.
    """

    TRUNK_CHANNELS = 48
    TOKEN_SIZE = 96
    ATTENTION_HEADS = 8
    ATTENTION_BLOCKS = 1
    SELF_CHANNEL = 3

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
        if in_channels <= self.SELF_CHANNEL:
            raise ValueError(
                f"HybridDQN requires self position channel {self.SELF_CHANNEL}."
            )

        self.grid_size = tuple(grid_size)
        self.n_actions = int(n_actions)

        # Relative coordinate channels are generated from the self-position
        # map. They make spatial relations explicit without changing replay.
        coordinate_x = torch.linspace(-1.0, 1.0, self.grid_size[0]).view(
            1, 1, self.grid_size[0], 1
        )
        coordinate_y = torch.linspace(-1.0, 1.0, self.grid_size[1]).view(
            1, 1, 1, self.grid_size[1]
        )
        self.register_buffer("coordinate_x", coordinate_x, persistent=False)
        self.register_buffer("coordinate_y", coordinate_y, persistent=False)

        self.spatial_stem = nn.Sequential(
            nn.Conv2d(
                in_channels + 2,
                self.TRUNK_CHANNELS,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(16, self.TRUNK_CHANNELS),
            nn.SiLU(),
        )

        dilations = (1, 2, 3, 1, 2, 3)
        self.spatial_blocks = nn.Sequential(
            *[
                ResidualBlock(self.TRUNK_CHANNELS, dilation=dilation)
                for dilation in dilations
            ]
        )
        self.token_projection = nn.Conv2d(
            self.TRUNK_CHANNELS,
            self.TOKEN_SIZE,
            kernel_size=1,
        )
        self.spatial_transformer = nn.Sequential(
            *[
                AxialSpatialTransformerBlock(
                    self.TOKEN_SIZE,
                    self.ATTENTION_HEADS,
                    self.grid_size,
                )
                for _ in range(self.ATTENTION_BLOCKS)
            ]
        )

        self.vector_encoder = nn.Sequential(
            nn.Linear(scalar_size, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.TOKEN_SIZE),
            nn.LayerNorm(self.TOKEN_SIZE),
            nn.SiLU(),
        )

        # Each query can attend to different spatial evidence: for example an
        # UP query can focus on the tile above the player while BOMB can focus
        # on escape routes, crates and enemies in the blast lanes.
        self.action_queries = nn.Parameter(
            torch.empty(1, self.n_actions, self.TOKEN_SIZE)
        )
        self.action_query_norm = nn.LayerNorm(self.TOKEN_SIZE)
        self.memory_norm = nn.LayerNorm(self.TOKEN_SIZE)
        self.action_attention = nn.MultiheadAttention(
            self.TOKEN_SIZE,
            self.ATTENTION_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.action_refinement = nn.Sequential(
            nn.LayerNorm(self.TOKEN_SIZE),
            nn.Linear(self.TOKEN_SIZE, 2 * self.TOKEN_SIZE),
            nn.SiLU(),
            nn.Linear(2 * self.TOKEN_SIZE, self.TOKEN_SIZE),
        )

        self.value_head = nn.Sequential(
            nn.Linear(3 * self.TOKEN_SIZE, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.LayerNorm(self.TOKEN_SIZE),
            nn.Linear(self.TOKEN_SIZE, self.TOKEN_SIZE),
            nn.SiLU(),
            nn.Linear(self.TOKEN_SIZE, 1),
        )

        self._initialize_weights()

    def _relative_coordinates(self, self_map: torch.Tensor) -> tuple:
        agent_x = (self_map * self.coordinate_x).sum(
            dim=(2, 3),
            keepdim=True,
        )
        agent_y = (self_map * self.coordinate_y).sum(
            dim=(2, 3),
            keepdim=True,
        )
        relative_x = (self.coordinate_x - agent_x) / 2.0
        relative_y = (self.coordinate_y - agent_y) / 2.0
        return (
            relative_x.expand(-1, -1, *self.grid_size),
            relative_y.expand(-1, -1, *self.grid_size),
        )

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        nn.init.uniform_(self.value_head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.value_head[-1].bias)
        nn.init.uniform_(self.advantage_head[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.advantage_head[-1].bias)

    def forward(
        self,
        board: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(board.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"Expected board size {self.grid_size}, got {tuple(board.shape[-2:])}."
            )

        self_map = board[:, self.SELF_CHANNEL:self.SELF_CHANNEL + 1]
        relative_x, relative_y = self._relative_coordinates(self_map)
        spatial = torch.cat((board, relative_x, relative_y), dim=1)
        spatial = self.spatial_stem(spatial)
        spatial = self.spatial_blocks(spatial)
        spatial = self.token_projection(spatial)

        spatial_tokens = spatial.flatten(2).transpose(1, 2)
        spatial_tokens = self.spatial_transformer(spatial_tokens)
        vector_token = self.vector_encoder(vector)

        self_weights = self_map.flatten(2).transpose(1, 2)
        self_features = (spatial_tokens * self_weights).sum(dim=1)
        global_features = spatial_tokens.mean(dim=1)

        memory = torch.cat(
            (spatial_tokens, vector_token.unsqueeze(1)),
            dim=1,
        )
        queries = self.action_queries.expand(board.shape[0], -1, -1)
        queries = queries + vector_token.unsqueeze(1)
        action_features, _ = self.action_attention(
            self.action_query_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        action_features = action_features + queries
        action_features = action_features + self.action_refinement(
            action_features
        )

        value_features = torch.cat(
            (self_features, global_features, vector_token),
            dim=1,
        )
        value = self.value_head(value_features)
        advantage = self.advantage_head(action_features).squeeze(-1)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
