import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import ACTIONS, state_to_features, MODEL_FILE, BEST_MODEL_FILE
import events as e
import settings as s

Transition = namedtuple('Transition', ('grid', 'scalar', 'action', 'next_grid', 'next_scalar', 'reward'))

# Hyperparameters
BUFFER_SIZE = 5000
BATCH_SIZE = 32
GAMMA = 0.97
LR = 3e-4
TARGET_UPDATE = 500  # steps
MIN_REPLAY_SIZE = 256


class DQN(nn.Module):
    '''
    The model combines a small CNN for grid channels with a MLP for scalar features, 
    such as (x, y) coordinates of the agent and the direction to the nearest coin. 
    The outputs are Q-values for each action.
    '''
    def __init__(self, in_channels : int, grid_size: tuple[int, int] , scalar_size : int, n_actions : int):
        super().__init__()
        H, W = grid_size

        # CNN for grid channels
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1)
        # Keep coarse spatial information for navigation-sensitive decisions.
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        cnn_out_dim = 64 * 4 * 4

        # MLP for scalar features
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Final MLP to combine CNN and scalar features
        combined_dim = cnn_out_dim + 64
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions) # output Q-values for each action
        )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, grid : torch.Tensor, scalar : torch.Tensor) -> torch.Tensor:
        # grid: (batch_size, in_channels, H, W)
        # scalar: (batch_size, scalar_size)

        X = F.relu(self.conv1(grid))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = self.spatial_pool(X)  # (batch_size, 64, 4, 4)
        X = X.view(X.size(0), -1)  # flatten

        scalar_out = self.scalar_fc(scalar)  # (batch_size, 64)
        combined = torch.cat([X, scalar_out], dim=1)  # (batch_size, combined_dim)
        Q = self.head(combined)  # (batch_size, n_actions)
        
        return Q

def setup_training(self):
    """Initialise training-related objects for the agent."""

    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0
    self.round_coins_collected = 0
    self.best_round_coins = -1

    # Determine sizes from self (set by callbacks.setup) or fall back to defaults
    C = getattr(self, 'grid_channels', 7)
    H, W = getattr(self, 'grid_size', (17, 17))
    S = getattr(self, 'scalar_size', 3)

    # Instantiate networks if not already present
    if not hasattr(self, 'policy_net'):
        self.policy_net = DQN(in_channels=C, grid_size=(H, W), scalar_size=S, n_actions=len(ACTIONS))
    if not hasattr(self, 'target_net'):
        self.target_net = DQN(in_channels=C, grid_size=(H, W), scalar_size=S, n_actions=len(ACTIONS))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is not trained, only used for evaluation

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.loss_fn = nn.MSELoss()

    # Device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    # For convenience: keep epsilon parameters on self (can be tuned)
    self.epsilon_start = 1.0
    self.epsilon_end = 0.05
    self.epsilon_decay = 60000

    # Attach a helper to compute current epsilon
    self.get_epsilon = lambda: self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)


def optimize_model(self):
    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    batch = random.sample(self.replay_buffer, BATCH_SIZE)

    states_grid = np.stack([b.grid for b in batch], axis=0)# (B, C, H, W)
    states_scalar = np.stack([b.scalar for b in batch], axis=0) # (B, scalar_size)
    actions_np = np.array([ACTIONS.index(b.action) for b in batch], dtype=np.int64) # (B,)
    rewards_np = np.array([b.reward for b in batch], dtype=np.float32).reshape(-1, 1) # (B, 1)

    non_final_mask_np = np.array([b.next_grid is not None for b in batch], dtype=bool) # (B,)
    non_final_next_grid_np = np.stack([b.next_grid for b in batch if b.next_grid is not None], axis=0) if non_final_mask_np.any() else np.empty((0, states_grid.shape[1], states_grid.shape[2], states_grid.shape[3]), dtype=np.float32)

    non_final_next_scalar_np = np.stack([b.next_scalar for b in batch if b.next_scalar is not None], axis=0) if non_final_mask_np.any() else np.empty((0, states_scalar.shape[1]), dtype=np.float32)

    # Convert to torch tensors

    states_grid = torch.from_numpy(states_grid.astype(np.float32)).to(self.device)
    states_scalar = torch.from_numpy(states_scalar.astype(np.float32)).to(self.device)
    actions = torch.from_numpy(actions_np.astype(np.int64)).to(self.device).unsqueeze(1)
    rewards = torch.from_numpy(rewards_np.astype(np.float32)).to(self.device)

    nonfinal_next_grid = torch.from_numpy(non_final_next_grid_np.astype(np.float32)).to(self.device) if non_final_next_grid_np.size else torch.empty((0, states_grid.shape[1], states_grid.shape[2], states_grid.shape[3]), device=self.device)

    nonfinal_next_scalar = torch.from_numpy(non_final_next_scalar_np.astype(np.float32)).to(self.device) if non_final_next_scalar_np.size else torch.empty((0, states_scalar.shape[1]), device=self.device)

    # Compute Q(s_t, a)
    q_values_all = self.policy_net(states_grid, states_scalar) # (B, n_actions)
    q_values = q_values_all.gather(1, actions) # (B, 1)

    # Double DQN target evaluation 
    next_q_values = torch.zeros((BATCH_SIZE, 1), device=self.device)
    if nonfinal_next_grid.size(0) > 0:
        # Get the best actions from policy network 
        next_policy_q = self.policy_net(nonfinal_next_grid, nonfinal_next_scalar) # (B', n_actions)
        next_actions = next_policy_q.argmax(dim=1, keepdim=True) # (B', 1)
        # Evaluate these actions using target network
        next_target_q = self.target_net(nonfinal_next_grid, nonfinal_next_scalar).gather(1, next_actions) # (B', 1)
        # Assign next Q-values into the full batch tensor at indices of non-final transitions
        # Convert boolean mask to torch on device
        non_final_mask = torch.from_numpy(non_final_mask_np).to(self.device)
        # next_target_q has shape (N,1); place its values into next_q_values where mask is True
        next_q_values[non_final_mask, 0] = next_target_q.squeeze(1).detach()

    expected_q = rewards + (GAMMA * next_q_values)
    loss = self.loss_fn(q_values, expected_q)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
    self.optimizer.step()

    # track loss value in log
    try:
        self.logger.info(f"loss={loss.item():.6f}")
    except Exception:
        pass


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Called once per step to allow intermediate rewards based on game events and to store transitions."""

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    events = list(events)
    self.round_coins_collected += events.count(e.COIN_COLLECTED)
    if new_game_state is not None:
        field = new_game_state["field"]
        explosion_map = new_game_state.get("explosion_map", np.zeros_like(field))
        _, _, _, (x, y) = new_game_state["self"]
        if explosion_map[x, y] > 0:
            events.append(e.IN_DANGER)

        if self_action == 'WAIT' and np.max(explosion_map) == 0 and new_game_state.get("coins"):
            events.append(e.SAFE_WAIT)

        if old_game_state is not None and old_game_state["coins"]:
            old_pos = old_game_state["self"][-1]
            new_pos = new_game_state["self"][-1]
            old_dists = [abs(old_pos[0] - cx) + abs(old_pos[1] - cy) for cx, cy in old_game_state["coins"]]
            new_dists = [abs(new_pos[0] - cx) + abs(new_pos[1] - cy) for cx, cy in new_game_state["coins"]]
            old_min = min(old_dists) if old_dists else 1e9
            new_min = min(new_dists) if new_dists else 1e9
            if new_min < old_min:
                events.append(e.MOVED_CLOSE_TO_COIN)
            else:
                events.append(e.MOVED_AWAY_FROM_COIN)

    reward = reward_from_events(self, events)
    old_feats = state_to_features(old_game_state)
    new_feats = state_to_features(new_game_state)

    if old_feats is not None:
        old_grid, old_scalar = old_feats
        if new_feats is not None:
            new_grid, new_scalar = new_feats
        else:
            new_grid, new_scalar = None, None

        # store transition (use copies to avoid accidental mutation)
        self.replay_buffer.append(Transition(old_grid.copy(), old_scalar.copy(),
                                             self_action,
                                             None if new_grid is None else new_grid.copy(),
                                             None if new_scalar is None else new_scalar.copy(),
                                             reward))

        # perform an optimization step
        optimize_model(self)

        # update step counter and occasionally update target network
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    self.round_coins_collected += events.count(e.COIN_COLLECTED)
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    # terminal state: next_state is None
    if last_state is not None:
        last_grid, last_scalar = last_state
        self.replay_buffer.append(Transition(last_grid.copy(), last_scalar.copy(), last_action, None, None, reward))

    # Do some final optimization passes
    for _ in range(10):
        optimize_model(self)

    # Save latest policy network
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(self.policy_net.state_dict(), MODEL_FILE)

    if self.round_coins_collected > self.best_round_coins:
        self.best_round_coins = self.round_coins_collected
        torch.save(self.policy_net.state_dict(), BEST_MODEL_FILE)
        self.logger.info(
            "Saved new best model with %d collected coins.",
            self.round_coins_collected,
        )

    self.round_coins_collected = 0


def reward_from_events(self, events: List[str]) -> float:
    """Map game events to scalar rewards.

    This function centralizes reward shaping. Values chosen below are a starting
    point and can be tuned. The function returns a float to allow fractional
    rewards (e.g., small penalties for dropping bombs).

    Design goals:
    - Encourage coin collection and finding coins.
    - Encourage destroying crates (makes coins available) but penalize suicides.
    - Penalize meaningless waiting or invalid actions.
    - Reward eliminating opponents and surviving rounds.
    """
    # Base rewards for individual events
    game_rewards = {
        # coin-related
        e.COIN_COLLECTED: 20.0,
        e.COIN_FOUND: 3.0,
        e.MOVED_CLOSE_TO_COIN: 3.0,
        e.MOVED_AWAY_FROM_COIN: -2.0,

        # movement / idle / invalid
        e.WAITED: -1,
        e.SAFE_WAIT: -2,
        e.INVALID_ACTION: -2.0,

        # bomb-related
        e.BOMB_DROPPED: -4.0,
        e.BOMB_EXPLODED: 0.0,
        e.IN_DANGER: -6.0,

        # crate / environment changes
        e.CRATE_DESTROYED: 4.0,

        # combat
        e.KILLED_OPPONENT: 10.0,
        e.KILLED_SELF: -20.0,
        e.GOT_KILLED: -8.0,
        e.OPPONENT_ELIMINATED: 1.0,
        e.SURVIVED_ROUND: 0.0,

        # proximity heuristics
        e.MOVED_CLOSE_TO_ENEMY: -0.5,
        e.MOVED_AWAY_FROM_ENEMY: 0.5,
    }

    reward_sum = 0.0
    for event in events:
        reward_sum += game_rewards.get(event, 0.0)

    # Extra shaping heuristics
    # If a crate was destroyed and the agent did not die on the same step, give a small bonus
    if e.CRATE_DESTROYED in events and e.KILLED_SELF not in events and e.GOT_KILLED not in events:
        reward_sum += 1.0

    # If agent killed itself (or got killed), ensure the heavy penalty applies
    if e.KILLED_SELF in events or e.GOT_KILLED in events:
        # additional negative reinforcement for unsafe bomb placement
        reward_sum -= 2.0

    # Log the reward (use info for traceability)
    try:
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    except Exception:
        pass

    return float(reward_sum)
