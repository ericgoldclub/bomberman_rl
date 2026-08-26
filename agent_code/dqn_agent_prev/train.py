import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .callbacks import ACTIONS, state_to_features, MODEL_FILE, BEST_MODEL_FILE, TRAINING_MODEL_FILE, _feature_dimensions, _is_valid_action, _policy_action_mask
import events as e
import settings as s

Transition = namedtuple('Transition', ('grid', 'scalar', 'action', 'next_grid', 'next_scalar', 'next_action_mask', 'reward'))

# Hyperparameters
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 8192  # steps
MIN_REPLAY_SIZE = 5000
TRAIN_EVERY_STEPS = 8


class DQN(nn.Module):
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
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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

def setup_training(self):
    """Initialise training-related objects for the agent."""

    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0
    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.best_score = -1
    self.previous_old_position = None
    self.previous_velocity = (0,0)
    self.position_history = deque(maxlen=4)

    # Determine sizes from self (set by callbacks.setup) or infer them from the feature extractor.
    C = getattr(self, 'grid_channels', None)
    if C is None:
        C, _ = _feature_dimensions()
    H, W = getattr(self, 'grid_size', (17, 17))
    S = getattr(self, 'scalar_size', None)
    if S is None:
        _, S = _feature_dimensions()

    # Instantiate networks if not already present
    if not hasattr(self, 'policy_net'):
        self.policy_net = DQN(in_channels=C, grid_size=(H, W), scalar_size=S, n_actions=len(ACTIONS))
    self.logger.info( "Training checkpoint source path=%s exists=%s",
        TRAINING_MODEL_FILE,
        os.path.isfile(TRAINING_MODEL_FILE),
    )
    if os.path.isfile(TRAINING_MODEL_FILE):
        load_device = getattr(self, 'device', torch.device("cpu"))
        self.policy_net.load_state_dict(torch.load(TRAINING_MODEL_FILE, map_location=load_device))
        self.logger.info("Loaded training checkpoint from %s", TRAINING_MODEL_FILE)
    if not hasattr(self, 'target_net'):
        self.target_net = DQN(in_channels=C, grid_size=(H, W), scalar_size=S, n_actions=len(ACTIONS))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is not trained, only used for evaluation

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.loss_fn = nn.SmoothL1Loss()

    # Device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    # For convenience: keep epsilon parameters on self (can be tuned)
    self.epsilon_start = 1.0
    self.epsilon_end = 0.05
    self.epsilon_decay = 514235

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
    non_final_next_action_mask_np = np.stack([b.next_action_mask for b in batch if b.next_action_mask is not None], axis=0) if non_final_mask_np.any() else np.empty((0, len(ACTIONS)), dtype=bool)

    # Convert to torch tensors

    states_grid = torch.from_numpy(states_grid.astype(np.float32)).to(self.device)
    states_scalar = torch.from_numpy(states_scalar.astype(np.float32)).to(self.device)
    actions = torch.from_numpy(actions_np.astype(np.int64)).to(self.device).unsqueeze(1)
    rewards = torch.from_numpy(rewards_np.astype(np.float32)).to(self.device)

    nonfinal_next_grid = torch.from_numpy(non_final_next_grid_np.astype(np.float32)).to(self.device) if non_final_next_grid_np.size else torch.empty((0, states_grid.shape[1], states_grid.shape[2], states_grid.shape[3]), device=self.device)

    nonfinal_next_scalar = torch.from_numpy(non_final_next_scalar_np.astype(np.float32)).to(self.device) if non_final_next_scalar_np.size else torch.empty((0, states_scalar.shape[1]), device=self.device)
    nonfinal_next_action_mask = torch.from_numpy(non_final_next_action_mask_np.astype(np.bool_)).to(self.device) if non_final_next_action_mask_np.size else torch.empty((0, len(ACTIONS)), dtype=torch.bool, device=self.device)

    # Compute Q(s_t, a)
    q_values_all = self.policy_net(states_grid, states_scalar) # (B, n_actions)
    q_values = q_values_all.gather(1, actions) # (B, 1)

    # Double DQN target evaluation
    next_q_values = torch.zeros((BATCH_SIZE, 1), device=self.device)
    if nonfinal_next_grid.size(0) > 0:
        with torch.no_grad():
            next_policy_q = self.policy_net(nonfinal_next_grid, nonfinal_next_scalar) # (B', n_actions)
            next_policy_q = next_policy_q.masked_fill(~nonfinal_next_action_mask, float('-inf'))
            next_actions = next_policy_q.argmax(dim=1, keepdim=True) # (B', 1)
            # Evaluate these actions using target network
            next_target_q = self.target_net(nonfinal_next_grid, nonfinal_next_scalar).gather(1, next_actions) # (B', 1)
        # Assign next Q-values into the full batch tensor at indices of non-final transitions
        # Convert boolean mask to torch on device
        non_final_mask = torch.from_numpy(non_final_mask_np).to(self.device)
        # next_target_q has shape (N,1); place its values into next_q_values where mask is True
        next_q_values[non_final_mask, 0] = next_target_q.squeeze(1)

    expected_q = rewards + (GAMMA * next_q_values)
    loss = self.loss_fn(q_values, expected_q)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
    self.optimizer.step()

    if getattr(self, "log_dqn_details", False):
        self.logger.info(f"loss={loss.item():.6f}")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Called once per step to allow intermediate rewards based on game events and to store transitions."""

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    events = list(events)
    self.round_coins_collected += events.count(e.COIN_COLLECTED)
    self.round_number_kills += events.count(e.KILLED_OPPONENT)
    if new_game_state is not None:
        field = new_game_state["field"]
        explosion_map = new_game_state.get("explosion_map", np.zeros_like(field))
        _, _, _, (x, y) = new_game_state["self"]
        if explosion_map[x, y] > 0:
            events.append(e.IN_DANGER)

        '''
        if self_action == 'WAIT' and np.max(explosion_map) == 0 and new_game_state.get("coins"):
            has_valid_move = any(
                _is_valid_action(new_game_state, action)
                for action in ('UP', 'RIGHT', 'DOWN', 'LEFT')
            )
            if has_valid_move:
                events.append(e.SAFE_WAIT)
       '''
        if old_game_state is not None:
            old_pos = old_game_state["self"][-1]
            new_pos = new_game_state["self"][-1]

            current_velocity = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])

            previous_vx, previous_vy = self.previous_velocity
            current_vx, current_vy = current_velocity

            if (current_vx !=0 or current_vy !=0) and (previous_vx !=0 or previous_vy !=0):
                skp = current_vx*previous_vx + current_vy*previous_vy
                if skp < 0:
                    events.append(e.REVERSED_DIRECTION)
           
            self.previous_old_position = old_pos
            # Update velocity for next step's reversal detection
            self.previous_velocity = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
         
            '''
            old_others = [o[-1] for o in old_game_state["others"] if o[-1] is not None]
            new_others = [o[-1] for o in new_game_state["others"] if o[-1] is not None]
            if old_others and new_others:
                old_enemy_min = min(abs(old_pos[0] - ox) + abs(old_pos[1] - oy) for ox, oy in old_others)
                new_enemy_min = min(abs(new_pos[0] - ox) + abs(new_pos[1] - oy) for ox, oy in new_others)
                if new_enemy_min < old_enemy_min:
                    events.append(e.MOVED_CLOSE_TO_ENEMY)
                elif new_enemy_min > old_enemy_min:
                    events.append(e.MOVED_AWAY_FROM_ENEMY)
            '''
    reward = reward_from_events(self, events)
    old_feats = state_to_features(old_game_state)
    new_feats = state_to_features(new_game_state)

    if old_feats is not None:
        old_grid, old_scalar = old_feats
        if new_feats is not None:
            new_grid, new_scalar = new_feats
            next_action_mask = _policy_action_mask(new_game_state)
        else:
            new_grid, new_scalar = None, None
            next_action_mask = None

        # store transition (use copies to avoid accidental mutation)
        self.replay_buffer.append(Transition(old_grid.copy(), old_scalar.copy(),
                                             self_action,
                                             None if new_grid is None else new_grid.copy(),
                                             None if new_scalar is None else new_scalar.copy(),
                                             None if next_action_mask is None else next_action_mask.copy(),
                                             reward))

        # update step counter and train periodically to reduce compute cost
        self.steps_done += 1
        if self.steps_done % TRAIN_EVERY_STEPS == 0:
            optimize_model(self)

        # occasionally update target network
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    #self.round_coins_collected += events.count(e.COIN_COLLECTED)
    #self.round_number_kills += events.count(e.KILLED_OPPONENT)
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    # terminal state: next_state is None
    if last_state is not None:
        last_grid, last_scalar = last_state
        self.replay_buffer.append(Transition(last_grid.copy(), last_scalar.copy(), last_action, None, None, None, reward))

    # Do some final optimization passes
    for _ in range(10):
        optimize_model(self)

    # Save latest policy network to MODEL_FILE; TRAINING_MODEL_FILE is the read-only start checkpoint
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(self.policy_net.state_dict(), MODEL_FILE)

    current_score = self.round_coins_collected + self.round_number_kills * 5
    if current_score > self.best_score:
        self.best_score = current_score
        torch.save(self.policy_net.state_dict(), BEST_MODEL_FILE)
        self.logger.info(
            "Saved new best model with score %d.",
            current_score,
        )

    self.round_coins_collected = 0
    self.round_number_kills = 0
    self.previous_old_position = None
    self.position_history.clear()


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
    # Split rewards into:
    # - major outcomes (coin collection / death), which should dominate learning
    # - shaping terms (movement heuristics), which are clipped per step
    major_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.COIN_FOUND: 0.10,
        e.OPPONENT_ELIMINATED: 1.0,
        e.CRATE_DESTROYED: 0.6,
    }
    shaping_rewards = {

        e.REVERSED_DIRECTION: -0.05,

        e.MOVED_CLOSE_TO_ENEMY: 0.00,
        e.MOVED_AWAY_FROM_ENEMY: 0.00,

        e.IN_DANGER: -0.10,
        e.SAFE_WAIT: 0.00,

        e.WAITED: -0.05,
        e.INVALID_ACTION: 0.0,
        e.BOMB_DROPPED: 0.1,
        e.BOMB_EXPLODED: 0.0,
        e.SURVIVED_ROUND: 0.0,
    }

    major_sum = 0.0
    shaping_sum = 0.0
    for event in events:
        major_sum += major_rewards.get(event, 0.0)
        shaping_sum += shaping_rewards.get(event, 0.0)

    # Prevent step-wise shaping terms from overshadowing major outcomes.
    shaping_sum = float(np.clip(shaping_sum, -0.25, 0.25))
    reward_sum = major_sum + shaping_sum

    if getattr(self, "log_dqn_details", False):
        self.logger.info(
            f"Awarded {reward_sum:.3f} (major={major_sum:.3f}, shaping={shaping_sum:.3f}) "
            f"for events {', '.join(events)}"
        )

    return float(reward_sum)
