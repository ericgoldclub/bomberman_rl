import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import ACTIONS, state_to_features, MODEL_FILE
import events as e

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE = 1000  # steps
MIN_REPLAY_SIZE = 500


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def setup_training(self):
    """Initialise training-related objects for the agent."""
    # Replay buffer
    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0
    self.policy_net = DQN(input_dim=7, output_dim=len(ACTIONS))
    self.target_net = DQN(input_dim=7, output_dim=len(ACTIONS))
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.loss_fn = nn.MSELoss()

    # Device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net.to(self.device)
    self.target_net.to(self.device)

    # For convenience: keep epsilon parameters on self (can be tuned)
    self.epsilon_start = 1.0
    self.epsilon_end = 0.05
    self.epsilon_decay = 20000


def optimize_model(self):
    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    batch = random.sample(self.replay_buffer, BATCH_SIZE)
    # Convert lists of numpy arrays to a single numpy.ndarray before converting to torch tensors
    states_np = np.array([b.state for b in batch], dtype=np.float32)
    states = torch.from_numpy(states_np).to(self.device)

    actions_np = np.array([ACTIONS.index(b.action) for b in batch], dtype=np.int64)
    actions = torch.from_numpy(actions_np).to(self.device).unsqueeze(1)

    rewards_np = np.array([b.reward for b in batch], dtype=np.float32)
    rewards = torch.from_numpy(rewards_np).to(self.device).unsqueeze(1)

    non_final_mask_np = np.array([b.next_state is not None for b in batch], dtype=bool)
    non_final_mask = torch.from_numpy(non_final_mask_np).to(self.device)

    non_final_next_list = [b.next_state for b in batch if b.next_state is not None]
    if len(non_final_next_list) > 0:
        non_final_next_states = torch.from_numpy(np.array(non_final_next_list, dtype=np.float32)).to(self.device)
    else:
        # Create an empty tensor with correct feature dimension (7)
        non_final_next_states = torch.empty((0, states.shape[1]), dtype=torch.float32, device=self.device)

    # Compute Q(s_t, a)
    q_values = self.policy_net(states).gather(1, actions)

    # Compute V(s_{t+1}) for all next states.
    next_q_values = torch.zeros((BATCH_SIZE, 1), device=self.device)
    if non_final_next_states.size(0) > 0:
        next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().unsqueeze(1)

    # Compute expected Q values
    expected_q_values = rewards + (GAMMA * next_q_values)

    loss = self.loss_fn(q_values, expected_q_values)

    self.optimizer.zero_grad()
    loss.backward()
    # small gradient clipping
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5)
    self.optimizer.step()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """Called once per step to allow intermediate rewards based on game events and to store transitions."""
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)

    # Store transition
    if old_state is not None:
        self.replay_buffer.append(Transition(old_state, self_action, new_state, reward))

    # Perform optimization step
    optimize_model(self)

    # Update target network periodically
    self.steps_done += 1
    if self.steps_done % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    # terminal state: next_state is None
    if last_state is not None:
        self.replay_buffer.append(Transition(last_state, last_action, None, reward))

    # Do some final optimization passes
    for _ in range(10):
        optimize_model(self)

    # Save the policy network
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save(self.policy_net.state_dict(), MODEL_FILE)


def reward_from_events(self, events: List[str]) -> int:
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.MOVED_CLOSE_TO_COIN: 3,
        e.MOVED_AWAY_FROM_COIN: -3,
        e.MOVED_CLOSE_TO_ENEMY: -1,
        e.MOVED_AWAY_FROM_ENEMY: 1,
        e.WAITED: -1,
        e.INVALID_ACTION: -3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
