import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import ACTIONS, state_to_features, MODEL_FILE, BOARD_CHANNELS, VECTOR_DIM, BOARD_SIZE
import events as e

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE = 1000  # steps
MIN_REPLAY_SIZE = 500


class HybridDQN(nn.Module):
    def __init__(self, board_channels, vector_dim, output_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        cnn_out = 64 * BOARD_SIZE * BOARD_SIZE  # Assuming BOARD_SIZE is defined elsewhere

        self.head = nn.Sequential(
            nn.Linear(cnn_out + vector_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, board, vector):
        board_features = self.cnn(board)
        combined = torch.cat((board_features, vector), dim=1)
        return self.head(combined)

def setup_training(self):
    """Initialise training-related objects for the agent."""
    # Replay buffer
    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0
    self.policy_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=len(ACTIONS))
    self.target_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=len(ACTIONS))
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.loss_fn = nn.MSELoss()

    # Device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.policy_net.to(self.device)
    self.target_net.to(self.device)



def optimize_model(self):
    if len(self.replay_buffer) < MIN_REPLAY_SIZE:
        return

    batch = random.sample(self.replay_buffer, BATCH_SIZE)
    # Convert lists of numpy arrays to a single numpy.ndarray before converting to torch tensors
    boards = torch.from_numpy(np.array([b.state[0] for b in batch], dtype=np.float32)).to(self.device)
    vectors = torch.from_numpy(np.array([b.state[1] for b in batch], dtype=np.float32)).to(self.device)

    actions_np = np.array([ACTIONS.index(b.action) for b in batch], dtype=np.int64)
    actions = torch.from_numpy(actions_np).to(self.device).unsqueeze(1)

    rewards_np = np.array([b.reward for b in batch], dtype=np.float32)
    rewards = torch.from_numpy(rewards_np).to(self.device).unsqueeze(1)

    non_final_mask_np = np.array([b.next_state is not None for b in batch], dtype=bool)
    non_final_mask = torch.from_numpy(non_final_mask_np).to(self.device)

    non_final_next_list = [b.next_state for b in batch if b.next_state is not None]

    # Compute Q(s_t, a)
    q_values = self.policy_net(boards, vectors).gather(1, actions)

    # Compute V(s_{t+1}) for all next states.
    next_q_values = torch.zeros((BATCH_SIZE, 1), device=self.device)
    with torch.no_grad():
        if len(non_final_next_list) > 0:
            next_boards = torch.from_numpy(np.array([s[0] for s in non_final_next_list], dtype=np.float32)).to(self.device)
            next_vectors = torch.from_numpy(np.array([s[1] for s in non_final_next_list], dtype=np.float32)).to(self.device)
            next_actions = self.policy_net(next_boards, next_vectors).argmax(1, keepdim=True)
            next_q_values[non_final_mask] = self.target_net(next_boards, next_vectors).gather(1, next_actions)

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

    #did we move closer to a coin or further away from it?
    if old_state is not None and new_state is not None:
        old_vector = old_state[1]
        new_vector = new_state[1]
        old_coin_dx, old_coin_dy = old_vector[0], old_vector[1]
        new_coin_dx, new_coin_dy = new_vector[0], new_vector[1]

        old_coin_distance = np.sqrt(old_coin_dx**2 + old_coin_dy**2)
        new_coin_distance = np.sqrt(new_coin_dx**2 + new_coin_dy**2)

        if new_coin_distance < old_coin_distance:
            events.append(e.MOVED_CLOSE_TO_COIN)
        elif new_coin_distance > old_coin_distance:
            events.append(e.MOVED_AWAY_FROM_COIN)

    #did we move closer to a crate or further away from it?
    if old_state is not None and new_state is not None:
        old_vector = old_state[1]
        new_vector = new_state[1]
        old_crate_dx, old_crate_dy = old_vector[2], old_vector[3]
        new_crate_dx, new_crate_dy = new_vector[2], new_vector[3]

        old_crate_distance = np.sqrt(old_crate_dx**2 + old_crate_dy**2)
        new_crate_distance = np.sqrt(new_crate_dx**2 + new_crate_dy**2)

        if new_crate_distance < old_crate_distance:
            events.append(e.MOVED_TOWARDS_CRATE)
        elif new_crate_distance > old_crate_distance:
            events.append(e.MOVED_AWAY_FROM_CRATE)

    if self_action == 'BOMB':
        if old_state is not None and old_state[1][4] == 1:  # can destroy crate
            events.append(e.USEFUL_BOMB_DROPPED)
        else:
            events.append(e.USELESS_BOMB_DROPPED)

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
        e.COIN_COLLECTED: 12,
        e.COIN_FOUND: 10,
        e.CRATE_DESTROYED: 8,
        e.USEFUL_BOMB_DROPPED: 1,
        e.USELESS_BOMB_DROPPED: -5,
        e.MOVED_CLOSE_TO_COIN: 3,
        e.MOVED_AWAY_FROM_COIN: -2,
        e.MOVED_TOWARDS_CRATE: 1,
        e.MOVED_AWAY_FROM_CRATE: -1,
        e.INVALID_ACTION: -4,
        e.KILLED_SELF: -40,
        e.GOT_KILLED: -40,
        e.WAITED: -3,

        }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
