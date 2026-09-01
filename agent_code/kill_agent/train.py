import os
import random
from collections import deque, namedtuple
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import (ACTIONS, BOMB_POWER, BOMB_TIMER, can_hit_crate_with_bomb, explosion_tiles_from_bomb, state_to_features, MODEL_FILE, BOARD_CHANNELS, BOARD_SIZE, VECTOR_DIM, useful_bomb_positions, valid_action_mask, can_hit_enemy_with_bomb, has_escape_after_bomb, enemy_has_escape_after_bomb)

import events as e

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_action_mask'))  # Added next_action to the Transition namedtuple

# Hyperparameters
BUFFER_SIZE = 40000
BATCH_SIZE = 16
GAMMA = 0.97
LR = 1e-3
TARGET_UPDATE = 512  # steps
MIN_REPLAY_SIZE = 256
TRAINING_STEPS = 4
TRAIN_EVERY = 4

class HybridDQN(nn.Module):
    def __init__(self, board_channels, vector_dim, output_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(board_channels, 16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        cnn_out = 32 * BOARD_SIZE**2 # Assuming BOARD_SIZE is defined elsewhere

        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(cnn_out + output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, board, vector):
        board_features = self.cnn(board)
        vector_features = self.mlp(vector)
        combined = torch.cat((board_features, vector_features), dim=1)
        return self.head(combined)

def setup_training(self):
    """Initialise training-related objects for the agent."""
    # Replay buffer
    self.replay_buffer = deque(maxlen=BUFFER_SIZE)
    self.steps_done = 0#
    self.position_history = deque(maxlen=4)  # Store the last 4 positions to detect oscillation
    self.last_bomb_positions = None  # Store the last bomb positions to detect if the agent is trapped
    self.steps_since_last_bomb = None  # Counter for steps since the last bomb was dropped

    
    # Device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    self.policy_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=len(ACTIONS)).to(self.device)
    self.target_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=len(ACTIONS)).to(self.device)

    if os.path.isfile(MODEL_FILE):
        self.logger.info("Loading existing DQN model for continued training.")
        state = torch.load(MODEL_FILE, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.policy_net.load_state_dict(state['model_state_dict'])
            self.target_net.load_state_dict(state['model_state_dict'])
            self.steps_done = state.get('steps_done', 0)
        else:
            self.policy_net.load_state_dict(state)
            self.target_net.load_state_dict(state)

    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
    self.loss_fn = nn.MSELoss()



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

            next_masks = torch.from_numpy(np.array([b.next_action_mask for b in batch if b.next_state is not None], dtype=np.float32)).to(self.device)

            next_policy_q_values = self.policy_net(next_boards, next_vectors)
            next_policy_q_values[~(next_masks.bool())] = float('-inf')

            next_actions = next_policy_q_values.argmax(dim=1, keepdim=True)

            next_target_q_values = self.target_net(next_boards, next_vectors).gather(1, next_actions)
            next_q_values[non_final_mask] = next_target_q_values

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
    if not hasattr(self, 'position_history'):
        self.position_history = deque(maxlen=4)

    old_state = state_to_features(old_game_state, position_history=self.position_history) if old_game_state is not None else None

    if new_game_state is not None:
        new_pos = new_game_state['self'][3]
        if len(self.position_history) == 0 or self.position_history[-1] != new_pos:
            self.position_history.append(new_pos)

    new_state = state_to_features(new_game_state, position_history=self.position_history) if new_game_state is not None else None
    #did we move closer to a coin or further away from it?
    if old_state is not None and new_state is not None:
        coins = new_game_state['coins']
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        if len(coins) > 0:
            old_coin_distance = min([np.sqrt((coin[0] - old_pos[0])**2 + (coin[1] - old_pos[1])**2) for coin in coins])
            new_coin_distance = min([np.sqrt((coin[0] - new_pos[0])**2 + (coin[1] - new_pos[1])**2) for coin in coins])

            if new_coin_distance < old_coin_distance:
                events.append(e.MOVED_CLOSE_TO_COIN)
            elif new_coin_distance > old_coin_distance:
                events.append(e.MOVED_AWAY_FROM_COIN)

    #did we move closer to a crate or further away from it?
    if old_state is not None and new_state is not None:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        old_targets = useful_bomb_positions(old_game_state["field"])
        new_targets = useful_bomb_positions(new_game_state["field"])

        if old_targets and new_targets:
            old_crate_distance = min([np.sqrt((target[0] - old_pos[0])**2 + (target[1] - old_pos[1])**2) for target in old_targets])
            new_crate_distance = min([np.sqrt((target[0] - new_pos[0])**2 + (target[1] - new_pos[1])**2) for target in new_targets])


            if new_crate_distance < old_crate_distance:
                events.append(e.MOVED_TOWARDS_CRATE)
            elif new_crate_distance > old_crate_distance:
                events.append(e.MOVED_AWAY_FROM_CRATE)

    if self_action == 'BOMB' and old_game_state is not None and new_game_state is not None:
        self.last_bomb_positions = old_game_state['self'][3]
        self.steps_since_last_bomb = 0

        enemies = [pos for _, _, _, pos in new_game_state["others"] if pos is not None]
        old_pos = old_game_state['self'][3]
        field = new_game_state["field"]
        bombs = new_game_state["bombs"]
        explosion_map = old_game_state["explosion_map"]

        hits_enemy = can_hit_enemy_with_bomb(field, old_pos[0], old_pos[1], enemies)
        hits_crate = can_hit_crate_with_bomb(field, old_pos[0], old_pos[1])
        escape_possible = has_escape_after_bomb(field, bombs, explosion_map, old_pos)
        traps_enemy = any(
            can_hit_enemy_with_bomb(field, old_pos[0], old_pos[1], [enemy])
            and not enemy_has_escape_after_bomb(field, bombs, explosion_map, enemy, old_pos)
            for enemy in enemies
        )

        closest_enemy_distance = min(
            abs(enemy[0] - old_pos[0]) + abs(enemy[1] - old_pos[1])
            for enemy in enemies
        ) if enemies else float('inf')

        if escape_possible and traps_enemy and hits_enemy and closest_enemy_distance <= BOMB_POWER:
            events.append(e.KILL_BOMB_DROPPED)
        elif escape_possible and hits_enemy:
            events.append(e.ENEMY_PRESSURE_BOMB_DROPPED)
        elif escape_possible and hits_crate and not hits_enemy:
            events.append(e.CRATE_BOMB_DROPPED)
        else:
            events.append(e.USELESS_BOMB_DROPPED)
    elif self.steps_since_last_bomb is not None:
        self.steps_since_last_bomb += 1  # Increment the counter if a bomb was dropped previously
        if self.steps_since_last_bomb > BOMB_TIMER + 1:  # Reset after 5 steps to avoid false positives
            self.steps_since_last_bomb = None
            self.last_bomb_positions = None  # Reset the last bomb position after the bomb has exploded

    

    #check for oscillation A -> B -> A -> B
    if old_state is not None and new_state is not None:
        if len(self.position_history) >= 3: 
            if self.position_history[-1] == self.position_history[-3]:
                events.append(e.OSCILLATION)

        if len(self.position_history) >= 4: 
            if self.position_history[-1] == self.position_history[-3] and self.position_history[-2] == self.position_history[-4]:
                events.append(e.OSCILLATION)

    new_state = state_to_features(new_game_state, position_history=self.position_history) if new_game_state is not None else None


    #check for enemy proximity
    if old_state is not None and new_state is not None:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]

        old_enemy_distances = [
            np.sqrt((enemy[3][0] - old_pos[0])**2 + (enemy[3][1] - old_pos[1])**2)
            for enemy in old_game_state['others']
            if enemy[3] is not None
        ]
        new_enemy_distances = [
            np.sqrt((enemy[3][0] - new_pos[0])**2 + (enemy[3][1] - new_pos[1])**2)
            for enemy in new_game_state['others']
            if enemy[3] is not None
        ]

        if old_enemy_distances and new_enemy_distances:
            old_enemy_distance = min(old_enemy_distances)
            new_enemy_distance = min(new_enemy_distances)

            if new_enemy_distance < old_enemy_distance:
                events.append(e.MOVED_CLOSE_TO_ENEMY)
            elif new_enemy_distance > old_enemy_distance:
                events.append(e.MOVED_AWAY_FROM_ENEMY)


    if old_state is not None and new_state is not None and self.last_bomb_positions is not None and self.steps_since_last_bomb is not None:
        old_pos = old_game_state['self'][3]
        new_pos = new_game_state['self'][3]
        field = new_game_state['field']

        old_dist = abs(old_pos[0] - self.last_bomb_positions[0]) + abs(old_pos[1] - self.last_bomb_positions[1])
        new_dist = abs(new_pos[0] - self.last_bomb_positions[0]) + abs(new_pos[1] - self.last_bomb_positions[1])

        own_blast_tiles = explosion_tiles_from_bomb(field, self.last_bomb_positions)

        if new_dist > old_dist:
            events.append(e.MOVED_AWAY_FROM_OWN_BOMB)
        elif new_dist < old_dist:
            events.append(e.MOVED_TOWARDS_OWN_BOMB)

        if new_pos in own_blast_tiles:   
            events.append(e.STAYED_IN_OWN_BLAST) 
       
        if old_pos in own_blast_tiles and new_pos not in own_blast_tiles:
            events.append(e.ESCAPED_OWN_BOMB)

    
    # Update counters and train only every n environment steps.
    self.steps_done += 1
    events.append(e.STEP_PENALTY)  # Add a small penalty for each step to encourage faster completion
    reward = reward_from_events(self, events)

    # Store transition
    if old_state is not None:
        self.replay_buffer.append(Transition(old_state, self_action, new_state, reward, valid_action_mask(new_game_state)))

    if self.steps_done % TRAIN_EVERY == 0:
        for _ in range(TRAINING_STEPS):
            optimize_model(self)

    # Update target network periodically
    if self.steps_done % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Called at the end of each game to handle final transition and save model."""
    last_state = state_to_features(last_game_state, position_history=self.position_history)
    reward = reward_from_events(self, events)
    # terminal state: next_state is None
    if last_state is not None:
        self.replay_buffer.append(Transition(last_state, last_action, None, reward, None ))

    # Do some final optimization passes
    for _ in range(10):
        optimize_model(self)

    
    
    # Save the policy network
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save({"model_state_dict": self.policy_net.state_dict(), "steps_done": self.steps_done}, MODEL_FILE)


def reward_from_events(self, events: List[str]) -> float:
    game_rewards = {
        # Real outcomes dominate, heuristic shaping only nudges.
        e.KILLED_OPPONENT: 10.0,
        e.KILLED_SELF: -10.0,
        e.GOT_KILLED: -8.0,
        e.SURVIVED_ROUND: 2.0,

        # Coins and crates.
        e.COIN_COLLECTED: 4.0,
        e.COIN_FOUND: 1.0,
        e.CRATE_DESTROYED: 1.5,

        # Bomb placement heuristics.
        e.KILL_BOMB_DROPPED: 2.5,
        e.ENEMY_PRESSURE_BOMB_DROPPED: 1.0,
        e.CRATE_BOMB_DROPPED: 0.8,
        e.USELESS_BOMB_DROPPED: -4.0,

        # Escape behavior after our own bomb.
        e.MOVED_AWAY_FROM_OWN_BOMB: 0.8,
        e.MOVED_TOWARDS_OWN_BOMB: -1.2,
        e.STAYED_IN_OWN_BLAST: -2.5,
        e.ESCAPED_OWN_BOMB: 3.0,

        # Movement shaping.
        e.MOVED_CLOSE_TO_ENEMY: 0.4,
        e.MOVED_AWAY_FROM_ENEMY: -0.3,
        e.MOVED_CLOSE_TO_COIN: 0.2,
        e.MOVED_AWAY_FROM_COIN: -0.2,
        e.MOVED_TOWARDS_CRATE: 0.15,
        e.MOVED_AWAY_FROM_CRATE: -0.15,

        # Bad behavior and time pressure.
        e.INVALID_ACTION: -3.0,
        e.WAITED: -0.4,
        e.OSCILLATION: -1.0,
        e.STEP_PENALTY: -0.05,
        }
    reward_sum = 0.0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum:.2f} for events {', '.join(events)}")
    return reward_sum
