import os
import pickle
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-saved-model.pt")
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0), (0, 0)]  # Added (0, 0) for BOMB action

BOARD_CHANNELS = 9
BOARD_SIZE = 30
VECTOR_SIZE = 7

BOMB_POWER = 3
BOMB_TIMER = 4


def look_for_targets(free_space, start, targets, logger=None):
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug(f'Suitable target found at {best}')
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def setup(self):
    """Setup called once when loading the agent."""
    # hyperparams for acting
    self.epsilon = 0.3
    # Load model if available
    try:
        from .train import DQN
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = 7
        output_dim = len(ACTIONS)
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        if not self.train and os.path.isfile(MODEL_FILE):
            self.logger.info("Loading DQN model from disk.")
            state = torch.load(MODEL_FILE, map_location=self.device)
            self.policy_net.load_state_dict(state)
            self.target_net.load_state_dict(state)
            self.policy_net.eval()
            self.target_net.eval()
        else:
            self.logger.info("Initializing new DQN model.")
    except Exception as exc:
        # If torch isn't available, fall back to a simple table like q_agent
        self.logger.warning("Torch not available or failed to initialize DQN, falling back to table model: %s", exc)
        self.model = {}


def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    coins = game_state["coins"]
    _, _, _, (x, y) = game_state["self"]

    coin_direction = look_for_targets(field == 0, (x, y), coins)
    if coin_direction is None:
        coin_dx, coin_dy = 0, 0
    else:
        coin_dx = coin_direction[0] - x
        coin_dy = coin_direction[1] - y

    free_tiles = []
    for dx, dy in DIRECTIONS:
        if 0 <= x + dx < field.shape[0] and 0 <= y + dy < field.shape[1]:
            free_tiles.append(int(field[x + dx, y + dy] == 0))
        else:
            free_tiles.append(0)

    state = np.array([coin_dx, coin_dy, *free_tiles], dtype=int)
    return state


def act(self, game_state: dict) -> str:
    state = tuple(state_to_features(game_state))
    self.logger.debug(f"State features: {state}")

    # If we fell back to table model
    if hasattr(self, 'model'):
        if state not in self.model:
            self.model[state] = np.zeros(len(ACTIONS))
        if self.train and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return ACTIONS[np.argmax(self.model[state])]

    # use DQN policy
    try:
        import torch
        state_arr = np.array(state, dtype=np.float32)
        state_tensor = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Epsilon-greedy
        if self.train and random.random() < self.epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            qvals = self.policy_net(state_tensor)
            action_index = int(torch.argmax(qvals).item())
            return ACTIONS[action_index]
    except Exception as exc:
        self.logger.warning("DQN act failed, falling back to random: %s", exc)
        return random.choice(ACTIONS)
