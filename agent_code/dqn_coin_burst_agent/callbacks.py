import os
import random

import numpy as np

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-coin-burst-saved-model.pt")
MOVEMENT_DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIRECTIONS = [*MOVEMENT_DIRECTIONS, (0, 0)]  # UP, RIGHT, DOWN, LEFT, WAIT

BOARD_CHANNELS = 9
VECTOR_DIM = 8
BOARD_SIZE = 17

BOMB_POWER = 3
BOMB_TIMER = 4

def bomb_danger_tiles(field, bombs, bomb_power=BOMB_POWER):
    danger = set()

    for (bx, by), timer in bombs:
        danger.add((bx, by))

        for dx, dy in MOVEMENT_DIRECTIONS:
            for distance in range(1, bomb_power + 1):
                nx = bx + dx * distance
                ny = by + dy * distance

                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    break
                if field[nx, ny] == -1:
                    break
                danger.add((nx, ny))

                if field[nx, ny] == 1:
                    break
    return danger

def board_to_channels(game_state):
    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    _, _, _, (x, y) = game_state["self"]

    board = np.zeros((BOARD_CHANNELS, field.shape[0], field.shape[1]), dtype=np.float32)

    board[0] = field == -1  # walls
    board[1] = field == 1   # crates
    board[8] = field == 0   # free space

    for cx, cy in coins:
        board[2, cx, cy] = 1.0  # coins

    board[3, x, y] = 1.0  # self position

    for px, py in useful_bomb_positions(field):
        board[4, px, py] = 1.0  # useful bomb positions

    danger = bomb_danger_tiles(field, bombs)

    for (bx, by), timer in bombs:
        board[5, bx, by] = timer / BOMB_TIMER  # bomb timer normalized

    for dx, dy in danger:
        board[6, dx, dy] = 1.0  # danger tiles

    board[7] = explosion_map > 0  # explosion map

    return board



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
        from .train import HybridDQN
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dim = len(ACTIONS)
        self.policy_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=output_dim).to(self.device)
        self.target_net = HybridDQN(board_channels=BOARD_CHANNELS, vector_dim=VECTOR_DIM, output_dim=output_dim).to(self.device)
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

def can_hit_crate_with_bomb(field, x, y, bomb_power=BOMB_POWER):
    for dx, dy in MOVEMENT_DIRECTIONS:
        for distance in range(1, bomb_power + 1):
            nx = x + dx * distance
            ny = y + dy * distance

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            if field[nx, ny] == 1:
                return True

    return False


def useful_bomb_positions(field):
    return [
        (x, y)
        for x in range(field.shape[0])
        for y in range(field.shape[1])
        if field[x, y] == 0 and can_hit_crate_with_bomb(field, x, y)
    ]

def state_to_features(game_state: dict) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    _, _, bombs_left, (x, y) = game_state["self"]

    coin_direction = look_for_targets(field == 0, (x, y), coins)
    if coin_direction is None:
        coin_dx, coin_dy = 0, 0
    else:
        coin_dx = coin_direction[0] - x
        coin_dy = coin_direction[1] - y

    crate_direction = look_for_targets(field == 0, (x, y), useful_bomb_positions(field))
    if crate_direction is None:
        crate_dx, crate_dy = 0, 0
    else:
        crate_dx = crate_direction[0] - x
        crate_dy = crate_direction[1] - y

    danger = bomb_danger_tiles(field, bombs)
    is_in_danger = int((x, y) in danger or explosion_map[x, y] > 0)

    safe_neighbors = []
    for dx, dy in MOVEMENT_DIRECTIONS:
        nx, ny = x + dx, y + dy
        safe_neighbors.append(
            0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0 and (nx, ny) not in danger and explosion_map[nx, ny] == 0)

    vector = np.array([
        coin_dx / BOARD_SIZE,
        coin_dy / BOARD_SIZE,
        crate_dx / BOARD_SIZE,
        crate_dy / BOARD_SIZE,
        int(can_hit_crate_with_bomb(field, x, y)),
        int(bombs_left),
        is_in_danger,
        int(any(safe_neighbors))]
        , dtype=np.float32
    )

    board = board_to_channels(game_state)

    return board, vector



def act(self, game_state: dict) -> str:
    state = state_to_features(game_state)
    board, vector = state

    self.logger.debug(f"State features: {state}")

    # If we fell back to table model
    if hasattr(self, 'model'):
        if self.train and random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return random.choice(ACTIONS)

    # use DQN policy
    try:
        import torch
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        vector_tensor = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        epsilon = self.epsilon if self.train else 0.0  # No exploration during evaluation

        if self.train and hasattr(self, 'steps_done'):
            epsilon = max(0.05, self.epsilon * (0.995 ** self.steps_done))  # Decay epsilon over time
        # Epsilon-greedy
        if self.train and random.random() < epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            qvals = self.policy_net(board_tensor, vector_tensor)
            action_index = int(torch.argmax(qvals).item())
            return ACTIONS[action_index]
    except Exception as exc:
        self.logger.warning("DQN act failed, falling back to random: %s", exc)
        return random.choice(ACTIONS)
