import os
import pickle
import random

import numpy as np
import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-saved-model.pt")
BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-best-model.pt")
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
VERBOSE_TRAIN_LOGS = True


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


def setup(self):
    """Setup called once when loading the agent."""
    # hyperparams for acting
    # Initialize device and feature size metadata
    import settings as s
    try:
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        self.device = None

    # Infer feature sizes from the feature extractor so network inputs stay aligned.
    self.grid_channels, self.scalar_size = _feature_dimensions()
    self.log_dqn_details = VERBOSE_TRAIN_LOGS
    # Grid size (COLS, ROWS) as used by the environment
    self.grid_size = (s.COLS, s.ROWS)

    # Do not instantiate the full model here; create it lazily in act() or in setup_training.
    # Provide a fallback tabular model if torch isn't available.
    if self.device is None:
        self.logger.warning("Torch not available; falling back to table policy")
        self.model = {}
        return

    # Build policy network for both train and eval mode and load persisted weights if available.
    try:
        import torch
        from .train import DQN

        self.policy_net = DQN(
            in_channels=self.grid_channels,
            grid_size=self.grid_size,
            scalar_size=self.scalar_size,
            n_actions=len(ACTIONS),
        ).to(self.device)

        load_path = BEST_MODEL_FILE if os.path.isfile(BEST_MODEL_FILE) else MODEL_FILE
        if os.path.isfile(load_path):
            self.policy_net.load_state_dict(torch.load(load_path, map_location=self.device))
            self.logger.info("Loaded DQN model from %s", load_path)
        else:
            self.logger.warning("No saved DQN model found; using freshly initialized policy network.")

        self.policy_net.eval()
    except Exception as exc:
        self.logger.warning("Failed to initialize/load DQN model in setup: %s", exc)





def _blast_positions(field, x, y, power):
    """Return all cells affected by a bomb placed at (x, y)."""
    blast = {(x, y)}
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for step in range(1, power + 1):
            nx, ny = x + dx * step, y + dy * step
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            blast.add((nx, ny))
    return blast


def _bomb_is_unsafe(game_state: dict) -> bool:
    """Bomb placement is discouraged unless it can hit a crate or enemy."""
    if game_state is None:
        return True

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    explosion_map = game_state.get('explosion_map', np.zeros_like(field))
    if explosion_map[x, y] > 0:
        return True

    blast = _blast_positions(field, x, y, s.BOMB_POWER)
    crates_in_range = any(field[cx, cy] == 1 for cx, cy in blast)
    enemies_in_range = any((ox, oy) == (cx, cy) for (cx, cy) in blast for ox, oy in [o[-1] for o in game_state["others"] if o[-1] is not None])

    return not crates_in_range and not enemies_in_range


def _is_valid_action(game_state: dict, action: str) -> bool:
    if game_state is None:
        return False

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    others = [o[-1] for o in game_state["others"] if o[-1] is not None]
    bomb_positions = {pos for (pos, _) in game_state.get("bombs", [])}

    if action == 'UP':
        return y > 0 and field[x, y - 1] != -1 and field[x, y - 1] != 1 and (x, y - 1) not in others and (x, y - 1) not in bomb_positions
    if action == 'RIGHT':
        return x + 1 < field.shape[0] and field[x + 1, y] != -1 and field[x + 1, y] != 1 and (x + 1, y) not in others and (x + 1, y) not in bomb_positions
    if action == 'DOWN':
        return y + 1 < field.shape[1] and field[x, y + 1] != -1 and field[x, y + 1] != 1 and (x, y + 1) not in others and (x, y + 1) not in bomb_positions
    if action == 'LEFT':
        return x > 0 and field[x - 1, y] != -1 and field[x - 1, y] != 1 and (x - 1, y) not in others and (x - 1, y) not in bomb_positions
    if action == 'WAIT':
        return True
    if action == 'BOMB':
        return bool(game_state["self"][2])

    return False


def _policy_action_mask(game_state: dict) -> np.ndarray:
    """Return the action mask used by the DQN policy."""
    mask = np.array([_is_valid_action(game_state, action) for action in ACTIONS], dtype=bool)
    if game_state is not None and mask[ACTIONS.index('BOMB')] and _bomb_is_unsafe(game_state):
        mask[ACTIONS.index('BOMB')] = False
    return mask


def _feature_dimensions() -> tuple[int, int]:
    """Infer feature dimensions from the current feature extractor."""
    dummy_field = np.zeros((s.COLS, s.ROWS), dtype=np.int8)
    dummy_state = {
        "field": dummy_field,
        "coins": [],
        "bombs": [],
        "explosion_map": np.zeros_like(dummy_field),
        "self": ("agent", 0, True, (0, 0)),
        "others": [],
        "step": 0,
    }
    grid, scalar = state_to_features(dummy_state)
    return grid.shape[0], scalar.shape[0]


def state_to_features(game_state: dict) -> np.array:
    '''
    Returning (grid,scalar) for a given game_state 

    grid : np.array of shape (C, H, W) dtype = float32
    scalar : np.array of shape (scalar_size,) dtype = float32
    
    Return None if player is dead (game_state is None)
    '''
    if game_state is None:
        return None
    
    field = game_state["field"] # shape (cols, rows)
    coins = game_state["coins"]
    bombs = game_state["bombs"] # list of ((x,y), timer)
    explosion_map = game_state.get('explosion_map', np.zeros_like(field))
    danger_map = (explosion_map > 0).astype(np.float32)

    _, _, _, (x, y) = game_state["self"]
    others = [o[-1] for o in game_state["others"] if o[-1] is not None]
    H, W = field.shape

    walls = (field == -1).astype(int)
    crates = (field == 1).astype(int)
    coins_map = np.zeros_like(field, dtype=np.float32)
    for cx, cy in coins:
        coins_map[cx, cy] = 1.0

    others_map = np.zeros_like(field, dtype=np.float32)
    for ox, oy in others:
        others_map[ox, oy] = 1.0

    self_map = np.zeros_like(field, dtype=np.float32)
    self_map[x, y] = 1.0

    bomb_timer = np.zeros_like(field, dtype=np.float32)
    max_timer = 1.0
    for (bx, by), t in bombs:
        bomb_timer[bx, by] = t
        max_timer = max(max_timer, t)
    if max_timer > 0:
        bomb_timer = bomb_timer.astype(np.float32) / float(max_timer) # normalize to [0,1]

    x_coords = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1)).T
    y_coords = np.tile(np.linspace(0.0, 1.0, H, dtype=np.float32), (W, 1))

    grid = np.stack([
        walls,
        crates,
        coins_map,
        others_map,
        self_map,
        bomb_timer,
        danger_map,
        x_coords,
        y_coords
    ], axis=0).astype(np.float32)


    bombs_left = 1.0 if game_state["self"][2] else 0.0


    def nearest_target(targets):
        if len(targets) == 0:
            return H + W
        return min(abs(px - x) + abs(py - y) for px, py in targets)

    dist_coin = nearest_target(coins) / float(H + W)
    dist_enemy = nearest_target(others) / float(H + W)
    time_remaining = 1.0 - (game_state["step"] / float(s.MAX_STEPS))

    scalar = np.array([bombs_left, dist_coin, dist_enemy, time_remaining], dtype=np.float32)

    return grid, scalar



def act(self, game_state: dict) -> str:

    feats = state_to_features(game_state)
    if feats is None:
        self.logger.debug("Game state is None, returning WAIT")
        return 'WAIT'

    grid, scalar = feats

    field = game_state["field"]
    explosion_map = game_state.get('explosion_map', np.zeros_like(field))
    danger_level = float(np.max(explosion_map)) if explosion_map.size else 0.0
    danger_cells = int(np.count_nonzero(explosion_map > 0))
    coins_remaining = len(game_state.get('coins', []))

    # epsilon-greedy policy with DQN
    try:
        import torch
        grid_t = torch.from_numpy(grid).unsqueeze(0).to(self.device)  # shape (1, C, H, W)
        scalar_t = torch.from_numpy(scalar).unsqueeze(0).to(self.device)  # shape (1, S)

        if self.train:
            epsilon = self.get_epsilon()
            if random.random() < epsilon:
                valid_actions = [a for a, allowed in zip(ACTIONS, _policy_action_mask(game_state)) if allowed]
                if not valid_actions:
                    chosen_action = 'WAIT'
                else:
                    candidates = valid_actions
                    if np.max(explosion_map) == 0 and coins_remaining > 0:
                        non_wait_candidates = [a for a in candidates if a != 'WAIT']
                        if non_wait_candidates:
                            candidates = non_wait_candidates
                    chosen_action = random.choice(candidates)
                if getattr(self, "log_dqn_details", False):
                    self.logger.info(
                        f"Step {game_state.get('step', '?')}: danger_level={danger_level:.3f}, "
                        f"danger_cells={danger_cells}, coins_remaining={coins_remaining}, action={chosen_action}, epsilon={epsilon:.3f}"
                    )
                return chosen_action
        with torch.no_grad():
            q = self.policy_net(grid_t, scalar_t)  # shape (1, n_actions)
            action_mask = _policy_action_mask(game_state)
            for i, allowed in enumerate(action_mask):
                if not allowed:
                    q[0, i] = float('-inf')
            action_idx = int(q.argmax(dim=1).item())
            chosen_action = ACTIONS[action_idx]
            epsilon_value = self.get_epsilon() if hasattr(self, "get_epsilon") else 0.0
            if getattr(self, "log_dqn_details", False):
                self.logger.info(
                    f"Step {game_state.get('step', '?')}: danger_level={danger_level:.3f}, "
                    f"danger_cells={danger_cells}, coins_remaining={coins_remaining}, action={chosen_action}, epsilon={epsilon_value:.3f}"
                )
            return chosen_action
    except Exception as exc:
        self.logger.warning("DQN act failed, falling back to table or random policy : %s", exc)
        candidates = [a for a, allowed in zip(ACTIONS, _policy_action_mask(game_state)) if allowed]
        chosen_action = random.choice(candidates) if candidates else 'WAIT'
        if getattr(self, "log_dqn_details", False):
            self.logger.info(
                f"Step {game_state.get('step', '?')}: danger_level={danger_level:.3f}, "
                f"danger_cells={danger_cells}, coins_remaining={coins_remaining}, action={chosen_action}, fallback=random"
            )
        return chosen_action
