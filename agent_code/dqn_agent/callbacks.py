import os
import pickle
import random
import heapq

import numpy as np
import settings as s
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-saved-model.pt")
BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-best-model.pt")
PRETRAINED_MODEL_FILE = os.path.join(os.path.dirname(__file__), "dqn-pretrained-model.pt")
REPLAY_BUFFER_FILE = os.path.join(os.path.dirname(__file__), "dqn-replay-buffer.pkl")
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
VERBOSE_TRAIN_LOGS = False

# Training defaults to resuming MODEL_FILE. For a one-time transfer-learning
# start, run with DQN_TRAINING_MODE=transfer and optionally set
# DQN_PRETRAINED_MODEL=/path/to/model.pt. Use fresh to ignore all checkpoints.
TRAINING_START_MODES = {"resume", "transfer", "fresh"}

GRID_CHANNELS = 10  # number of channels in the grid input to the DQN
SCALAR_FEATURES = 8 + len(ACTIONS)  # base scalar features + one-hot last action




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

    frontier = deque([start])
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.popleft()
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
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
  
    try:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        self.device = None

    self.grid_channels = GRID_CHANNELS
    self.scalar_size = SCALAR_FEATURES

    self.logger.info(
        "Feature dimensions: grid_channels=%d, scalar_size=%d",
        self.grid_channels,
        self.scalar_size,
    )
    self.grid_size = (s.COLS, s.ROWS)
    self.log_dqn_details = VERBOSE_TRAIN_LOGS
    self._last_action_index = ACTIONS.index("WAIT")
    self._cached_state_key = None
    self._cached_features = None
    self._acted_state_key = None
    self._acted_features = None
    self._resume_checkpoint = None

    from .train import DQN_net

    self.policy_net = DQN_net(
        in_channels=self.grid_channels,
        grid_size=self.grid_size,
        scalar_size=self.scalar_size,
        n_actions=len(ACTIONS),
    ).to(self.device)

    if self.train:
        self.training_start_mode = os.environ.get("DQN_TRAINING_MODE", "resume").lower()
        if self.training_start_mode not in TRAINING_START_MODES:
            raise RuntimeError(
                "DQN_TRAINING_MODE must be one of: "
                + ", ".join(sorted(TRAINING_START_MODES))
            )

        if self.training_start_mode == "resume":
            load_path = MODEL_FILE
        elif self.training_start_mode == "transfer":
            load_path = os.environ.get("DQN_PRETRAINED_MODEL", PRETRAINED_MODEL_FILE)
        else:
            load_path = None
    elif os.path.isfile(BEST_MODEL_FILE):
        self.training_start_mode = None
        load_path = BEST_MODEL_FILE
    else:
        self.training_start_mode = None
        load_path = MODEL_FILE

    self.logger.info(
        "Model selection mode=%s training_start=%s path=%s exists=%s",
        "train" if self.train else "eval", self.training_start_mode,
        load_path,
        bool(load_path and os.path.isfile(load_path)),
    )

    if load_path and os.path.isfile(load_path):
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            policy_state_dict = checkpoint["policy_state_dict"]
        else:
            # Backward compatibility with existing weights-only model files.
            policy_state_dict = checkpoint
        try:
            self.policy_net.load_state_dict(policy_state_dict)
            if (
                self.train
                and self.training_start_mode == "resume"
                and isinstance(checkpoint, dict)
                and "policy_state_dict" in checkpoint
            ):
                self._resume_checkpoint = checkpoint
            self.logger.info("Loaded DQN model from %s", load_path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load model from {load_path} due to architecture mismatch: {exc}"
            ) from exc

    elif self.train and self.training_start_mode == "transfer":
        raise FileNotFoundError(
            f"Transfer-learning model not found: {load_path}. Set DQN_PRETRAINED_MODEL "
            "to the model you want to use or place it at PRETRAINED_MODEL_FILE."
        )

    elif self.train and self.training_start_mode == "resume":
        self.logger.warning(
            "No latest training checkpoint at %s; starting a fresh training run.",
            load_path,
        )
        self.training_start_mode = "fresh"

    elif load_path:
        self.logger.warning(
            "No saved DQN model found; using freshly initialized policy network."
        )

    self.policy_net.eval()



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
    """Reject bomb actions unless they can hit a crate or an enemy."""
    if game_state is None:
        return True

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    explosion_map = game_state.get('explosion_map', np.zeros_like(field))
    if explosion_map[x, y] > 0:
        return True

    blast = _blast_positions(field, x, y, s.BOMB_POWER)
    crates_in_range = any(field[cx, cy] == 1 for cx, cy in blast)
    enemies_in_range = any(
        (ox, oy) == (cx, cy)
        for (cx, cy) in blast
        for ox, oy in [o[-1] for o in game_state["others"] if o[-1] is not None]
    )
    return not crates_in_range and not enemies_in_range


def _policy_action_mask(game_state: dict) -> np.ndarray:
    """Return the action mask used by the DQN policy."""
    mask = np.array([_is_valid_action(game_state, action) for action in ACTIONS], dtype=bool)
    if game_state is not None and mask[ACTIONS.index('BOMB')] and _bomb_is_unsafe(game_state):
        mask[ACTIONS.index('BOMB')] = False
    return mask

def _feature_dimensions() -> tuple:
    """Return (grid_channels, scalar_size) from the fixed constants."""
    return GRID_CHANNELS, SCALAR_FEATURES


def _distance_map_on_free_space(
    free_space: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    """Compute shortest-path distances on walkable tiles with one BFS pass."""
    X, Y = free_space.shape
    distance_map = np.full((X, Y), fill_value=np.inf, dtype=np.float32)
    sx, sy = start
    distance_map[sx, sy] = 0.0

    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        current_distance = distance_map[cx, cy]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < X and 0 <= ny < Y):
                continue
            if not free_space[nx, ny]:
                continue
            if np.isfinite(distance_map[nx, ny]):
                continue
            distance_map[nx, ny] = current_distance + 1.0
            queue.append((nx, ny))
    return distance_map


def _nearest_distance_via_look_for_targets(
    distance_map: np.ndarray,
    free_space: np.ndarray,
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> int | None:
    """Use look_for_targets to validate target pursuit, then read nearest BFS distance."""
    if not targets:
        return None
    target_set = set(targets)
    if start in target_set:
        return 0

    next_step = look_for_targets(free_space, start, targets)
    if next_step is None or next_step == start:
        return None

    remaining = []
    for tx, ty in targets:
        dist = distance_map[tx, ty]
        if np.isfinite(dist):
            remaining.append(float(dist))
    if not remaining:
        return None
    return int(min(remaining))



def state_to_features(game_state: dict, last_action_index: int | None = None) -> np.array:
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
    explosion_map = game_state.get('explosion_map', np.zeros_like(field, dtype=np.float32))
    X, Y = field.shape

    walls = (field == -1).astype(int)
    crates = (field == 1).astype(int)

    coins_map = np.zeros((X, Y), dtype=np.float32)
    for cx, cy in coins:
        if 0 <= cx < X and 0 <= cy < Y:
            coins_map[cx, cy] = 1.0
    
    _, _, _, (x, y) = game_state["self"]

    self_map = np.zeros((X, Y), dtype=np.float32)
    if 0 <= x < X and 0 <= y < Y:
        self_map[x, y] = 1.0

    others = [o[-1] for o in game_state["others"] if o[-1] is not None]

    enemy_maps = []

    for i in range(s.MAX_AGENTS - 1):
        enemy_map = np.zeros((X, Y), dtype=np.float32)
        if i < len(others):
            ox, oy = others[i]

            if 0 <= ox < X and 0 <= oy < Y:
                enemy_map[ox, oy] = 1.0
        enemy_maps.append(enemy_map)

    bomb_map = np.zeros((X, Y), dtype=np.float32)
    bomb_timer_map = np.zeros((X, Y), dtype=np.float32)

    danger_time = np.full((X, Y), fill_value=np.inf, dtype=np.float32) #inf means safe, 0 means danger now, 1 means danger in 1 step, etc.
    danger_time[explosion_map > 0] = 0.0



    for (bomb_x, bomb_y), timer in bombs:
        if not (0 <= bomb_x < X and 0 <= bomb_y < Y):
            continue

        bomb_map[bomb_x, bomb_y] = 1.0

        bomb_timer_map[bomb_x, bomb_y] = float(timer) / max(1.0, float(s.BOMB_TIMER))

        if timer < danger_time[bomb_x, bomb_y]:
            danger_time[bomb_x, bomb_y] = float(timer)

        # Propagate blast from this bomb
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for step in range(1, s.BOMB_POWER + 1):
                nx = bomb_x + dx * step
                ny = bomb_y + dy * step
                if not (0 <= nx < X and 0 <= ny < Y):
                    break
                if field[nx, ny] == -1:
                    break
                explosion_time = float(timer)
                if explosion_time < danger_time[nx, ny]:
                    danger_time[nx, ny] = explosion_time
                if field[nx, ny] == 1:
                    break

    danger_map = np.zeros((X, Y), dtype=np.float32)
    finite_danger_time = np.isfinite(danger_time)

    MAX_DANGER_TIME = max(1.0, float(s.BOMB_TIMER))

    danger_map[finite_danger_time] = np.clip(1.0 - danger_time[finite_danger_time] / MAX_DANGER_TIME, 0.0, 1.0)
    '''
    x_coord = np.zeros((X, Y), dtype=np.float32)
    y_coord = np.zeros((X, Y), dtype=np.float32)

    for jx in range(X):
        x_coord[jx, :] = float(jx) / max(1.0, float(X - 1))
    for jy in range(Y):
        y_coord[:, jy] = float(jy) / max(1.0, float(Y - 1))
    '''

    grid = np.stack([
        walls,
        crates,
        coins_map,
        self_map,

        enemy_maps[0],
        enemy_maps[1],
        enemy_maps[2],

        bomb_map,
        bomb_timer_map,
        danger_map,
       # x_coord,
       # y_coord

    ], axis=0).astype(np.float32)


    bombs_left = 1.0 if game_state["self"][2] else 0.0

    coins_left_normalized = float(len(coins)) / (max(1.0, float(s.COIN_COUNT)))
    time_remaining = np.clip(1.0 - float(game_state["step"])/float(s.MAX_STEPS), 0.0, 1.0)

    enemies_remaining = float(len(others)) / max(1.0, float(s.MAX_AGENTS - 1))

    valid_coins = []
    for cx, cy in coins:
        if 0 <= cx < X and 0 <= cy < Y:
            valid_coins.append((cx, cy))
    if valid_coins:
        free_space = field == 0
        distance_map = _distance_map_on_free_space(free_space, (x, y))
        coin_distances = []
        for cx, cy in valid_coins:
            distance = distance_map[cx, cy]
            if np.isfinite(distance):
                coin_distances.append(float(distance))
    else:
        coin_distances = []

    if coin_distances and valid_coins:
        nearest_coin_distance = _nearest_distance_via_look_for_targets(
            distance_map,
            free_space,
            (x, y),
            valid_coins,
        )
        if nearest_coin_distance is None:
            nearest_coin_distance = float(min(coin_distances))
        mean_coin_distance = np.mean(coin_distances)
        farthest_coin_distance = max(coin_distances)
        reachable_coins_count = len(coin_distances)

    else:
        nearest_coin_distance = 0
        mean_coin_distance = 0
        farthest_coin_distance = 0
        reachable_coins_count = 0

    distance_normalizer = float(max(1, s.COLS + s.ROWS - 2))  # max distance in the grid

    nearest_coin_distance = np.clip(nearest_coin_distance / distance_normalizer, 0.0, 1.0)
    mean_coin_distance = np.clip(mean_coin_distance / distance_normalizer, 0.0, 1.0)
    farthest_coin_distance = np.clip(farthest_coin_distance / distance_normalizer, 0.0, 1.0)

    reachable_coins_fraction = float(reachable_coins_count) / max(1.0, len(coins))

    scalar = np.array([
        bombs_left,
        coins_left_normalized,
        time_remaining,
        enemies_remaining,

        reachable_coins_fraction,
        nearest_coin_distance,
        mean_coin_distance,
        farthest_coin_distance,
    ], dtype=np.float32)

    if last_action_index is None:
        last_action_index = ACTIONS.index("WAIT")
    last_action_index = int(np.clip(last_action_index, 0, len(ACTIONS) - 1))
    last_action_one_hot = np.zeros(len(ACTIONS), dtype=np.float32)
    last_action_one_hot[last_action_index] = 1.0
    scalar = np.concatenate([scalar, last_action_one_hot], axis=0).astype(np.float32)

    return grid, scalar


def state_to_features_cached(self, game_state: dict):
    """Cache the latest state->feature conversion to avoid duplicate work per step."""
    if game_state is None:
        return None

    step = game_state.get("step", None)
    self_state = game_state.get("self", None)
    self_pos = self_state[-1] if self_state is not None else None
    state_key = (id(game_state), step, self_pos, self._last_action_index)
    if (
        getattr(self, "_cached_state_key", None) == state_key
        and getattr(self, "_cached_features", None) is not None
    ):
        return self._cached_features

    feats = state_to_features(game_state, last_action_index=self._last_action_index)
    if feats is None:
        return None

    cached = feats
    self._cached_state_key = state_key
    self._cached_features = cached
    return cached


def _acted_state_key(game_state: dict | None):
    """Identify the state for which the policy most recently chose an action."""
    if game_state is None:
        return None

    self_state = game_state.get("self")
    self_pos = self_state[-1] if self_state is not None else None
    return game_state.get("round"), game_state.get("step"), self_pos


def features_used_for_action(self, game_state: dict):
    """Return the exact features on which the action for game_state was based.

    Recomputing these features after ``act`` is incorrect because ``act`` has
    already advanced ``_last_action_index`` to the selected action. That would
    put the selected action into the old-state input stored in replay.
    """
    if (
        getattr(self, "_acted_state_key", None) == _acted_state_key(game_state)
        and getattr(self, "_acted_features", None) is not None
    ):
        return self._acted_features

    self.logger.warning(
        "No recorded policy features for round=%s step=%s; recomputing old-state features.",
        game_state.get("round") if game_state is not None else None,
        game_state.get("step") if game_state is not None else None,
    )
    return state_to_features_cached(self, game_state)



def act(self, game_state: dict) -> str:

    feats = state_to_features_cached(self, game_state)
    if feats is None:
        self.logger.debug("Game state is None, returning WAIT")
        return 'WAIT'

    grid, scalar = feats

    # Keep the exact representation seen by the behavior policy. Training is
    # called after this method has updated _last_action_index, so recomputing
    # the old state there would leak the selected action into its input.
    self._acted_state_key = _acted_state_key(game_state)
    self._acted_features = feats

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

        evaluation_round = bool(
            self.train and getattr(self, "_evaluation_round", False)
        )
        epsilon = (
            self.get_epsilon()
            if self.train and not evaluation_round and hasattr(self, "get_epsilon")
            else 0.0
        )

        if self.train and not evaluation_round and random.random() < epsilon:
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
            self._last_action_index = ACTIONS.index(chosen_action)
            return chosen_action

        with torch.no_grad():
            q = self.policy_net(grid_t, scalar_t)  # shape (1, n_actions)
            action_mask = _policy_action_mask(game_state)
            for i, allowed in enumerate(action_mask):
                if not allowed:
                    q[0, i] = float('-inf')
            action_idx = int(q.argmax(dim=1).item())
            chosen_action = ACTIONS[action_idx]

        if getattr(self, "log_dqn_details", False):
            self.logger.info(
                f"Step {game_state.get('step', '?')}: danger_level={danger_level:.3f}, "
                f"danger_cells={danger_cells}, coins_remaining={coins_remaining}, action={chosen_action}, epsilon={epsilon:.3f}"
            )
        self._last_action_index = ACTIONS.index(chosen_action)
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
        self._last_action_index = ACTIONS.index(chosen_action)
        return chosen_action
