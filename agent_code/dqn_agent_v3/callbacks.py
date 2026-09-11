import os
import random
import heapq

import numpy as np
import settings as s
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

AGENT_DIRECTORY = os.path.dirname(__file__)

BEST_MODEL_FILE = os.path.join(
    AGENT_DIRECTORY,
    "dqn-current-best-model.pt",
)

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
NAVIGATION_DIRECTIONS = len(DIRECTIONS)

GRID_CHANNELS = 10  # number of channels in the grid input to the DQN
SCALAR_FEATURES = (
    8
    + NAVIGATION_DIRECTIONS  # first step toward nearest coin
    + NAVIGATION_DIRECTIONS  # first step toward nearest enemy
    + len(ACTIONS)  # one-hot last action
)
COIN_COUNT = 9  # Coins spawned in the tournament's classic scenario.




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
    from .train import (
        DQN_net,
        LATEST_CHECKPOINT_FILE,
        MODEL_ARCHITECTURE,
        PRETRAINED_MODEL_FILE,
        TRAINING_START_MODES,
        VERBOSE_TRAIN_LOGS,
    )

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

    self.policy_net = DQN_net(
        in_channels=self.grid_channels,
        grid_size=self.grid_size,
        scalar_size=self.scalar_size,
        n_actions=len(ACTIONS),
    ).to(self.device)


    self._resume_checkpoint = None

    if self.train:
        # Determine the training start mode based on the environment variable.
        self.training_start_mode = os.environ.get("DQN_TRAINING_MODE","resume",).strip().lower() 
        

        if self.training_start_mode not in TRAINING_START_MODES:
            raise RuntimeError("DQN_TRAINING_MODE must be one of: " + ", ".join(sorted(TRAINING_START_MODES)))

        if self.training_start_mode == "resume":
            load_path = LATEST_CHECKPOINT_FILE

        elif self.training_start_mode == "transfer":
            load_path = os.environ.get("DQN_PRETRAINED_MODEL", PRETRAINED_MODEL_FILE)

        else:
            # Fresh training loads no model
            load_path = None
    else:
        self.training_start_mode = None

        if os.path.isfile(BEST_MODEL_FILE):
            load_path = BEST_MODEL_FILE
        elif os.path.isfile(LATEST_CHECKPOINT_FILE):
            load_path = LATEST_CHECKPOINT_FILE
        elif os.path.isfile(PRETRAINED_MODEL_FILE):
            load_path = PRETRAINED_MODEL_FILE
        else:
            load_path = None

    self.logger.info("Model selection: train=%s start_mode=%s load_path=%s exists=%s",
                     self.train,
                     self.training_start_mode,
                     load_path,
                     bool(load_path and os.path.isfile(load_path)),)

    if load_path and os.path.isfile(load_path):
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=True)

        # Latest checkpoints contain complete training state. Best and
        # pretrained files may contain weights only.
        if (
        isinstance(checkpoint, dict)
        and "policy_state_dict" in checkpoint
        ):
            policy_state_dict = checkpoint["policy_state_dict"]
        else:
            policy_state_dict = checkpoint

        state_dict_compatible = (
            set(policy_state_dict) == set(self.policy_net.state_dict())
            and all(
                policy_state_dict[name].shape == parameter.shape
                for name, parameter in self.policy_net.state_dict().items()
            )
        )

        checkpoint_architecture = (
            checkpoint.get("model_architecture")
            if isinstance(checkpoint, dict)
            else None
        )

        if not state_dict_compatible:
            incompatibility = (
                f"checkpoint architecture={checkpoint_architecture!r}, "
                f"required architecture={MODEL_ARCHITECTURE!r}"
            )

            if self.train and self.training_start_mode == "transfer":
                raise RuntimeError(
                    f"Could not transfer model from {load_path}: {incompatibility}. "
                    "DQN_prev requires a compatible DQN_prev checkpoint."
                )

            self.logger.warning(
                "Ignoring incompatible model at %s (%s); starting with freshly "
                "initialized DQN_prev weights.",
                load_path,
                incompatibility,
            )

            if self.train:
                # Do not restore optimizer, epsilon, or replay state belonging
                # to the incompatible architecture.
                self.training_start_mode = "fresh"
        else:
            self.policy_net.load_state_dict(policy_state_dict)

        if (
            state_dict_compatible
            and self.train
            and self.training_start_mode == "resume"
            and isinstance(checkpoint, dict)
            and "policy_state_dict" in checkpoint
        ):
            self._resume_checkpoint = checkpoint

        if state_dict_compatible:
            self.logger.info("Loaded DQN model from %s", load_path)

    elif self.train and self.training_start_mode == "transfer":
        raise FileNotFoundError(
            f"Transfer model not found: {load_path}. "
            "Set DQN_PRETRAINED_MODEL to an existing model file."
        )

    elif self.train and self.training_start_mode == "resume":
        self.logger.warning(
            "No latest checkpoint found at %s; starting fresh.",
            LATEST_CHECKPOINT_FILE,
        )
        self.training_start_mode = "fresh"

    elif not self.train:
        self.logger.warning(
            "No best, latest, or pretrained model exists; using random weights."
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
    """Return all cells affected by a bomb placed at (x, y).

    This environment's blast implementation only stops at indestructible
    walls; it continues through crates.  Keep this helper aligned with
    ``items.Bomb.get_blast_coords`` because it is also used by the safety mask
    and the killer reward shaping.
    """
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


def _build_hazard_timeline(
    field,
    bombs,
    explosion_map,
    horizon=s.BOMB_TIMER + 2,
):
    """Return the blast tiles reached after each future action."""
    hazards = {step: set() for step in range(horizon + 1)}

    for x, y in np.argwhere(explosion_map > 0):
        remaining_steps = int(np.ceil(explosion_map[x, y]))
        for step in range(1, min(remaining_steps, horizon) + 1):
            hazards[step].add((int(x), int(y)))

    for bomb_position, timer in bombs:
        explosion_step = int(timer) + 1
        blast = _blast_positions(
            field,
            bomb_position[0],
            bomb_position[1],
            s.BOMB_POWER,
        )
        # An explosion is dangerous when created and for the following action.
        for step in (explosion_step, explosion_step + 1):
            if 0 <= step <= horizon:
                hazards[step].update(blast)

    return hazards


def _has_survival_path(
    game_state: dict,
    first_action: str | None = None,
    place_bomb: bool = False,
    horizon=s.BOMB_TIMER + 2,
) -> bool:
    """Search the time-expanded board for a safe continuation."""
    if game_state is None:
        return False

    field = game_state["field"]
    start = tuple(game_state["self"][-1])
    bombs = [
        (tuple(position), int(timer))
        for position, timer in game_state.get("bombs", ())
    ]
    if place_bomb and all(position != start for position, _ in bombs):
        bombs.append((start, s.BOMB_TIMER))

    hazards = _build_hazard_timeline(
        field,
        bombs,
        game_state.get("explosion_map", np.zeros_like(field)),
        horizon,
    )
    bomb_expiry = {
        position: timer + 1
        for position, timer in bombs
    }
    enemies = {
        other[-1]
        for other in game_state.get("others", ())
        if other[-1] is not None
    }
    start_bomb = start if start in bomb_expiry else None

    # position, future step, whether a bomb underneath the agent was left
    frontier = deque([(start, 0, False)])
    visited = {(start, 0, False)}

    while frontier:
        position, time, left_start_bomb = frontier.popleft()
        if time >= horizon:
            return True

        if time == 0 and first_action is not None:
            if first_action in ACTIONS[:4]:
                candidate_directions = (DIRECTIONS[ACTIONS.index(first_action)],)
            else:
                candidate_directions = ((0, 0),)
        else:
            candidate_directions = DIRECTIONS

        for dx, dy in candidate_directions:
            next_position = (position[0] + dx, position[1] + dy)
            next_time = time + 1
            nx, ny = next_position

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0:
                continue
            if next_time == 1 and next_position in enemies:
                continue

            next_left_start_bomb = (
                left_start_bomb
                or (start_bomb is not None and next_position != start_bomb)
            )
            expiry = bomb_expiry.get(next_position)
            if expiry is not None and next_time <= expiry:
                may_remain_on_start_bomb = (
                    next_position == start_bomb and not left_start_bomb
                )
                if not may_remain_on_start_bomb:
                    continue
            if next_position in hazards[next_time]:
                continue

            search_state = (
                next_position,
                next_time,
                next_left_start_bomb,
            )
            if search_state in visited:
                continue
            visited.add(search_state)
            frontier.append(search_state)

    return False


def _bomb_has_escape_route(game_state: dict) -> bool:
    """Return whether a newly placed bomb has a timed survival route."""
    return _has_survival_path(
        game_state,
        first_action="WAIT",
        place_bomb=True,
    )


def _bomb_targets_enemy(game_state: dict) -> bool:
    """Return whether a bomb at the agent position currently covers an enemy."""
    if game_state is None:
        return False

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    blast = _blast_positions(field, x, y, s.BOMB_POWER)
    return any(
        other[-1] in blast
        for other in game_state.get("others", ())
        if other[-1] is not None
    )


def _enemy_can_escape_blast(
    game_state: dict,
    enemy_position: tuple[int, int],
    blast: set[tuple[int, int]],
) -> bool:
    """Conservatively test whether an enemy can leave a prospective blast."""
    field = game_state["field"]
    blocked = {
        position
        for position, _ in game_state.get("bombs", ())
    }
    blocked.update(
        other[-1]
        for other in game_state.get("others", ())
        if other[-1] is not None and other[-1] != enemy_position
    )

    frontier = deque([(enemy_position, 0)])
    visited = {enemy_position}

    while frontier:
        position, distance = frontier.popleft()
        if distance > 0 and position not in blast:
            return True
        if distance >= s.BOMB_TIMER:
            continue

        x, y = position
        for dx, dy in DIRECTIONS[:4]:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                continue
            if field[nx, ny] != 0 or neighbor in blocked:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            frontier.append((neighbor, distance + 1))

    return False


def _bomb_traps_enemy(game_state: dict) -> bool:
    """Return whether a bomb at self covers an enemy with no static escape."""
    if game_state is None:
        return False

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    blast = _blast_positions(field, x, y, s.BOMB_POWER)

    return any(
        enemy_position in blast
        and not _enemy_can_escape_blast(game_state, enemy_position, blast)
        for enemy_position in (
            other[-1]
            for other in game_state.get("others", ())
            if other[-1] is not None
        )
    )


def _bomb_is_unsafe(game_state: dict) -> bool:
    """Reject bombs that are useless or leave no timed escape route."""
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
    useful = crates_in_range or enemies_in_range
    return not useful or not _bomb_has_escape_route(game_state)


def _policy_action_mask(game_state: dict) -> np.ndarray:
    """Mask illegal actions and choices without a timed survival path."""
    legal_mask = np.array(
        [_is_valid_action(game_state, action) for action in ACTIONS],
        dtype=bool,
    )

    safe_mask = legal_mask.copy()
    for index, action in enumerate(ACTIONS):
        if not safe_mask[index]:
            continue
        if action == "BOMB":
            safe_mask[index] = not _bomb_is_unsafe(game_state)
        else:
            safe_mask[index] = _has_survival_path(
                game_state,
                first_action=action,
            )

    # In a fully trapped state, retain the legal mask so the DQN still emits a
    # valid game action and can learn from the unavoidable terminal outcome.
    return safe_mask if safe_mask.any() else legal_mask

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


def _direction_to_nearest_target(
    free_space: np.ndarray,
    start: tuple[int, int],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    """Return UP/RIGHT/DOWN/LEFT/NONE for the nearest reachable target."""
    direction = np.zeros(NAVIGATION_DIRECTIONS, dtype=np.float32)
    no_target_index = NAVIGATION_DIRECTIONS - 1

    if not targets:
        direction[no_target_index] = 1.0
        return direction

    target_set = set(targets)
    frontier = deque([start])
    first_step = {start: start}

    while frontier:
        current = frontier.popleft()
        if current in target_set and current != start:
            step = first_step[current]
            delta = (step[0] - start[0], step[1] - start[1])
            try:
                direction[DIRECTIONS.index(delta)] = 1.0
            except ValueError:
                direction[no_target_index] = 1.0
            return direction

        cx, cy = current
        for dx, dy in DIRECTIONS[:4]:
            neighbor = (cx + dx, cy + dy)
            nx, ny = neighbor
            if not (0 <= nx < free_space.shape[0] and 0 <= ny < free_space.shape[1]):
                continue
            if not free_space[nx, ny] or neighbor in first_step:
                continue
            first_step[neighbor] = neighbor if current == start else first_step[current]
            frontier.append(neighbor)

    direction[no_target_index] = 1.0
    return direction



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

    coins_left_normalized = float(len(coins)) / max(1.0, float(COIN_COUNT))
    time_remaining = np.clip(1.0 - float(game_state["step"])/float(s.MAX_STEPS), 0.0, 1.0)

    enemies_remaining = float(len(others)) / max(1.0, float(s.MAX_AGENTS - 1))

    bomb_positions = {
        position for position, _timer in bombs
    }
    free_space = field == 0
    for blocked_position in bomb_positions.union(others):
        bx, by = blocked_position
        if 0 <= bx < X and 0 <= by < Y:
            free_space[bx, by] = False
    free_space[x, y] = True

    valid_coins = []
    for cx, cy in coins:
        if 0 <= cx < X and 0 <= cy < Y:
            valid_coins.append((cx, cy))
    if valid_coins:
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

    # Navigation directions use shortest paths instead of straight-line
    # offsets, so walls and crates are respected. Bombs and agents are treated
    # as temporary blockers for coin pursuit. Enemy cells remain valid targets
    # in the enemy-specific search.
    coin_direction = _direction_to_nearest_target(
        free_space,
        (x, y),
        valid_coins,
    )

    valid_enemies = [
        (ex, ey)
        for ex, ey in others
        if 0 <= ex < X and 0 <= ey < Y
    ]
    enemy_free_space = field == 0
    for bx, by in bomb_positions:
        if 0 <= bx < X and 0 <= by < Y:
            enemy_free_space[bx, by] = False
    for ex, ey in valid_enemies:
        enemy_free_space[ex, ey] = True
    enemy_free_space[x, y] = True
    enemy_direction = _direction_to_nearest_target(
        enemy_free_space,
        (x, y),
        valid_enemies,
    )

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
    scalar = np.concatenate(
        [scalar, coin_direction, enemy_direction, last_action_one_hot],
        axis=0,
    ).astype(np.float32)

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

def _game_state_key(game_state: dict | None):
    """Identify a state across act() and the subsequent training callback."""
    if game_state is None:
        return None

    self_state = game_state.get("self")
    self_position = self_state[-1] if self_state is not None else None

    return (
        game_state.get("round"),
        game_state.get("step"),
        self_position,
    )


def features_used_for_action(self, game_state: dict):
    """Return the exact features that act() used for this state."""
    if game_state is None:
        return None

    if (
        getattr(self, "_acted_state_key", None) == _game_state_key(game_state)
        and getattr(self, "_acted_features", None) is not None
    ):
        return self._acted_features

    # This should only occur if act() was not called for this state.
    self.logger.warning(
        "No saved action features for round=%s step=%s; recomputing features.",
        game_state.get("round"),
        game_state.get("step"),
    )
    return state_to_features_cached(self, game_state)



def act(self, game_state: dict) -> str:

    feats = state_to_features_cached(self, game_state)
    if feats is None:
        self.logger.debug("Game state is None, returning WAIT")
        return 'WAIT'

    self._acted_state_key = _game_state_key(game_state)
    self._acted_features = feats

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

        if getattr(self, "_evaluation_round", False):
            epsilon = 0.0
        elif self.train and hasattr(self, "get_epsilon"):
            epsilon = self.get_epsilon()
        else:
            epsilon = 0.0
            
        if self.train and random.random() < epsilon:
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
