import os
import random
import logging

import numpy as np

from collections import deque



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "Ultra_Network_agent_saved_model_v2.pt")
MOVEMENT_DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIRECTIONS = [*MOVEMENT_DIRECTIONS, (0, 0)]  # UP, RIGHT, DOWN, LEFT, WAIT
ACTION_DELTAS = dict(zip(ACTIONS[:5], DIRECTIONS))

BOARD_CHANNELS = 14
VECTOR_DIM = 22
BOARD_SIZE = 17

BOMB_POWER = 3
BOMB_TIMER = 4

def valid_action_mask(game_state):
    field = game_state["field"]
    bombs = game_state["bombs"]
    _, _, bombs_left, (x, y) = game_state["self"]
    enemies = [pos for _, _, _, pos in game_state["others"] if pos is not None]

    mask = np.zeros(len(ACTIONS), dtype=bool)

    for i, action in enumerate(ACTIONS):
        if action == 'BOMB':
            useful_against_enemy = can_hit_enemy_with_bomb(field, x, y, enemies)
            useful_against_crate = can_hit_crate_with_bomb(field, x, y)

            mask[i] = (
                bombs_left > 0
                and (useful_against_enemy or useful_against_crate)
                and has_survival_path(
                    game_state,
                    first_action='WAIT',
                    place_bomb=True,
                )
            )
            continue

        mask[i] = has_survival_path(game_state, first_action=action)

    #fallback falls alles gefährlich ist, wenigstens alles was möglich ist erlauben

    if not mask.any():
        bomb_pos = set(pos for pos, _ in bombs)

        for i, action in enumerate(ACTIONS):
            if action in ['WAIT', 'BOMB']:
                mask[i] = False
                continue

            dx, dy = ACTION_DELTAS[action]

            nx, ny = x + dx, y + dy
            next_pos = (nx, ny)
            mask[i] = (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 0 and next_pos not in bomb_pos and next_pos not in enemies)

    if not mask.any():
        mask[ACTIONS.index('WAIT')] = True

    return mask


def explosion_tiles_from_bomb(field, bomb_position, bomb_power=BOMB_POWER):
    x, y = bomb_position
    explosion_tiles = set()
    explosion_tiles.add((x, y))

    for dx, dy in MOVEMENT_DIRECTIONS:
        for distance in range(1, bomb_power + 1):
            nx = x + dx * distance
            ny = y + dy * distance

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            explosion_tiles.add((nx, ny))
            # In this game variant crates are destroyed, but do not block
            # the blast. Only indestructible walls stop an explosion.

    return explosion_tiles

def bomb_danger_tiles(field, bombs, bomb_power=BOMB_POWER):
    danger_tiles = set()
    for bomb_position, _ in bombs:
        danger_tiles.update(explosion_tiles_from_bomb(field, bomb_position, bomb_power))
    return danger_tiles

def bomb_explosion_times(field, bombs, bomb_power=BOMB_POWER):
    explosion_times = {}
    for bomb_position, timer in bombs:
        explosion_tiles = explosion_tiles_from_bomb(field, bomb_position, bomb_power)
        for tile in explosion_tiles:
            if tile not in explosion_times or timer < explosion_times[tile]:
                explosion_times[tile] = timer
    return explosion_times


def build_hazard_timeline(field, bombs, explosion_map, horizon=BOMB_TIMER + 2):
    """Return dangerous tiles for each future environment step.

    Actions are applied before bombs and explosions are updated. Therefore a
    bomb whose displayed timer is zero becomes dangerous one action from now.
    An explosion remains dangerous for its creation step and the following
    action step in the current engine.
    """
    hazards = {step: set() for step in range(horizon + 1)}

    for x, y in np.argwhere(explosion_map > 0):
        remaining_steps = int(np.ceil(explosion_map[x, y]))
        for step in range(1, min(remaining_steps, horizon) + 1):
            hazards[step].add((int(x), int(y)))

    for bomb_position, timer in bombs:
        explosion_step = int(timer) + 1
        blast = explosion_tiles_from_bomb(field, tuple(bomb_position))

        for step in (explosion_step, explosion_step + 1):
            if 0 <= step <= horizon:
                hazards[step].update(blast)

    return hazards


def _has_survival_path(
    field,
    bombs,
    explosion_map,
    start,
    first_action=None,
    extra_bomb_position=None,
    blocked_first_step=(),
    horizon=BOMB_TIMER + 2,
):
    """Search the time-expanded board for a path surviving every hazard."""
    start = tuple(start)
    simulated_bombs = [
        (tuple(position), int(timer))
        for position, timer in bombs
    ]

    if (
        extra_bomb_position is not None
        and all(position != tuple(extra_bomb_position) for position, _ in simulated_bombs)
    ):
        simulated_bombs.append((tuple(extra_bomb_position), BOMB_TIMER))

    hazards = build_hazard_timeline(
        field,
        simulated_bombs,
        explosion_map,
        horizon,
    )

    # A bomb still occupies its tile during the action in which it explodes.
    bomb_expiry = {}
    for position, timer in simulated_bombs:
        expiry = timer + 1
        bomb_expiry[position] = min(bomb_expiry.get(position, expiry), expiry)

    blocked_first_step = {tuple(position) for position in blocked_first_step}
    start_bomb = start if start in bomb_expiry else None

    # position, future step, whether the bomb below the start was already left
    frontier = deque([(start, 0, False)])
    visited = {(start, 0, False)}

    while frontier:
        position, time, left_start_bomb = frontier.popleft()

        if time >= horizon:
            return True

        if time == 0 and first_action is not None:
            candidate_directions = (ACTION_DELTAS[first_action],)
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
            if next_time == 1 and next_position in blocked_first_step:
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

            search_state = (next_position, next_time, next_left_start_bomb)
            if search_state in visited:
                continue

            visited.add(search_state)
            frontier.append(search_state)

    return False


def has_survival_path(
    game_state,
    first_action=None,
    place_bomb=False,
    horizon=BOMB_TIMER + 2,
):
    """Return whether the selected action has at least one surviving follow-up."""
    if game_state is None:
        return False

    start = tuple(game_state["self"][3])
    enemies = [
        position
        for _, _, _, position in game_state["others"]
        if position is not None
    ]

    return _has_survival_path(
        game_state["field"],
        game_state["bombs"],
        game_state["explosion_map"],
        start,
        first_action=first_action,
        extra_bomb_position=start if place_bomb else None,
        blocked_first_step=enemies,
        horizon=horizon,
    )




def board_to_channels(game_state, safe_movement_mask=None):
    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    explosion_times = bomb_explosion_times(field, bombs)
    enemies = [pos for _, _, _, pos in game_state["others"] if pos is not None]
    _, _, _, (x, y) = game_state["self"]
    board = np.zeros((BOARD_CHANNELS, field.shape[0], field.shape[1]), dtype=np.float32)
    

    for _, _, _, (ex, ey) in game_state["others"]:
        if (ex, ey) is not None:
            board[9, ex, ey] = 1.0  # other agents

    for (tx, ty), timer in explosion_times.items():
        board[10, tx, ty] = 1.0 - (timer / BOMB_TIMER)  # normalized explosion times

        
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

    if safe_movement_mask is None:
        safe_movement_mask = [
            has_survival_path(game_state, first_action=action)
            for action in ACTIONS[:4]
        ]

    for (dx, dy), is_safe in zip(MOVEMENT_DIRECTIONS, safe_movement_mask):
        nx, ny = x + dx, y + dy
        if is_safe and 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            board[11, nx, ny] = 1.0  # movement has a time-safe continuation

    for tx, ty in explosion_tiles_from_bomb(field, (x, y)):
        board[12, tx, ty] = 1.0  # tiles that would be affected by a bomb at self position

    for ex, ey in enemies:
        if (ex, ey) is not None:
            for tx, ty in explosion_tiles_from_bomb(field, (ex, ey)):
                board[13, tx, ty] = 1.0  # tiles that would be affected by a bomb at enemy position


    #0 walls
    #1 crates
    #2 coins
    #3 self position
    #4 useful bomb positions
    #5 bomb timers
    #6 danger tiles
    #7 active explosion map
    #8 free space
    #9 enemy positions
    #10 normalized explosion times
    #11 immediately safe neighbors
    #12 tiles that would be affected by a bomb at self position
    #13 tiles that would be affected by a bomb at enemy position

    return board



def look_for_targets(free_space, start, targets, logger=None):
    if len(targets) == 0:
        return None

    frontier = deque([start])
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.popleft()
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] < best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            best = current
            break
        x, y = current
        neighbors = []
        for dx, dy in MOVEMENT_DIRECTIONS:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < free_space.shape[0]
                and 0 <= ny < free_space.shape[1]
                and free_space[nx, ny]
            ):
                neighbors.append((nx, ny))

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
    self.epsilon = 1.0
    self.decay_const = 569840
    self.position_history = deque(maxlen=4)
    self.last_features = None
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
            state = torch.load(MODEL_FILE, map_location=self.device, weights_only = False)

            if isinstance(state, dict) and 'model_state_dict' in state:
                model_state = state['model_state_dict']
                self.steps_done = int(round(float(state.get('steps_done', 0))))
            else:
                model_state = state

            self.policy_net.load_state_dict(model_state)
            self.target_net.load_state_dict(model_state)
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

def can_hit_enemy_with_bomb(field, x, y, enemies, bomb_power=BOMB_POWER):
    for dx, dy in MOVEMENT_DIRECTIONS:
        for distance in range(1, bomb_power + 1):
            nx = x + dx * distance
            ny = y + dy * distance

            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:
                break
            if (nx, ny) in enemies:
                return True

    return False

def enemy_has_escape_after_bomb(field, bombs, explosion_map, enemy_pos, bomb_pos, max_depth = BOMB_TIMER + 2):
    return _has_survival_path(
        field,
        bombs,
        explosion_map,
        enemy_pos,
        extra_bomb_position=bomb_pos,
        horizon=max_depth,
    )


def useful_bomb_positions(field):
    return [
        (x, y)
        for x in range(field.shape[0])
        for y in range(field.shape[1])
        if field[x, y] == 0 and can_hit_crate_with_bomb(field, x, y)
    ]

def has_escape_after_bomb(field, bombs, explosion_map, start, max_depth = BOMB_TIMER + 2):
    return _has_survival_path(
        field,
        bombs,
        explosion_map,
        start,
        extra_bomb_position=start,
        horizon=max_depth,
    )

def state_to_features(game_state: dict, position_history=None) -> np.array:
    if game_state is None:
        return None

    field = game_state["field"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    explosion_map = game_state["explosion_map"]
    _, score, bombs_left, (x, y) = game_state["self"]
    enemies = [pos for _, _, _, pos in game_state["others"] if pos is not None]

    

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

    safe_neighbors = [
        has_survival_path(game_state, first_action=action)
        for action in ACTIONS[:4]
    ]

    prev_dx, prev_dy = 0, 0
    two_back_dx, two_back_dy = 0, 0

    if position_history is not None and len(position_history) >= 2:
        px, py = position_history[-2]
        prev_dx = (px - x) / BOARD_SIZE
        prev_dy = (py - y) / BOARD_SIZE

    if position_history is not None and len(position_history) >= 3:
        tx, ty = position_history[-3]
        two_back_dx = (tx - x) / BOARD_SIZE
        two_back_dy = (ty - y) / BOARD_SIZE

    steps = game_state.get("step", 0)

    
    enemy_direction = look_for_targets(field == 0, (x, y), enemies)

    if enemy_direction is None:
        enemy_dx, enemy_dy = 0, 0
    else:
        enemy_dx = enemy_direction[0] - x
        enemy_dy = enemy_direction[1] - y

    if enemies:
        enemy_distance = [abs(enemy_dx) + abs(enemy_dy) for enemy_dx, enemy_dy in [(ex - x, ey - y) for ex, ey in enemies]]
        closest_enemy_distance = min(enemy_distance)
    else:
        closest_enemy_distance = BOARD_SIZE * 2  # Max possible distance on the board

    enemy_in_bomb_range = int(can_hit_enemy_with_bomb(field, x, y, enemies))
    safe_after_bomb = int(
        has_survival_path(
            game_state,
            first_action='WAIT',
            place_bomb=True,
        )
    )
    good_kill_opportunity = int(enemy_in_bomb_range and safe_after_bomb)
    enemy_trapped_by_bomb = int(any(
        can_hit_enemy_with_bomb(field, x, y, [(ex, ey)])
        and not enemy_has_escape_after_bomb(field, bombs, explosion_map, (ex, ey), (x, y))
        for ex, ey in enemies
    ))
    
    vector = np.array([
        coin_dx / BOARD_SIZE,
        coin_dy / BOARD_SIZE,
        crate_dx / BOARD_SIZE,
        crate_dy / BOARD_SIZE,
        int(can_hit_crate_with_bomb(field, x, y)),
        int(bombs_left),
        is_in_danger,
        int(any(safe_neighbors)),
        prev_dx,
        prev_dy,
        two_back_dx,
        two_back_dy,
        steps/400,
        closest_enemy_distance / BOARD_SIZE,
        enemy_in_bomb_range,
        safe_after_bomb,
        good_kill_opportunity,
        enemy_trapped_by_bomb,
        enemy_dx / BOARD_SIZE,
        enemy_dy / BOARD_SIZE,
        len(enemies)/3,
        score
    ], dtype=np.float32)

    board = board_to_channels(game_state, safe_neighbors)

    return board, vector



def act(self, game_state: dict) -> str:
    _, _, _, current_pos = game_state["self"]

    if not hasattr(self, "position_history"):
        self.position_history = deque(maxlen=4)

    if len(self.position_history) == 0 or self.position_history[-1] != current_pos:
        self.position_history.append(current_pos)

    state = state_to_features(game_state, self.position_history)
    self.last_features = state
    board, vector = state

    #self.logger.debug(f"Vector features: {vector}")

    mask = valid_action_mask(game_state)
    valid_actions = [action for action, valid in zip(ACTIONS, mask) if valid]
    epsilon = self.epsilon if self.train else 0.0

    if self.train and hasattr(self, 'steps_done'):
        epsilon = 0.05 + (1-0.05)*np.exp(-self.steps_done/self.decay_const)

    logging.getLogger('BombeRLeWorld').info(
        f'Agent <Ultra_network_agent> current epsilon {epsilon:.4f}'
    )

    # If we fell back to table model
    if hasattr(self, 'model'):
        if self.train and random.random() < epsilon:
            return random.choice(valid_actions)
        return random.choice(valid_actions)

    # use DQN policy
    try:
        import torch
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        vector_tensor = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Epsilon-greedy
        if self.train and random.random() < epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            qvals = self.policy_net(board_tensor, vector_tensor).squeeze(0)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
            qvals[~mask_tensor] = float('-inf')  # Mask invalid actions
            action_index = int(torch.argmax(qvals).item())
            return ACTIONS[action_index]
    except Exception as exc:
        self.logger.warning("DQN act failed, falling back to random: %s", exc)
        return random.choice(valid_actions)
