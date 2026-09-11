import os
import pickle
import random
from collections import deque, namedtuple

import numpy as np

import settings as s


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

DIRECTIONS = [((0, -1), 0), ((1, 0), 1), ((0, 1), 2), ((-1, 0), 3)]

Q_TABLE_FILE = os.path.join(
    os.path.dirname(__file__),
    "q_table.pkl"
)

StateFeatures = namedtuple("StateFeatures", [
    "target_direction",  # 0-3: Richtung zum aktuellen Nahziel, 4: kein Ziel/schon da
    "free_up",
    "free_right",
    "free_down",
    "free_left",
    "bomb_available",
    "crate_adjacent",
    "in_danger",
    "coin_visible",
    "safe_bomb",
    "movement_pattern",
])


def valid_actions_from_state(state):

    valid_actions = []

    if state.free_up:
        valid_actions.append("UP")

    if state.free_right:
        valid_actions.append("RIGHT")

    if state.free_down:
        valid_actions.append("DOWN")

    if state.free_left:
        valid_actions.append("LEFT")


    valid_actions.append("WAIT")

    if state.bomb_available:
        valid_actions.append("BOMB")

    return valid_actions


def setup(self):
    self.epsilon = 0.1

    if os.path.isfile(Q_TABLE_FILE):
        self.logger.info("Loading Q-table.")
        with open(Q_TABLE_FILE, "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.logger.info("Starting with empty Q-table.")
        self.q_table = {}

    self.current_target_position = None
    self.current_target_kind = None

    self.last_state = None
    self.last_target_position = None
    self.last_distance = None
    self.position_history = deque(maxlen=6)


def detect_movement_pattern(position_history, new_position=None):

    positions = list(position_history)

    if new_position is not None:
        positions.append(tuple(new_position))

    if len(positions) >= 6:
        recent_positions = positions[-6:]
        unique_positions = set(recent_positions)

        if len(unique_positions) == 1:
            return 2

        if len(unique_positions) == 2:
            return 3

    if len(positions) >= 3:
        pos_a, pos_b, pos_c = positions[-3:]

        # A -> A -> A
        if pos_a == pos_b == pos_c:
            return 2

        if pos_a == pos_c and pos_a != pos_b:
            return 1

    return 0


def act(self, game_state: dict) -> str:

    current_position = tuple(game_state["self"][3])
    self.position_history.append(current_position)
    movement_pattern = detect_movement_pattern(
        self.position_history
    )

    update_current_target(self, game_state)

    state = state_to_features(
        game_state,
        self.current_target_position,
        movement_pattern=movement_pattern,
    )

    self.last_state = state
    self.last_target_position = self.current_target_position
    self.last_distance = distance_to_target(
        game_state,
        self.current_target_position
    )

    valid_actions = valid_actions_from_state(state)

    if self.train and random.random() < self.epsilon:
        return random.choice(valid_actions)

    q_values = {
        action: self.q_table.get((state, action), 0.0)
        for action in valid_actions
    }

    max_q = max(q_values.values())

    best_actions = [
        action
        for action, value in q_values.items()
        if value == max_q
    ]

    return random.choice(best_actions)


def state_to_features(game_state: dict, target_position=None, movement_pattern=0):

    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_available, (x, y) = game_state["self"]

    danger_tiles = compute_danger_tiles(game_state)
    in_danger = (x, y) in danger_tiles

    free_up = is_free(game_state, x, y - 1)
    free_right = is_free(game_state, x + 1, y)
    free_down = is_free(game_state, x, y + 1)
    free_left = is_free(game_state, x - 1, y)

    crate_adjacent = has_adjacent_crate(field, (x, y))
    coin_visible = len(game_state["coins"]) > 0

    safe_bomb = False
    if bomb_available and not in_danger:
        safe_bomb = can_escape_own_bomb(game_state, danger_tiles)

    target_direction, _ = get_target_direction_and_distance(
        game_state, danger_tiles, in_danger, target_position,
    )

    return StateFeatures(
        target_direction,
        free_up,
        free_right,
        free_down,
        free_left,
        bool(bomb_available),
        crate_adjacent,
        in_danger,
        coin_visible,
        safe_bomb,
        movement_pattern,
    )

def distance_to_target(game_state, target_position=None):

    if game_state is None:
        return None

    _, _, _, (x, y) = game_state["self"]
    danger_tiles = compute_danger_tiles(game_state)
    in_danger = (x, y) in danger_tiles

    _, distance = get_target_direction_and_distance(
        game_state, danger_tiles, in_danger, target_position,
    )
    return distance


def get_target_direction_and_distance(
    game_state,
    danger_tiles,
    in_danger,
    target_position=None
):


    if in_danger:
        return direction_to_safety(game_state, danger_tiles)

    if target_position is not None:
        return direction_to_position(game_state, target_position)


    if game_state["coins"]:
        return direction_to_coin(game_state)

    return direction_to_crate(game_state)


def is_free(game_state, x, y):


    field = game_state["field"]

    if x < 0 or y < 0:
        return False

    if x >= field.shape[0] or y >= field.shape[1]:
        return False

    if field[x, y] != 0:
        return False

    bomb_positions = {
        position
        for position, _timer in game_state["bombs"]
    }

    if (x, y) in bomb_positions:
        return False

    other_positions = {
        position
        for _name, _score, _bombs_left, position in game_state["others"]
    }

    if (x, y) in other_positions:
        return False

    return True


def has_adjacent_crate(field, position):
    x, y = position

    for (dx, dy), _direction_id in DIRECTIONS:
        nx, ny = x + dx, y + dy

        if (
            0 <= nx < field.shape[0]
            and 0 <= ny < field.shape[1]
            and field[nx, ny] == 1
        ):
            return True

    return False



def bfs_target_position(game_state, is_goal, block_danger=True):

    _, _, _, start = game_state["self"]

    danger_tiles = (
        compute_danger_tiles(game_state)
        if block_danger
        else set()
    )

    if is_goal(start):
        return start

    queue = deque([start])
    visited = {start}

    while queue:
        x, y = queue.popleft()

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)

            if next_position in visited:
                continue

            if not is_free(game_state, nx, ny):
                continue

            if next_position in danger_tiles:
                continue

            visited.add(next_position)

            if is_goal(next_position):
                return next_position

            queue.append(next_position)

    return None


def update_current_target(self, game_state):

    if game_state is None:
        return

    field = game_state["field"]
    _, _, bomb_available, position = game_state["self"]

    danger_tiles = compute_danger_tiles(game_state)

    def is_valid_crate_target(pos):
        if not has_adjacent_crate(field, pos):
            return False

        if pos in danger_tiles:
            return False

        if not bomb_available:
            return True

        return can_escape_bomb_from_position(
            game_state,
            pos,
            danger_tiles
        )


    if position in danger_tiles:
        return

    coins = set(game_state["coins"])



    if coins:
        if (
            self.current_target_kind == "coin"
            and self.current_target_position in coins
        ):
            return

        new_target = bfs_target_position(
            game_state,
            lambda pos: pos in coins,
            block_danger=True,
        )

        self.current_target_position = new_target
        self.current_target_kind = "coin" if new_target is not None else None
        return


    if (
            self.current_target_kind == "crate"
            and self.current_target_position is not None
            and is_valid_crate_target(self.current_target_position)
    ):
        return

    new_target = bfs_target_position(
        game_state,
        is_valid_crate_target,
        block_danger=True,
    )

    self.current_target_position = new_target
    self.current_target_kind = "crate" if new_target is not None else None



def bfs_target(game_state, is_goal, block_danger, danger_tiles=None):

    _, _, _, start = game_state["self"]

    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state) if block_danger else set()
    elif not block_danger:
        danger_tiles = set()

    if is_goal(start):
        return 4, 0

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in DIRECTIONS:
        nx, ny = start[0] + dx, start[1] + dy

        if is_free(game_state, nx, ny) and (nx, ny) not in danger_tiles:
            visited.add((nx, ny))
            queue.append(((nx, ny), direction_id, 1))

    while queue:
        position, first_direction, distance = queue.popleft()

        if is_goal(position):
            return first_direction, distance

        x, y = position

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = x + dx, y + dy
            next_position = (nx, ny)

            if (
                next_position not in visited
                and is_free(game_state, nx, ny)
                and next_position not in danger_tiles
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction, distance + 1))

    return 5, None


def direction_to_coin(game_state):
    coins = set(game_state["coins"])
    return bfs_target(game_state, lambda pos: pos in coins, block_danger=True)


def direction_to_crate(game_state):
    field = game_state["field"]
    return bfs_target(
        game_state,
        lambda pos: has_adjacent_crate(field, pos),
        block_danger=True,
    )


def direction_to_safety(game_state, danger_tiles=None):
    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state)

    return bfs_target(
        game_state,
        lambda pos: pos not in danger_tiles,
        block_danger=False,
    )


def direction_to_position(game_state, target_position):
    if target_position is None:
        return 4, None

    return bfs_target(
        game_state,
        lambda pos: pos == target_position,
        block_danger=True,
    )


def get_blast_coords(field, x, y, power):

    blast_coords = [(x, y)]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i

            if field[nx, ny] == -1:
                break

            blast_coords.append((nx, ny))

    return blast_coords


def compute_danger_tiles(game_state):

    field = game_state["field"]
    danger_tiles = set()

    explosion_map = game_state["explosion_map"]
    xs, ys = np.nonzero(explosion_map > 0)
    danger_tiles.update(zip(xs.tolist(), ys.tolist()))

    for (bx, by), _timer in game_state["bombs"]:
        danger_tiles.update(get_blast_coords(field, bx, by, s.BOMB_POWER))

    return danger_tiles


def can_escape_bomb_from_position(game_state, start, danger_tiles=None):

    field = game_state["field"]

    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state)

    own_blast = set(
        get_blast_coords(
            field,
            start[0],
            start[1],
            s.BOMB_POWER
        )
    )

    combined_danger = danger_tiles | own_blast

    max_steps = max(s.BOMB_TIMER - 1, 1)

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (px, py), distance = queue.popleft()

        if (px, py) not in combined_danger:
            return True

        if distance >= max_steps:
            continue

        for (dx, dy), _direction_id in DIRECTIONS:
            nx, ny = px + dx, py + dy
            next_position = (nx, ny)

            if (
                next_position not in visited
                and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, distance + 1))

    return False

def can_escape_own_bomb(game_state, danger_tiles=None):
    _, _, bomb_available, position = game_state["self"]

    if not bomb_available:
        return False

    return can_escape_bomb_from_position(
        game_state,
        position,
        danger_tiles
    )
