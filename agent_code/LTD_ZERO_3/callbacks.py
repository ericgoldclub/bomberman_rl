import os
import pickle
import random
from collections import deque

import numpy as np


"""
Game state dictionary format

Keys (key : type) — description:

- 'round' : int
    The number of rounds since launching the environment, starting at 1.

- 'step' : int
    The number of steps in the episode so far, starting at 1.

- 'field' : numpy.ndarray (shape: width x height)
    A 2D numpy array describing the tiles of the game board.
    Tile values:
      1  = crate
     -1  = stone wall
      0  = free tile
    Note: image coordinates are (x, y); printing game_state['field'] will show the board transposed compared to the GUI.

- 'bombs' : list[tuple[(int, int), int]]
    A list of bombs currently active. Each entry is ((x, y), t) where:
      (x, y) = bomb coordinates (ints)
      t      = countdown in steps until explosion (int)
    A countdown of 0 means the bomb is about to explode.

- 'explosion_map' : numpy.ndarray (shape: width x height)
    A 2D numpy array where each tile value is the number of remaining steps
    an explosion will be present on that tile. Tiles with no explosion have value 0.

- 'coins' : list[tuple[int, int]]
    A list of (x, y) coordinates for all currently collectable coins.

- 'self' : tuple[str, int, bool, tuple[int, int]]
    A tuple describing your agent: (name, score, bomb_possible, (x, y))
      name (str)          = agent's name
      score (int)         = current score
      bomb_possible (bool)= True if the BOMB action is currently allowed
                            (i.e., the agent has no own bomb currently ticking)
      (x, y) (tuple)      = agent's current coordinates on the field

- 'others' : list[tuple[str, int, bool, tuple[int, int]]]
    A list of tuples like 'self' for each opponent still in the game:
    (name, score, bomb_possible, (x, y))

- 'user_input' : str | None
    User input from the GUI; may be None when there is no input.

Example (illustrative, not runnable as-is):

game_state = {
    'round': 1,                          # int
    'step': 1,                           # int
    'field': np.zeros((width, height)),  # np.ndarray of ints: 1/crate, -1/stone, 0/free
    'bombs': [((3, 5), 2), ((7, 2), 0)], # list of ((x,y), countdown)
    'explosion_map': np.zeros((width, height)), # np.ndarray of ints
    'coins': [(2, 4), (8, 1)],           # list of (x, y)
    'self': ("agent_1", 10, True, (4, 4)),
    'others': [("opponent", 5, False, (1, 1))],
    'user_input': None
}
"""

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
MOVES = ['UP', 'RIGHT', 'DOWN', 'LEFT']
MAX_DIST = 21.0


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = np.zeros(11)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.action = 'DOWN'



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    if self.train:
        return self.action
    else:
        return policy(game_state,self)

def action_state_to_features(game_state:dict, action) -> np.array:
    if game_state is None:
        return None
    _, _, bomb, (x, y) = game_state["self"]
    explosion_map = game_state["explosion_map"]
    features = []
    danger_tiles = compute_danger_tiles(game_state)
    on_coin = (x, y) in set(game_state["coins"])
    coin_direction, crate_direction, safety_direction, opponent_direction = compute_directions(game_state, x, y, danger_tiles)
    coin = 0
    crate = 0
    if ACTIONS[crate_direction] == action:
        crate = 1
    else:
        crate = 0
    if on_coin:
        coin = 1
    elif coin_direction != 4:
        if ACTIONS[coin_direction] == action:
            coin = 1
        else:
            coin = 0

    if (x,y) not in danger_tiles:
        safety = 0
    else:
        if ACTIONS[safety_direction] == action:
            safety = 1
        else:
            safety = 0

    opponent = 0
    if opponent_direction != 4:
        if ACTIONS[opponent_direction] == action:
            opponent = 1
        else:
            opponent = 0
    xold,yold = x,y
    x,y = move_cords(action,game_state)
    free = is_free(game_state,x,y)
    danger = (x,y) in danger_tiles
    planted_next_to_crate = 0
    if is_next_to_crate(x,y,game_state) == 1 and action =="BOMB":
        planted_next_to_crate = 1
    bomb_threatens_opponent = 0
    if action == "BOMB" and bomb_endangers_nearest_opponent(game_state, xold, yold):
        bomb_threatens_opponent = 1
    if free:
        will_die = gonna_die(game_state,x,y)
    else:
        will_die = gonna_die(game_state,xold,yold)
    features.append(safety)
    features.append(bomb)
    features.append(will_die)
    features.append(danger)
    features.append(planted_next_to_crate)
    features.append(free)
    features.append(coin)
    features.append(crate)
    features.append(opponent)
    features.append(bomb_threatens_opponent)
    features.append(1)

    return np.array(features)

def is_next_to_crate(x,y,gamestate):
    field = gamestate['field']
    next_to_crate = 0
    if 0 < x and x < field.shape[0] - 1:
        if field[x+1][y] == 1:
            next_to_crate = 1
        if field[x-1][y] == 1:
            next_to_crate = 1
    if 0 < y and y < field.shape[1] - 1:
        if field[x][y+1] == 1:
            next_to_crate = 1
        if field[x][y-1] == 1:
            next_to_crate = 1
    return next_to_crate
def is_free(game_state, x, y):
    field = game_state['field']
    bombs_all = game_state['bombs']
    bombs = [bomb[0] for bomb in bombs_all]
    if x < 0 or y < 0:
        return False
    if x >= field.shape[0] or y >= field.shape[1]:
        return False
    return field[x,y] == 0 and (x,y) not in bombs

def move_cords(action,game_state):
    _, _, _, (x, y) = game_state["self"]
    match action:
        case 'UP':
            y = y-1
        case 'DOWN':
            y = y+1
        case 'LEFT':
            x = x-1
        case 'RIGHT':
            x = x+1
        case 'WAIT':
            pass
        case 'BOMB':
            pass
    return x,y


def can_destroy_crate(x, y, game_state):
    field = game_state['field']
    if field[x][y] != 0: return False
    for i in range(4):
        if field[x+i,y] == -1: break
        if field[x+i,y] == 1: return True
    for i in range(4):
        if field[x-i,y] == -1: break
        if field[x-i,y] == 1: return True
    for i in range(4):
        if field[x,y+i] == -1: break
        if field[x,y+i] == 1: return True
    for i in range(4):
        if field[x,y-i] == -1: break
        if field[x,y-i] == 1: return True
    return False

def compute_danger_tiles(game_state):

    field = game_state["field"]
    danger_tiles = set()

    explosion_map = game_state["explosion_map"]
    xs, ys = np.nonzero(explosion_map > 0)
    danger_tiles.update(zip(xs.tolist(), ys.tolist()))

    for (bx, by), _timer in game_state["bombs"]:
        danger_tiles.update(get_blast_coords(field, bx, by, 3))

    return danger_tiles

def gonna_die(game_state, x,y):
    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    for (bx, by), timer in game_state["bombs"]:
        blast_cords = get_blast_coords(field,bx,by,3)
        #print((x,y) in blast_cords)
        #print(timer)
        if (x,y) in blast_cords and timer == 0:
            return 1
    if explosion_map[x][y] == 0:
        return 0
    else:
        return 1

def get_blast_coords(field, x, y, power):

    blast_coords = [(x, y)]

    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        for i in range(1, power + 1):
            nx, ny = x + dx * i, y + dy * i

            if field[nx, ny] == -1:
                break

            blast_coords.append((nx, ny))

    return blast_coords


def bomb_endangers_nearest_opponent(game_state, x, y):
    """
    Returns True if placing a bomb at (x, y) would put the nearest opponent
    (by Manhattan distance) within its blast radius, False if there are no
    opponents or the nearest one would be safe.
    """
    others = [pos for (_, _, _, pos) in game_state["others"]]
    if not others:
        return False

    field = game_state["field"]
    nearest = min(others, key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y))
    blast_coords = get_blast_coords(field, x, y, 3)
    return nearest in blast_coords


DIRECTIONS = [
    ((0, -1), 0),
    ((1, 0), 1),
    ((0, 1), 2),
    ((-1, 0), 3)
]


def compute_directions(game_state, x, y, danger_tiles=None):
    """
    Single breadth-first search from (x, y) that simultaneously determines the
    first-step direction (0-3, or 4 if unreachable) towards:
      - the nearest coin
      - the nearest tile adjacent to a crate (5 if (x, y) is already adjacent)
      - the nearest safe tile (outside any bomb blast radius / explosion)
      - the nearest opponent agent

    Returns a tuple (coin_direction, crate_direction, safety_direction, opponent_direction).
    """
    coins = set(game_state["coins"])
    if danger_tiles is None:
        danger_tiles = compute_danger_tiles(game_state)
    bombs = {bomb[0] for bomb in game_state['bombs']}
    opponents = {other[3] for other in game_state["others"]}

    start = (x, y)

    # Determine the opponent direction first (independently of coin/crate/safety)
    # so that the safety search below can steer away from it.
    opponent_direction = None if opponents else 4
    if opponent_direction is None:
        opponent_direction = _bfs_first_direction(game_state, start, lambda pos: pos in opponents)
        if opponent_direction is None:
            opponent_direction = 4

    coin_direction = None
    crate_direction = 5 if is_next_to_crate(x, y, game_state) else None
    safety_direction = None
    safety_direction_fallback = None

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in DIRECTIONS:
        nx = start[0] + dx
        ny = start[1] + dy

        if is_free(game_state, nx, ny):
            queue.append(((nx, ny), direction_id))
            visited.add((nx, ny))

    while queue and (
            coin_direction is None
            or crate_direction is None
            or safety_direction is None
    ):
        position, first_direction = queue.popleft()
        px, py = position

        if coin_direction is None and position in coins:
            coin_direction = first_direction
        if crate_direction is None and is_next_to_crate(px, py, game_state):
            crate_direction = first_direction
        if position not in bombs and position not in danger_tiles:
            if safety_direction_fallback is None:
                safety_direction_fallback = first_direction
            if safety_direction is None and first_direction != opponent_direction:
                safety_direction = first_direction

        for (dx, dy), _ in DIRECTIONS:
            nx = px + dx
            ny = py + dy
            next_position = (nx, ny)

            if (
                    next_position not in visited
                    and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction))

    # Prefer a safe direction that does not coincide with the opponent's
    # direction; only fall back to it if no other safe route was found.
    if safety_direction is None:
        safety_direction = safety_direction_fallback

    return (
        coin_direction if coin_direction is not None else 4,
        crate_direction if crate_direction is not None else 4,
        safety_direction if safety_direction is not None else 4,
        opponent_direction,
    )


def _bfs_first_direction(game_state, start, is_goal):
    """
    Minimal BFS from `start` that returns the first-step direction (0-3)
    towards the nearest tile satisfying `is_goal`, or None if unreachable.
    """
    if is_goal(start):
        return None

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in DIRECTIONS:
        nx, ny = start[0] + dx, start[1] + dy
        if is_free(game_state, nx, ny):
            visited.add((nx, ny))
            queue.append(((nx, ny), direction_id))

    while queue:
        position, first_direction = queue.popleft()
        if is_goal(position):
            return first_direction

        px, py = position
        for (dx, dy), _ in DIRECTIONS:
            nx, ny = px + dx, py + dy
            next_position = (nx, ny)
            if next_position not in visited and is_free(game_state, nx, ny):
                visited.add(next_position)
                queue.append((next_position, first_direction))

    return None

def action_value_aprox_Function(action, game_state:dict, self):
    features = action_state_to_features(game_state,action)
    return features.T @ self.model

def policy(game_state:dict,self):
    random_prob =  0
    #print(game_state['explosion_map'])
    if self.train and random.random() < random_prob:
        # 1/6 for any action
        prob = 1/6
        action = np.random.choice(ACTIONS, p=[0,0,0,0,0,1])
        self.logger.debug(f"Choosing action purely at random. :. {action}")
        return action
    actionvalue = -np.inf
    for a in ACTIONS:
        valtemp = action_value_aprox_Function(a,game_state,self)
        if valtemp > actionvalue:
            action = a
            actionvalue = valtemp
    self.logger.debug(f'Querying model for action :. {action}')
    return action