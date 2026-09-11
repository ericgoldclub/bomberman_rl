import os
import pickle
import random
from collections import deque

import numpy as np


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
        self.model = np.zeros(9)
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
    coin_direction = direction_to_nearest_coin(game_state, x, y)
    coin = 0
    crate = 0
    crate_direction = direction_to_crate(game_state,x,y)
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
        safety_direction = direction_to_safety(game_state,x,y)
        if ACTIONS[safety_direction] == action:
            safety = 1
        else:
            safety = 0
    xold,yold = x,y
    x,y = move_cords(action,game_state)
    free = is_free(game_state,x,y)
    danger = (x,y) in danger_tiles
    planted_next_to_crate = 0
    if is_next_to_crate(x,y,game_state) == 1 and action =="BOMB":
        planted_next_to_crate = 1
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


def direction_to_safety(game_state,x,y):
    start = (x,y)
    danger_tiles = compute_danger_tiles(game_state)
    bombs_all = game_state['bombs']
    bombs = [bomb[0] for bomb in bombs_all]
    directions = [
        ((0, -1), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((-1, 0), 3)
    ]
    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in directions:
        nx = start[0] + dx
        ny = start[1] + dy

        if is_free(game_state, nx, ny):
            queue.append(((nx, ny), direction_id))
            visited.add((nx, ny))

    while queue:
        position, first_direction = queue.popleft()
        x, y = position
        if (x,y) not in bombs and (x,y) not in danger_tiles:
            return first_direction
        for (dx, dy), _ in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if (
                    next_position not in visited
                    and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction))
    return 4


def direction_to_nearest_coin(game_state,x,y):
    #Bestimmt mittels Breitensuche den ersten Schritt auf dem kürzesten Weg zum nächsten Coin

    start = (x,y)
    coins = set(game_state["coins"])

    if not coins:
        return 4

    directions = [
        ((0, -1), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((-1, 0), 3)
    ]

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in directions:
        nx = start[0] + dx
        ny = start[1] + dy

        if is_free(game_state, nx, ny):
            queue.append(((nx, ny), direction_id))
            visited.add((nx, ny))

    while queue:
        position, first_direction = queue.popleft()
        if position in coins:
            return first_direction
        x, y = position
        for (dx, dy), _ in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if (
                    next_position not in visited
                    and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction))
    return 4

def direction_to_crate(game_state, x, y):
    start = (x,y)
    if is_next_to_crate(x,y,game_state):
        return 5
    directions = [
        ((0, -1), 0),
        ((1, 0), 1),
        ((0, 1), 2),
        ((-1, 0), 3)
    ]

    queue = deque()
    visited = {start}

    for (dx, dy), direction_id in directions:
        nx = start[0] + dx
        ny = start[1] + dy

        if is_free(game_state, nx, ny):
            queue.append(((nx, ny), direction_id))
            visited.add((nx, ny))

    while queue:
        position, first_direction = queue.popleft()
        x, y = position
        if is_next_to_crate(x,y,game_state):
            return first_direction
        for (dx, dy), _ in directions:
            nx = x + dx
            ny = y + dy
            next_position = (nx, ny)

            if (
                    next_position not in visited
                    and is_free(game_state, nx, ny)
            ):
                visited.add(next_position)
                queue.append((next_position, first_direction))
    return 4

def action_value_aprox_Function(action, game_state:dict, self):
    features = action_state_to_features(game_state,action)
    return features.T @ self.model

def policy(game_state:dict,self):
    random_prob = .1
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