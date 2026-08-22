import os
import pickle
from pyexpat import features
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']
MODEL_FILE = os.path.join(os.path.dirname(__file__), "my-saved-model.pt")
DIRECTIONS = [(0,-1), (1,0), (0,1), (-1,0), (0,0)]

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
  
    #setup the q learning model
    self.epsilon = 0.3  # Exploration rate
    self.alpha = 0.1    # Learning rate
    self.gamma = 0.9    # Discount factor

    

    
    if self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Setting up model from scratch.")
        
        self.model = {}
        
    else:
        self.logger.info("Loading model from saved state.")
        with open(MODEL_FILE, "rb") as file:
            self.model = pickle.load(file)





def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
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
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = tuple(state_to_features(game_state))
    self.logger.debug(f"State features: {state}")
    self.logger.info(f"State features: {state}")
    # todo Exploration vs exploitation
    
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS))
    
    
    if self.train and random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 20% wait.
        return random.choices(ACTIONS, weights=[0.2, 0.2, 0.2, 0.2, 0.2])[0]

    
    self.logger.debug("Querying model for action.")
    return ACTIONS[np.argmax(self.model[state])]  # Choose the action with the highest Q-value



