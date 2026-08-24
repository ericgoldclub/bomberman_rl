from collections import namedtuple, deque

import numpy as np
from .callbacks import ACTIONS, state_to_features, MODEL_FILE

import pickle
from typing import List

import events as e


# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

ACTION_TO_DELTA = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "WAIT": (0, 0),
}


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    old_state = tuple(state_to_features(old_game_state))
    new_state = tuple(state_to_features(new_game_state))
    action_index = ACTIONS.index(self_action)
    

    #did we move closer to or away from a coin? 
    if old_game_state["coins"]:
        coin_dx, coin_dy = old_state[0], old_state[1]
        action_dx, action_dy = ACTION_TO_DELTA[self_action]

        if (action_dx, action_dy) == (coin_dx, coin_dy):
            events.append(e.MOVED_CLOSE_TO_COIN)
        else:
            events.append(e.MOVED_AWAY_FROM_COIN)

    reward = reward_from_events(self, events)

    #did we move closer to enemy?
    if old_game_state["others"]:
        enemy_dx, enemy_dy = old_state[2], old_state[3]
        action_dx, action_dy = ACTION_TO_DELTA[self_action]

        if (action_dx, action_dy) == (enemy_dx, enemy_dy):
            events.append(e.MOVED_CLOSE_TO_ENEMY)
        else:
            events.append(e.MOVED_AWAY_FROM_ENEMY)

    if old_state not in self.model:
        self.model[old_state] = np.zeros(len(ACTIONS))

    if new_state not in self.model:
        self.model[new_state] = np.zeros(len(ACTIONS))

    old_q_value = self.model[old_state][action_index]
    future_q_value = np.max(self.model[new_state])

    new_q_value = old_q_value + self.alpha * (
        reward + self.gamma * future_q_value - old_q_value
    )

    self.model[old_state][action_index] = new_q_value

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(old_state, self_action, new_state, reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    last_state = tuple(state_to_features(last_game_state))
    reward = reward_from_events(self, events)
    

    self.transitions.append(Transition(last_state, last_action, None, reward))

    if last_state not in self.model:
        self.model[last_state] = np.zeros(len(ACTIONS))

    old_q_value = self.model[last_state][ACTIONS.index(last_action)]
    new_q_value = old_q_value + self.alpha * (reward - old_q_value)

    self.model[last_state][ACTIONS.index(last_action)] = new_q_value
    
    # Store the model
    with open(MODEL_FILE, "wb") as file:
        pickle.dump(self.model, file)
    


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 8,
        e.MOVED_CLOSE_TO_COIN: 3,
        e.MOVED_AWAY_FROM_COIN: -3,
        e.MOVED_CLOSE_TO_ENEMY: -2,
        e.MOVED_AWAY_FROM_ENEMY: 2,
        e.WAITED: -1,
        e.INVALID_ACTION : -3,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
