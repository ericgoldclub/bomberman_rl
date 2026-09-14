import os
import pickle

import events as e

from .callbacks import (
    ACTIONS,
    Q_TABLE_FILE,
    state_to_features,
    distance_to_target,
    direction_to_safety,
    direction_to_position,
    update_current_target,
    valid_actions_from_state,
    detect_movement_pattern,
)

ACTION_TO_DIRECTION = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3,
}

OSCILLATION_PENALTY = 0.5
STAGNATION_PENALTY = 1.0
LOCAL_LOOP_PENALTY = 1.5
LOOP_ESCAPE_REWARD = 1.5

def setup_training(self):
    self.alpha = 0.1
    self.gamma = 0.9


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: list
):
    old_state = self.last_state
    old_target = self.last_target_position
    old_distance = self.last_distance

    update_current_target(self, new_game_state)

    new_movement_pattern = 0

    if new_game_state is not None:
        new_position = tuple(new_game_state["self"][3])

        new_movement_pattern = detect_movement_pattern(
            self.position_history,
            new_position
        )

    new_state = state_to_features(
        new_game_state,
        self.current_target_position,
        movement_pattern=new_movement_pattern
    )


    if old_state is not None and old_state.in_danger:
        _, new_distance_for_reward = direction_to_safety(new_game_state)
    else:
        if old_target is not None:
            _, new_distance_for_reward = direction_to_position(
                new_game_state,
                old_target
            )
        else:
            new_distance_for_reward = None

    reward = reward_from_events(self_action, old_state, events)


    if old_distance is not None and new_distance_for_reward is not None:
        if new_distance_for_reward < old_distance:
            reward += 0.4
        elif new_distance_for_reward > old_distance:
            reward -= 0.4


    if old_state is not None and new_state is not None:
        if old_state.in_danger and not new_state.in_danger:
            reward += 3.0
        elif not old_state.in_danger and new_state.in_danger:
            reward -= 3.0


    if old_state is not None and new_state is not None:
        # In Gefahr wiegt Loopen doppelt so schwer.
        danger_now = old_state.in_danger or new_state.in_danger
        danger_multiplier = 2.0 if danger_now else 1.0

        if new_state.movement_pattern == 2:
            reward -= STAGNATION_PENALTY * danger_multiplier

        if new_state.movement_pattern == 1:
            reward -= OSCILLATION_PENALTY * danger_multiplier

        elif new_state.movement_pattern == 3:
            reward -= LOCAL_LOOP_PENALTY * danger_multiplier

        if old_state.movement_pattern in (1, 3):
            new_position = tuple(new_game_state["self"][3])

            recent_positions = set(
                list(self.position_history)[-3:]
            )

            if (
                    new_state.movement_pattern == 0
                    and new_position not in recent_positions
            ):
                reward += LOOP_ESCAPE_REWARD * danger_multiplier


    old_q = self.q_table.get((old_state, self_action), 0.0)

    valid_next_actions = valid_actions_from_state(new_state)

    next_q_values = [
        self.q_table.get((new_state, action), 0.0)
        for action in valid_next_actions
    ]

    max_next_q = max(next_q_values)

    self.q_table[(old_state, self_action)] = (
        old_q
        + self.alpha * (
            reward
            + self.gamma * max_next_q
            - old_q
        )
    )


def reward_from_events(self_action: str, old_state, events: list) -> float:
    reward = -0.1

    if e.COIN_COLLECTED in events:
        reward += 20

    if e.CRATE_DESTROYED in events:
        reward += 3

    if e.COIN_FOUND in events:
        reward += 2

    if e.INVALID_ACTION in events:
        reward -= 1

    if e.WAITED in events:
        reward -= 0.1

    if e.KILLED_SELF in events:
        reward -= 50

    if e.GOT_KILLED in events:
        reward -= 50

    if e.KILLED_OPPONENT in events:
        reward += 40

    if old_state is not None and old_state.in_danger and self_action == "WAIT":
        reward -= 1.5

    if self_action == "BOMB" and e.BOMB_DROPPED in events and old_state is not None:

        if old_state.in_danger:
            reward -= 3

        elif not old_state.safe_bomb:
            reward -= 5

        elif old_state.opponent_in_blast_range:
            reward += 5

        elif old_state.crate_adjacent:
            reward += 3

        else:
            reward -= 1

    return reward


def end_of_round(
    self,
    last_game_state: dict,
    last_action: str,
    events: list
):
    last_state = self.last_state

    reward = reward_from_events(last_action, last_state, events)

    old_q = self.q_table.get((last_state, last_action), 0.0)

    self.q_table[(last_state, last_action)] = (
        old_q
        + self.alpha * (
            reward - old_q
        )
    )

    with open(Q_TABLE_FILE, "wb") as file:
        pickle.dump(self.q_table, file)

    self.position_history.clear()
