import pickle

import events as e
from .callbacks import (
    Q_TABLE_FILE,
    state_to_features,
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

LOOP_ENTRY_PENALTY = 1.0
LOOP_CONTINUE_PENALTY = 2.0
STAGNATION_PENALTY = 2.0

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
    new_loop_return_direction = 4

    if new_game_state is not None:
        new_position = tuple(
            new_game_state["self"][3]
        )

        (
            new_movement_pattern,
            new_loop_return_direction,
        ) = detect_movement_pattern(
            self.position_history,
            new_position
        )

    new_state = state_to_features(
        new_game_state,
        self.current_target_position,
        movement_pattern=new_movement_pattern,
        loop_return_direction=new_loop_return_direction,
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

    # In Bombengefahr soll der Agent die berechnete Fluchtrichtung lernen.
    # Bei akuter Gefahr ist das Signal staerker als bei einer spaeteren Bombe.
    if (
            old_state is not None
            and old_state.in_danger
            and old_state.target_direction in (0, 1, 2, 3)
    ):
        action_direction = ACTION_TO_DIRECTION.get(self_action)

        if old_state.danger_urgency == 2:
            if action_direction == old_state.target_direction:
                reward += 2.0
            elif action_direction is not None:
                reward -= 1.0

        elif old_state.danger_urgency == 1:
            if action_direction == old_state.target_direction:
                reward += 0.8
            elif action_direction is not None:
                reward -= 0.2


    if old_distance is not None and new_distance_for_reward is not None:
        if new_distance_for_reward < old_distance:
            reward += 0.4
        elif new_distance_for_reward > old_distance:
            reward -= 0.4


    if old_state is not None and new_state is not None:
        # Der entscheidende Sicherheitsfortschritt: Gefahr verlassen.
        if old_state.in_danger and not new_state.in_danger:
            reward += 4.0

        # Nicht freiwillig in eine neue Bombengefahr laufen.
        elif not old_state.in_danger and new_state.in_danger:
            reward -= 4.0

        # Auch innerhalb einer Gefahr ist der Timer relevant:
        # von akut -> weniger akut ist gut, umgekehrt schlecht.
        elif old_state.in_danger and new_state.in_danger:
            if new_state.danger_urgency < old_state.danger_urgency:
                reward += 1.0
            elif new_state.danger_urgency > old_state.danger_urgency:
                reward -= 2.0

        danger_multiplier = 2.0 if (
                old_state.in_danger
                or new_state.in_danger
        ) else 1.0

        # A -> B -> A:
        # Der Agent ist gerade wieder auf das vorletzte Feld gelaufen.
        if new_state.movement_pattern == 1:
            reward -= (
                    LOOP_ENTRY_PENALTY
                    * danger_multiplier
            )

        # Der Agent befindet sich bereits in A -> B -> A
        # und laeuft wieder direkt zurueck nach B.
        if (
                old_state.movement_pattern == 1
                and old_state.loop_return_direction in (0, 1, 2, 3)
                and ACTION_TO_DIRECTION.get(self_action)
                == old_state.loop_return_direction
        ):
            reward -= (
                    LOOP_CONTINUE_PENALTY
                    * danger_multiplier
            )

        # Wiederholtes Stehenbleiben.
        if (
                old_state.movement_pattern == 2
                and self_action == "WAIT"
        ):
            reward -= (
                    STAGNATION_PENALTY
                    * danger_multiplier
            )

    old_q = self.q_table.get((old_state, self_action), 0.0)

    valid_next_actions = valid_actions_from_state(
        new_state
    )

    # Nur Aktionen beruecksichtigen,
    # deren Q-Wert tatsaechlich gelernt wurde.
    known_next_q_values = [
        self.q_table[(new_state, action)]
        for action in valid_next_actions
        if (new_state, action) in self.q_table
    ]

    # Ist der komplette Folgezustand noch unbekannt,
    # ist 0.0 ein sinnvoller initialer Schaetzwert.
    max_next_q = (
        max(known_next_q_values)
        if known_next_q_values
        else 0.0
    )

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
        reward += 15

    if e.CRATE_DESTROYED in events:
        reward += 3

    if e.COIN_FOUND in events:
        reward += 2

    if e.INVALID_ACTION in events:
        reward -= 1

    if e.WAITED in events:
        reward -= 1

    if e.KILLED_SELF in events:
        reward -= 30

    if e.GOT_KILLED in events:
        reward -= 30

    if e.KILLED_OPPONENT in events:
        reward += 20

    if old_state is not None and self_action == "WAIT":
        if old_state.danger_urgency == 2:
            reward -= 4.0
        elif old_state.danger_urgency == 1:
            reward -= 1.5

    if self_action == "BOMB" and e.BOMB_DROPPED in events and old_state is not None:

        if old_state.in_danger:
            reward -= 5

        elif not old_state.safe_bomb:
            reward -= 8

        elif old_state.opponent_in_blast_range:
            reward += 6

        elif old_state.crate_in_blast_range:
            reward += 3

        else:
            # Bombe ohne sichtbaren taktischen Zweck nicht komplett verbieten,
            # aber klar unattraktiver machen.
            reward -= 2

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
    self.current_target_position = None
    self.current_target_kind = None
    self.opponent_target_lock_until = None
