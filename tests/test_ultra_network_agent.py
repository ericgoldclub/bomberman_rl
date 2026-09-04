import logging
import unittest
from collections import deque
from unittest.mock import patch

import numpy as np
import torch

import events as e
from agent_code.Ultra_Network_agent import train as ultra_train
from agent_code.Ultra_Network_agent.callbacks import (
    ACTIONS,
    board_to_channels,
    build_hazard_timeline,
    can_hit_enemy_with_bomb,
    explosion_tiles_from_bomb,
    state_to_features,
    valid_action_mask,
)
from agent_code.Ultra_Network_agent.train import (
    HybridDQN,
    Transition,
    end_of_round,
    reward_from_events,
)


def make_game_state(*, position=(1, 3), bombs=(), crates=(), enemies=()):
    field = np.zeros((17, 17), dtype=np.int8)
    field[0, :] = -1
    field[-1, :] = -1
    field[:, 0] = -1
    field[:, -1] = -1
    for crate in crates:
        field[crate] = 1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("test", 0, True, position),
        "others": [
            (f"enemy-{index}", 0, True, enemy)
            for index, enemy in enumerate(enemies)
        ],
        "bombs": list(bombs),
        "coins": [(7, 7)],
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


class UltraNetworkAgentTest(unittest.TestCase):
    def test_blast_and_enemy_detection_continue_behind_crates(self):
        game_state = make_game_state(
            position=(1, 3),
            crates=((3, 3),),
            enemies=((4, 3),),
        )
        field = game_state["field"]

        blast = explosion_tiles_from_bomb(field, (1, 3))

        self.assertIn((3, 3), blast)
        self.assertIn((4, 3), blast)
        self.assertTrue(can_hit_enemy_with_bomb(field, 1, 3, [(4, 3)]))

    def test_indestructible_wall_stops_blast(self):
        game_state = make_game_state(position=(1, 3))
        field = game_state["field"]
        field[3, 3] = -1

        blast = explosion_tiles_from_bomb(field, (1, 3))

        self.assertNotIn((3, 3), blast)
        self.assertNotIn((4, 3), blast)

    def test_timeline_and_action_mask_use_future_blast_time(self):
        game_state = make_game_state(
            position=(1, 3),
            bombs=(((3, 3), 0),),
            crates=((2, 3),),
        )

        hazards = build_hazard_timeline(
            game_state["field"],
            game_state["bombs"],
            game_state["explosion_map"],
        )
        mask = valid_action_mask(game_state)

        self.assertIn((1, 3), hazards[1])
        self.assertIn((1, 3), hazards[2])
        self.assertFalse(mask[ACTIONS.index("WAIT")])
        self.assertTrue(mask[ACTIONS.index("UP")])

    def test_direction_features_are_deterministic_and_match_safety_mask(self):
        game_state = make_game_state(
            position=(1, 3),
            bombs=(((3, 3), 1),),
            crates=((2, 3),),
        )

        first_board, first_vector = state_to_features(game_state)
        movement_mask = valid_action_mask(game_state)[:4]

        for _ in range(10):
            board, vector = state_to_features(game_state)
            np.testing.assert_array_equal(board, first_board)
            np.testing.assert_array_equal(vector, first_vector)

        board = board_to_channels(game_state, movement_mask)
        encoded_safe_moves = int(board[11].sum())
        self.assertEqual(encoded_safe_moves, int(movement_mask.sum()))

    def test_network_shape_and_non_stacked_suicide_reward(self):
        network = HybridDQN(board_channels=14, vector_dim=22, output_dim=6)
        output = network(torch.zeros(2, 14, 17, 17), torch.zeros(2, 22))
        self.assertEqual(tuple(output.shape), (2, 6))

        agent = type("Agent", (), {"logger": logging.getLogger(__name__)})()
        reward = reward_from_events(agent, [e.KILLED_SELF, e.GOT_KILLED])
        self.assertEqual(reward, -20.0)

    def test_final_survivor_transition_is_replaced_not_duplicated(self):
        game_state = make_game_state()
        features = state_to_features(game_state)
        previous_reward = -0.03
        transition = Transition(
            features,
            "WAIT",
            features,
            previous_reward,
            np.ones(len(ACTIONS), dtype=bool),
        )
        policy_net = type("Policy", (), {"state_dict": lambda self: {}})()
        agent = type(
            "Agent",
            (),
            {
                "logger": logging.getLogger(__name__),
                "replay_buffer": deque([transition]),
                "last_features": features,
                "last_transition_step": game_state["step"],
                "last_transition_action": "WAIT",
                "position_history": deque([(1, 3)], maxlen=4),
                "policy_net": policy_net,
                "steps_done": 1,
                "last_bomb_positions": None,
                "steps_since_last_bomb": None,
            },
        )()

        with (
            patch.object(ultra_train, "optimize_model"),
            patch.object(ultra_train.torch, "save"),
        ):
            end_of_round(agent, game_state, "WAIT", [e.SURVIVED_ROUND])

        self.assertEqual(len(agent.replay_buffer), 1)
        final_transition = agent.replay_buffer[0]
        self.assertIsNone(final_transition.next_state)
        self.assertIsNone(final_transition.next_action_mask)
        self.assertAlmostEqual(final_transition.reward, previous_reward + 2.0)


if __name__ == "__main__":
    unittest.main()
