import logging
import random
import unittest

import numpy as np
import torch

import events as e
from agent_code.dqn_agent_v3.Networks import DQN_prev
from agent_code.dqn_agent_v3.callbacks import (
    ACTIONS,
    _bomb_traps_enemy,
    _policy_action_mask,
    state_to_features,
)
from agent_code.dqn_agent_v3.train import (
    KILLER_REPLAY_SIGNAL_FRACTION,
    KILLER_REPLAY_SIGNAL_THRESHOLD,
    REWARD_PROFILES,
    ReplayBuffer,
    Transition,
    _append_killer_events,
    _model_selection_score,
    reward_from_events,
)


def make_game_state(
    *,
    position=(1, 3),
    bombs=(),
    crates=(),
    enemies=(),
    field=None,
):
    if field is None:
        field = np.zeros((17, 17), dtype=np.int8)
        field[0, :] = -1
        field[-1, :] = -1
        field[:, 0] = -1
        field[:, -1] = -1
    else:
        field = field.copy()

    for crate in crates:
        field[crate] = 1

    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("killer", 0, True, position),
        "others": [
            (f"enemy-{index}", 0, True, enemy)
            for index, enemy in enumerate(enemies)
        ],
        "bombs": list(bombs),
        "coins": [],
        "explosion_map": np.zeros_like(field, dtype=np.float32),
    }


def killer_reward_agent():
    profile = REWARD_PROFILES["killer"]
    return type(
        "Agent",
        (),
        {
            "logger": logging.getLogger(__name__),
            "log_dqn_details": False,
            "reward_profile_name": "killer",
            "major_rewards": profile["major_rewards"],
            "shaping_rewards": profile["shaping_rewards"],
            "suicide_reward": profile["suicide_reward"],
            "step_time_cost": profile["step_time_cost"],
            "shaping_clip": profile["shaping_clip"],
        },
    )()


class KillerConfigurationTest(unittest.TestCase):
    def test_compact_dqn_prev_sees_the_full_board(self):
        network = DQN_prev(10, (17, 17), 24, len(ACTIONS))

        self.assertEqual(network.receptive_field_size(), 17)
        self.assertLess(
            sum(parameter.numel() for parameter in network.parameters()),
            500_000,
        )
        output = network(
            torch.zeros(2, 10, 17, 17),
            torch.zeros(2, 24),
        )
        self.assertEqual(tuple(output.shape), (2, len(ACTIONS)))

        # Confirm the center feature of the deepest convolution really depends
        # on both edges, rather than trusting only the arithmetic above.
        for module in network.cnn:
            if isinstance(module, torch.nn.Conv2d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
        board = torch.ones(1, 10, 17, 17, requires_grad=True)
        center_feature = network.cnn(board)[0, 0, 4, 4]
        center_feature.backward()
        support = board.grad[0, 0] != 0
        self.assertTrue(bool(support[0, 0]))
        self.assertTrue(bool(support[-1, -1]))

    def test_killer_selection_prefers_a_real_kill_over_a_coin_sweep(self):
        coin_sweep = _model_selection_score("killer", 9.0, 0.0, 0.0)
        one_kill_in_thirty = _model_selection_score(
            "killer",
            5.0 / 30.0,
            1.0 / 30.0,
            1.0 / 30.0,
        )

        self.assertGreater(one_kill_in_thirty, coin_sweep)
        self.assertEqual(
            _model_selection_score("standard", 4.25, 3.0, 1.0),
            4.25,
        )

    def test_killer_rewards_do_not_stack_generic_death_on_suicide(self):
        agent = killer_reward_agent()

        suicide = reward_from_events(agent, [e.KILLED_SELF, e.GOT_KILLED])
        kill = reward_from_events(agent, [e.KILLED_OPPONENT])

        self.assertAlmostEqual(suicide, -6.002)
        self.assertAlmostEqual(kill, 9.998)

    def test_signal_balanced_replay_includes_rare_lethal_outcomes(self):
        replay = ReplayBuffer(32)
        for reward in [0.0] * 17 + [10.0, -6.0, -4.0]:
            replay.append(
                Transition(None, None, 0, None, None, None, reward)
            )

        random.seed(7)
        batch = replay.sample(
            8,
            priority_predicate=lambda transition: abs(transition.reward)
            >= KILLER_REPLAY_SIGNAL_THRESHOLD,
            priority_fraction=KILLER_REPLAY_SIGNAL_FRACTION,
        )

        self.assertGreaterEqual(
            sum(abs(transition.reward) >= 3.0 for transition in batch),
            2,
        )


class KillerSafetyAndShapingTest(unittest.TestCase):
    def test_danger_channel_matches_blasts_that_continue_through_crates(self):
        game_state = make_game_state(
            bombs=(((1, 3), 1),),
            crates=((2, 3),),
        )

        grid, _ = state_to_features(game_state)

        self.assertGreater(grid[9, 3, 3], 0.0)

    def test_action_mask_rejects_useless_bombs_and_immediate_blasts(self):
        calm_state = make_game_state()
        self.assertFalse(
            _policy_action_mask(calm_state)[ACTIONS.index("BOMB")]
        )

        dangerous_state = make_game_state(
            bombs=(((3, 3), 0),),
            crates=((2, 3),),
        )
        mask = _policy_action_mask(dangerous_state)
        self.assertFalse(mask[ACTIONS.index("WAIT")])
        self.assertTrue(mask[ACTIONS.index("UP")])

    def test_trap_bomb_receives_targeted_shaping_event(self):
        field = np.full((17, 17), -1, dtype=np.int8)
        for tile in ((1, 3), (2, 3), (3, 3), (1, 2), (1, 1), (2, 1)):
            field[tile] = 0

        old_state = make_game_state(
            position=(1, 3),
            enemies=((3, 3),),
            field=field,
        )
        new_state = make_game_state(
            position=(1, 3),
            bombs=(((1, 3), 3),),
            enemies=((3, 3),),
            field=field,
        )
        events = [e.BOMB_DROPPED]

        self.assertTrue(_bomb_traps_enemy(new_state))
        _append_killer_events(
            killer_reward_agent(),
            old_state,
            "BOMB",
            new_state,
            events,
        )

        self.assertIn(e.KILL_BOMB_DROPPED, events)

    def test_timed_mask_forces_progress_along_an_escape_route(self):
        field = np.full((17, 17), -1, dtype=np.int8)
        for tile in ((1, 3), (1, 2), (1, 1), (2, 1)):
            field[tile] = 0

        game_state = make_game_state(
            position=(1, 3),
            bombs=(((1, 3), 2),),
            field=field,
        )
        game_state["self"] = ("killer", 0, False, (1, 3))

        mask = _policy_action_mask(game_state)

        self.assertTrue(mask[ACTIONS.index("UP")])
        self.assertFalse(mask[ACTIONS.index("WAIT")])


if __name__ == "__main__":
    unittest.main()
