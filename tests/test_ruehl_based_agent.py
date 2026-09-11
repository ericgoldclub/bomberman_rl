import logging
import os
import tempfile
import unittest
from collections import Counter, deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import events as e
import settings as s
from agent_code.RUEHL_BASED_AGENT import callbacks, train
from agent_code.RUEHL_BASED_AGENT.Networks import DQN


def game_state(position=(1, 1), coins=None, others=None, bombs=None):
    field = np.zeros((s.COLS, s.ROWS), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2::2, 2::2] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("me", 0, True, position),
        "coins": coins or [],
        "others": others or [],
        "bombs": bombs or [],
        "explosion_map": np.zeros_like(field),
    }


class RuehlBasedAgentTests(unittest.TestCase):
    def test_features_and_network_cover_the_board(self):
        board, scalar = callbacks.state_to_features(
            game_state(coins=[(15, 15)], others=[("enemy", 0, True, (3, 1))])
        )
        self.assertEqual(board.shape, (7, 17, 17))
        self.assertEqual(scalar.shape, (10,))

        model = DQN(7, (17, 17), 10, 6)
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 500_000)
        self.assertEqual(model(torch.from_numpy(board)[None], torch.from_numpy(scalar)[None]).shape, (1, 6))

        for layer in model.board:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.ones_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        image = torch.ones(1, 7, 17, 17, requires_grad=True)
        model.board[:-1](image)[0, 0, 4, 4].backward()
        self.assertGreater(image.grad[0, 0, 0, 0].item(), 0)
        self.assertGreater(image.grad[0, 0, 16, 16].item(), 0)

    def test_coin_normalization_uses_total_coins(self):
        with patch.object(callbacks, "TOTAL_COINS", 2):
            _, scalar = callbacks.state_to_features(game_state(coins=[(1, 3)]))
        self.assertEqual(scalar[1], 0.5)

    def test_action_mask_only_allows_useful_escapable_bombs(self):
        state = game_state()
        self.assertFalse(callbacks.action_mask(state)[callbacks.ACTIONS.index("BOMB")])

        state["others"] = [("enemy", 0, True, (3, 1))]
        self.assertTrue(callbacks.action_mask(state)[callbacks.ACTIONS.index("BOMB")])

    def test_reward_profiles_do_not_use_directional_target_events(self):
        agent = SimpleNamespace(reward_profile=train.REWARD_PROFILES["killer"])
        killer_reward = train.reward_from_events(
            agent, [e.KILLED_OPPONENT, e.KILL_BOMB_DROPPED]
        )
        suicide_reward = train.reward_from_events(
            agent, [e.KILLED_SELF, e.GOT_KILLED, e.BOMB_EXPLODED]
        )
        self.assertGreater(killer_reward, 10)
        self.assertAlmostEqual(suicide_reward, -6.002)

        directional_events = {
            e.MOVED_CLOSE_TO_COIN,
            e.MOVED_AWAY_FROM_COIN,
            e.MOVED_TOWARDS_CRATE,
            e.MOVED_AWAY_FROM_CRATE,
            e.MOVED_CLOSE_TO_ENEMY,
            e.MOVED_AWAY_FROM_ENEMY,
        }
        for profile in train.REWARD_PROFILES.values():
            self.assertTrue(directional_events.isdisjoint(profile["shaping"]))

        old_state = game_state(position=(1, 1), coins=[(3, 1)])
        new_state = game_state(position=(2, 1), coins=[(3, 1)])
        events = []
        train.add_shaping_events(
            SimpleNamespace(previous_velocity=(0, 0)),
            old_state,
            "RIGHT",
            new_state,
            events,
        )
        self.assertTrue(directional_events.isdisjoint(events))

    def test_killer_evaluation_promotes_game_performance(self):
        agent = SimpleNamespace(
            logger=logging.getLogger("ruehl-test"),
            log_eval_events=False,
            policy_net=DQN(7, (17, 17), 10, 6),
            reward_profile_name="killer",
            best_score=-1e12,
            evaluation_results=[
                {"game_score": 5, "coins": 0, "kills": 1, "completed": 0, "time_left": 0, "suicided": 0, "crates": 1},
                {"game_score": 0, "coins": 0, "kills": 0, "completed": 0, "time_left": 0, "suicided": 1, "crates": 0},
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            model_file = os.path.join(directory, "best.pt")
            with patch.object(train, "BEST_MODEL_FILE", model_file):
                train.complete_evaluation(agent)
            self.assertTrue(os.path.isfile(model_file))
        self.assertEqual(agent.best_mean_enemies_killed, 0.5)
        self.assertEqual(agent.best_suicide_rate, 0.5)

    def test_evaluation_event_log_reports_per_round_means(self):
        logger = logging.getLogger("ruehl-event-log-test")
        result = {
            "game_score": 0,
            "coins": 0,
            "kills": 0,
            "completed": 0,
            "time_left": 0,
            "suicided": 0,
            "crates": 0,
        }
        agent = SimpleNamespace(
            logger=logger,
            log_eval_events=True,
            reward_profile_name="coin",
            best_score=1e12,
            evaluation_results=[
                {**result, "events": {e.WAITED: 50, e.COIN_COLLECTED: 40}},
                {**result, "events": {e.WAITED: 60, e.COIN_COLLECTED: 30}},
            ],
        )

        with self.assertLogs(logger, level="INFO") as captured:
            train.complete_evaluation(agent)

        output = "\n".join(captured.output)
        self.assertIn("WAITED=55.00", output)
        self.assertIn("COIN_COLLECTED=35.00", output)

    def test_evaluation_rounds_are_scheduled_and_greedy(self):
        state = game_state()
        board, scalar = callbacks.state_to_features(state)
        key = train.transition_key(state, "WAIT")
        agent = SimpleNamespace(
            evaluation_round=False,
            evaluation_left=0,
            evaluation_results=[],
            training_rounds=0,
            eval_every=1,
            eval_rounds=2,
            _processed_key=key,
            _pending_key=key,
            memory=deque([
                train.Transition(
                    train.pack(board), train.pack(scalar), train.ACTION_INDEX["WAIT"],
                    board, scalar, np.ones(len(callbacks.ACTIONS)), 0.0,
                )
            ]),
            reward_profile=train.REWARD_PROFILES["standard"],
            end_of_round_opt_steps=0,
            replay_saves_due=0,
            logger=logging.getLogger("ruehl-eval-test"),
            round_game_score=0.0,
            round_coins=0,
            round_kills=0,
            round_crates=0,
            round_completed=False,
            round_time_left=0,
            previous_velocity=(0, 0),
            log_eval_events=True,
            round_event_counts=Counter(),
        )
        with (
            patch.object(train, "save_checkpoint"),
            patch.object(train, "save_hyperparameters"),
            patch.object(train, "save_replay"),
        ):
            train.end_of_round(agent, state, "WAIT", [e.SURVIVED_ROUND])
        self.assertTrue(agent.evaluation_round)
        self.assertEqual(agent.evaluation_left, 2)

        agent._processed_key = key
        agent.round_event_counts.update([e.WAITED])
        with (
            patch.object(train, "save_checkpoint"),
            patch.object(train, "save_hyperparameters"),
            patch.object(train, "complete_evaluation") as finish,
        ):
            train.end_of_round(agent, state, "WAIT", [e.WAITED, e.SURVIVED_ROUND])
            self.assertEqual(
                agent.evaluation_results[0]["events"],
                {e.WAITED: 1, e.SURVIVED_ROUND: 1},
            )
            train.end_of_round(agent, state, "WAIT", [e.SURVIVED_ROUND])
        finish.assert_called_once_with(agent)
        self.assertFalse(agent.evaluation_round)

    def test_every_hyperparameter_progress_value_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "Hyperparams.prm")
            with open(train.HYPERPARAMS_FILE, encoding="utf-8") as source:
                original = source.read()
            with open(filename, "w", encoding="utf-8") as target:
                target.write(original)

            agent = SimpleNamespace()
            with patch.object(train, "HYPERPARAMS_FILE", filename):
                train.load_hyperparameters(agent)
                agent.steps_done = 123
                agent.best_mean_enemies_killed = 0.75
                train.save_hyperparameters(agent)
                with open(filename, encoding="utf-8") as saved_file:
                    saved = saved_file.read()

        for key, _, _ in train.PROGRESS:
            self.assertIn(key + "=", saved)
        self.assertIn("STEPS_DONE=123", saved)
        self.assertIn("BEST_MODEL_MEAN_ENEMIES_KILLED=0.7500000000", saved)

    def test_double_dqn_update_runs(self):
        model = DQN(7, (17, 17), 10, 6)
        agent = SimpleNamespace(
            memory=deque(),
            _pending_key=None,
            min_replay_size=2,
            batch_size=2,
            device=torch.device("cpu"),
            policy_net=model,
            target_net=DQN(7, (17, 17), 10, 6),
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            gamma=0.99,
            gradient_steps=0,
            target_update=100,
        )
        board = train.pack(np.zeros((7, 17, 17), dtype=np.float32))
        scalar = train.pack(np.zeros(10, dtype=np.float32))
        mask = np.ones(6, dtype=np.uint8)
        agent.memory.extend(
            [
                train.Transition(board, scalar, 0, board, scalar, mask, 1.0),
                train.Transition(board, scalar, 1, None, None, None, -1.0),
            ]
        )
        self.assertIsInstance(train.optimize_model(agent), float)
        self.assertEqual(agent.gradient_steps, 1)


if __name__ == "__main__":
    unittest.main()
