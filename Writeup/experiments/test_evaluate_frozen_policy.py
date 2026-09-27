"""Test evaluator invariants with temporary fixtures."""
from __future__ import annotations

import csv
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_frozen_policy as evaluator
from items import Explosion
from mask_audit import engine_timeline, make_world, oracle_path, state_from_field, wall_board


def assert_numpy_state_equal(test, left, right):
    test.assertEqual(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    test.assertEqual(left[2:], right[2:])


class RandomnessTests(unittest.TestCase):
    def test_callback_preserves_external_streams_and_replays_its_own(self):
        callback = SimpleNamespace(act=lambda self, state: (random.random(), float(np.random.random())))
        backend = evaluator.EvaluationBackend(callback, "rng-test")
        backend.seed(123)
        saved_python, saved_numpy = random.getstate(), np.random.get_state()
        backend.send_event("act", {})
        first = backend.get("act")
        self.assertEqual(random.getstate(), saved_python)
        assert_numpy_state_equal(self, np.random.get_state(), saved_numpy)
        backend.send_event("act", {})
        second = backend.get("act")
        self.assertNotEqual(first, second)
        backend.seed(123)
        backend.send_event("act", {})
        self.assertEqual(first, backend.get("act"))
        backend.send_event("act", {})
        self.assertEqual(second, backend.get("act"))

    def test_other_agents_cannot_advance_a_callback_stream(self):
        callback = SimpleNamespace(act=lambda self, state: (random.random(), float(np.random.random())))
        left = evaluator.EvaluationBackend(callback, "left")
        right = evaluator.EvaluationBackend(callback, "right")
        left.seed(42)
        right.seed(42)
        interloper = evaluator.EvaluationBackend(callback, "other")
        interloper.seed(999)
        for _ in range(5):
            left.send_event("act", {})
            for _ in range(7):
                interloper.send_event("act", {})
            right.send_event("act", {})
            self.assertEqual(left.get("act"), right.get("act"))

    def test_external_streams_restored_when_callback_raises(self):
        def failing(self, state):
            random.random()
            np.random.random()
            raise ValueError("intentional callback failure")

        backend = evaluator.EvaluationBackend(SimpleNamespace(act=failing), "failure")
        saved_python, saved_numpy = random.getstate(), np.random.get_state()
        with self.assertRaisesRegex(ValueError, "intentional"):
            backend.send_event("act", {})
        self.assertEqual(random.getstate(), saved_python)
        assert_numpy_state_equal(self, np.random.get_state(), saved_numpy)


class HazardTests(unittest.TestCase):
    def test_existing_explosion_export_and_reconstruction_match_engine(self):
        for timer in (1, evaluator.s.EXPLOSION_TIMER):
            with self.subTest(timer=timer):
                state = state_from_field(wall_board(), (1, 1))
                world = make_world(state)
                owner = SimpleNamespace(name="test-explosion", bombs_left=False)
                world.explosions = [Explosion([(2, 1)], [], owner, timer)]
                actor = SimpleNamespace(dead=False, get_state=lambda: state["self"])
                world.active_agents = [actor]
                world.round, world.step, world.user_input = 1, 1, "WAIT"
                exported = evaluator.BombeRLeWorld.get_state_for_agent(world, actor)
                self.assertEqual(exported["explosion_map"][2, 1], timer - 1)
                actual = engine_timeline(exported, world=world)
                reference = evaluator.observed_paths(exported)
                for action in evaluator.ACTIONS[:5]:
                    self.assertEqual(reference[action] is not None,
                                     oracle_path(exported, action, actual) is not None)
                self.assertEqual(reference["RIGHT"] is None, timer > 1)

    def test_current_opponent_blocks_first_movement(self):
        state = state_from_field(wall_board(), (1, 1))
        state["others"] = [("opponent", 0, True, (2, 1))]
        paths = evaluator.observed_paths(state)
        self.assertIsNone(paths["RIGHT"])
        self.assertIsNotNone(paths["WAIT"])

    def test_wait_on_own_bomb_and_unavailable_bomb(self):
        state = state_from_field(wall_board(), (1, 1), [((1, 1), 3)], False)
        paths = evaluator.observed_paths(state)
        self.assertIsNotNone(paths["WAIT"])
        self.assertIsNone(paths["BOMB"])


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.saved_mask = evaluator.final.valid_action_mask
        self.saved_model_path = evaluator.final.MODEL_FILE
        self.saved_callbacks = getattr(evaluator.EvaluationWorld, "focal_callbacks", None)
        self.saved_observer = getattr(evaluator.EvaluationWorld, "observe", None)

    def tearDown(self):
        evaluator.final.valid_action_mask = self.saved_mask
        evaluator.final.MODEL_FILE = self.saved_model_path
        evaluator.EvaluationWorld.focal_callbacks = self.saved_callbacks
        evaluator.EvaluationWorld.observe = self.saved_observer

    def test_native_timeout_discards_action_and_skips_next_callback(self):
        calls = []
        focal = SimpleNamespace(setup=lambda self: None,
                                act=lambda self, state: calls.append(state["step"]) or "BOMB")
        opponent = SimpleNamespace(setup=lambda self: None, act=lambda self, state: "WAIT")
        evaluator.EvaluationWorld.focal_callbacks = focal
        evaluator.EvaluationWorld.observe = staticmethod(lambda *args: None)
        with tempfile.TemporaryDirectory() as temp:
            args = evaluator.WorldArgs(True, 30, False, 0, False, None, False, True,
                                       temp, False, "test", 42, False, "classic")
            with patch.object(evaluator.importlib, "import_module", return_value=opponent):
                world = evaluator.EvaluationWorld(args, [("random_agent", False)] * 4)
            world.new_round()
            actor = world.agents[0]
            original_get = actor.backend.get_with_time
            with patch.object(actor.backend, "get_with_time",
                              side_effect=lambda name: (original_get(name)[0], 1.1)):
                world.do_step()
            self.assertEqual(world.replay["actions"]["focal"][-1], "WAIT")
            self.assertEqual(actor.statistics["bombs"], 0)
            self.assertLess(actor.available_think_time, 0)
            world.do_step()
            self.assertEqual(calls, [1])
            self.assertEqual(world.replay["actions"]["focal"], ["WAIT", "WAIT"])
            self.assertGreater(actor.available_think_time, 0)
            world.end()

    def test_paired_layouts_reproducible_gameplay_and_unchanged_checkpoint(self):
        checkpoint = evaluator.ROOT / "agent_code/final_agent/Ultra_Network_agent_saved_model_v2.pt"
        checksum = evaluator.digest_file(checkpoint)
        weights, metadata = evaluator.verify_checkpoint(checkpoint)
        self.assertEqual(metadata["architecture_version"], 2)
        self.assertEqual(tuple(weights["cnn.0.weight"].shape), (32, 14, 3, 3))
        rows = {}
        with tempfile.TemporaryDirectory() as temp:
            for config in evaluator.CONFIGS:
                output = Path(temp) / config
                output.mkdir()
                evaluator.evaluate(config, "strong", 1, 270920260, checkpoint, output, 0)
                with (output / f"{config}_strong.csv").open() as stream:
                    rows[config] = next(csv.DictReader(stream))
                self.assertEqual(int(rows[config]["timeout_events"]), 0)
                self.assertEqual(int(rows[config]["score"]),
                                 int(rows[config]["coins"]) + 5 * int(rows[config]["kills"]))
                self.assertTrue(0 <= float(rows[config]["win_share"]) <= 1)
                run_metadata = json.loads((output / f"{config}_strong.json").read_text())
                if config.startswith("final_"):
                    self.assertEqual(run_metadata["checkpoint_sha256"], checksum)
            self.assertEqual(len({row["layout_sha256"] for row in rows.values()}), 1)
            repeated = Path(temp) / "repeat"
            repeated.mkdir()
            evaluator.evaluate("final_native", "strong", 1, 270920260, checkpoint, repeated, 0)
            with (repeated / "final_native_strong.csv").open() as stream:
                repeat = next(csv.DictReader(stream))
            for key in rows["final_native"]:
                if not key.startswith("latency_"):
                    self.assertEqual(rows["final_native"][key], repeat[key], key)
        self.assertEqual(evaluator.digest_file(checkpoint), checksum)

    def test_native_inference_failure_is_not_silently_randomized(self):
        checkpoint = evaluator.ROOT / "agent_code/final_agent/Ultra_Network_agent_saved_model_v2.pt"
        evaluator.final.MODEL_FILE = str(checkpoint)
        backend = evaluator.EvaluationBackend(
            SimpleNamespace(setup=evaluator.final.setup, act=evaluator.NATIVE_FINAL_ACT), "learned")
        backend.send_event("setup")
        self.assertFalse(backend.fake_self.train)
        self.assertFalse(backend.fake_self.policy_net.training)
        self.assertFalse(hasattr(backend.fake_self, "model"))
        state = state_from_field(wall_board(), (1, 1))
        with patch.object(backend.fake_self.policy_net, "forward",
                          side_effect=RuntimeError("intentional inference failure")):
            with self.assertRaisesRegex(RuntimeError, "intentional inference failure"):
                backend.send_event("act", state)


if __name__ == "__main__":
    unittest.main()
