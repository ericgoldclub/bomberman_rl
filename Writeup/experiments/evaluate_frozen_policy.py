#!/usr/bin/env python3
"""Evaluate frozen policies using unmodified game rules."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import events as e
import settings as s
from agents import Agent
from environment import BombeRLeWorld, WorldArgs
from items import Explosion
from agent_code.final_agent import callbacks as final
from agent_code.final_agent.train import HybridDQN
from agent_code.RUEHL_BASED_AGENT import callbacks as ruehl
from mask_audit import make_world, engine_timeline, oracle_path

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ACTIONS = final.ACTIONS
NATIVE_FINAL_MASK = final.valid_action_mask
NATIVE_FINAL_ACT = final.act
OPPONENTS = {"strong": ["rule_based_agent"] * 3,
             "mixed": ["rule_based_agent", "coin_collector_agent", "peaceful_agent"]}
CONFIGS = ("final_native", "final_ruehl_mask", "random_final_mask", "random_ruehl_mask")


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StrictCallbackLogger:
    """Make native fallback warnings visible rather than silently randomizing."""
    def debug(self, *args, **kwargs):
        pass

    info = debug

    def warning(self, message, *args, **kwargs):
        raise RuntimeError(message % args if args else message)

    error = warning
    exception = warning


class EvaluationBackend:
    """Sequential callback backend with isolated RNGs and no file logging."""
    def __init__(self, callbacks, name, observer=None):
        self.callbacks = callbacks
        self.fake_self = SimpleNamespace(train=False, logger=StrictCallbackLogger())
        self.runner = SimpleNamespace(fake_self=self.fake_self, callbacks=callbacks)
        self.observer = observer
        self.actor = None
        self.result = None
        self.seed(0)

    def seed(self, seed):
        self.python_state = random.Random(int(seed)).getstate()
        self.numpy_state = np.random.RandomState(int(seed)).get_state()

    def send_event(self, name, *args):
        saved_python, saved_numpy = random.getstate(), np.random.get_state()
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        try:
            start = perf_counter()
            value = getattr(self.callbacks, name)(self.fake_self, *args)
            elapsed = perf_counter() - start
            self.python_state, self.numpy_state = random.getstate(), np.random.get_state()
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
        self.result = (name, value, elapsed)
        # Instrumentation is excluded from callback latency and think time.
        if name == "act" and self.observer is not None:
            self.observer(args[0], value, elapsed, self.actor.available_think_time)

    def get(self, name):
        return self.get_with_time(name)[0]

    def get_with_time(self, name):
        event, value, elapsed = self.result
        assert event == name
        return value, elapsed


class EvaluationWorld(BombeRLeWorld):
    def setup_logging(self):
        self.logger = logging.getLogger("frozen-evaluation-world")
        self.logger.disabled = True
        self.game_log_path = None

    def setup_agents(self, entries):
        self.agents = []
        for index, (code, _) in enumerate(entries):
            callback = self.focal_callbacks if index == 0 else importlib.import_module(f"agent_code.{code}.callbacks")
            backend = EvaluationBackend(callback, str(index), self.observe if index == 0 else None)
            color = self.colors.pop()
            agent = Agent("focal" if index == 0 else f"opponent_{index}", code, str(index),
                          False, backend, color, color)
            backend.actor = agent
            self.agents.append(agent)


def observed_paths(state):
    """Reference for observed states; future opponents and new bombs omitted."""
    def timeline(place_bomb=False):
        world = make_world(state, place_bomb)
        owner = SimpleNamespace(name="observed-explosion", bombs_left=False)
        for x, y in np.argwhere(state["explosion_map"] > 0):
            remaining = int(state["explosion_map"][x, y])
            world.explosions.append(Explosion([(int(x), int(y))], [], owner, remaining + 1))
        times = engine_timeline(state, world=world)
        # Match the declared conditional reference: current opponents block
        # the first action; later movement is unknown and is not forecast.
        field, occupied, lethal = times[0]
        times[0] = (field, occupied | {tuple(other[-1]) for other in state["others"]}, lethal)
        return times
    ordinary = timeline()
    return {a: oracle_path(state, a, timeline(True) if a == "BOMB" else ordinary) for a in ACTIONS}


def serialize_state(state):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in state.items() if key != "user_input"}


def verify_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    weights = data.get("model_state_dict", data)
    model = HybridDQN(14, 22, 6)
    model.load_state_dict(weights, strict=True)
    return weights, {k: data.get(k) for k in ("architecture_version", "steps_done", "logged_rounds", "logged_env_steps")}


def evaluate(config, opponents, episodes, first_seed, checkpoint, output, audit_stride):
    learned = config.startswith("final_")
    mask = NATIVE_FINAL_MASK if config.endswith("native") or config == "random_final_mask" else ruehl.action_mask
    final.valid_action_mask = mask
    final.MODEL_FILE = str(checkpoint)
    weights, checkpoint_metadata = verify_checkpoint(checkpoint) if learned else (None, None)
    if learned:
        callback = SimpleNamespace(setup=final.setup, act=NATIVE_FINAL_ACT)
    else:
        def random_act(self, state):
            allowed = mask(state)
            return random.choice([a for a, valid in zip(ACTIONS, allowed) if valid])
        callback = SimpleNamespace(setup=lambda self: None, act=random_act)

    latency, records, selected_examples = [], [], []
    metrics = {}
    episode_number = 0
    def observe(state, action, elapsed, available):
        latency.append(elapsed * 1000)
        record = {"step": state["step"], "action": action, "latency_ms": elapsed * 1000,
                  "timeout": elapsed > available, "audited": False, "unsafe": False,
                  "threat": False, "predicted_trap": False}
        if action == "BOMB":
            blast = ruehl.blast_tiles(state["field"], state["self"][-1])
            record["threat"] = any(tuple(other[-1]) in blast for other in state["others"])
            record["predicted_trap"] = any(tuple(other[-1]) in blast and not final.enemy_has_escape_after_bomb(
                state["field"], state["bombs"], state["explosion_map"], other[-1], state["self"][-1])
                for other in state["others"])
        if audit_stride and (len(records) % audit_stride == 0) and not record["timeout"]:
            paths = observed_paths(state)
            if any(paths[a] is not None for a in ACTIONS[:5]):
                record["audited"] = True
                record["unsafe"] = paths[action] is None
                if record["unsafe"] and not selected_examples:
                    selected_examples.append({"config": config, "opponents": opponents, "episode": episode_number,
                                              "step": state["step"], "action": action, "state": serialize_state(state),
                                              "surviving_paths": paths})
        records.append(record)

    args = WorldArgs(no_gui=True, fps=30, turn_based=False, update_interval=0,
                     save_replay=False, replay=None, make_video=False, continue_without_training=True,
                     log_dir=str(output), save_stats=False, match_name="frozen-evaluation",
                     seed=first_seed, silence_errors=False, scenario="classic")
    EvaluationWorld.focal_callbacks = callback
    EvaluationWorld.observe = staticmethod(observe)
    world = EvaluationWorld(args, [("final_agent" if learned else "random_agent", False)] +
                            [(name, False) for name in OPPONENTS[opponents]])
    if learned:
        player = world.agents[0].backend.fake_self
        if hasattr(player, "model"):
            raise RuntimeError("Native agent fell back instead of loading its network")
        for key, tensor in player.policy_net.state_dict().items():
            if not torch.equal(tensor.cpu(), weights[key]):
                raise RuntimeError(f"Loaded network differs from declared checkpoint at {key}")
    csv_path = output / f"{config}_{opponents}.csv"
    actions_path = output / f"{config}_{opponents}_actions.csv"
    if csv_path.exists() or actions_path.exists():
        raise FileExistsError(f"Refusing to overwrite {csv_path}")
    rows = []
    for episode_number in range(episodes):
        seed = first_seed + episode_number
        world.rng = np.random.default_rng(seed)
        for index, actor in enumerate(world.agents):
            actor.backend.seed(int(np.random.SeedSequence([seed, index, 1701]).generate_state(1)[0]))
        world.new_round()
        # Define independent episodes by clearing episodic input memory.
        focal = world.agents[0]
        fake_self = focal.backend.fake_self
        if hasattr(fake_self, "position_history"):
            fake_self.position_history.clear()
            fake_self.last_features = None
        layout = {"arena": world.arena.tolist(), "coins": [(int(c.x), int(c.y), bool(c.collectable)) for c in world.coins],
                  "spawns": [(int(a.x), int(a.y)) for a in world.agents]}
        layout_hash = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        world.rng = np.random.default_rng(np.random.SeedSequence([seed, 7201]))
        latency.clear()
        records.clear()
        timeout_skips = 0
        while world.running:
            count = len(records)
            was_alive = not focal.dead
            world.do_step()
            if was_alive and len(records) == count:
                timeout_skips += 1
            if len(records) > count:
                records[-1]["executed_action"] = world.replay["actions"][focal.name][-1]
                records[-1]["bomb_dropped"] = e.BOMB_DROPPED in focal.events
                if focal.dead:
                    for record in records:
                        record["death_within_six"] = 0 <= world.step - record["step"] < 6
        maximum = max(actor.score for actor in world.agents)
        winners = sum(actor.score == maximum for actor in world.agents)
        audited = [r for r in records if r["audited"]]
        row = {"config": config, "opponents": opponents, "episode": episode_number, "seed": seed,
               "layout_sha256": layout_hash, "score": focal.score, "coins": focal.statistics["coins"],
               "kills": focal.statistics["kills"], "suicides": focal.statistics["suicides"],
               "survived": int(not focal.dead), "win_share": 1 / winners if focal.score == maximum else 0,
               "episode_steps": world.step, "bombs": focal.statistics["bombs"], "invalid": focal.statistics["invalid"],
               "timeout_events": sum(r["timeout"] for r in records), "timeout_skips": timeout_skips,
               "action_count": len(records), "unsafe_selected": sum(r["unsafe"] for r in audited),
               "checked_actions": len(audited), "audited_deaths": sum(r.get("death_within_six", False) for r in audited),
               "unsafe_selected_deaths": sum(r["unsafe"] and r.get("death_within_six", False) for r in audited),
               "threat_bombs": sum(r["threat"] and r.get("bomb_dropped", False) for r in records),
               "trapped_threat_bombs": sum(r["predicted_trap"] and r.get("bomb_dropped", False) for r in records),
               "latency_median_ms": float(np.median(latency)) if latency else 0,
               "latency_p95_ms": float(np.percentile(latency, 95)) if latency else 0,
               "latency_max_ms": max(latency, default=0)}
        assert row["score"] == row["coins"] + 5 * row["kills"]
        rows.append(row)
        # Preserve streaming results if a long evaluation is interrupted.
        with csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if episode_number == 0:
                writer.writeheader()
            writer.writerow(row)
        with actions_path.open("a", newline="") as stream:
            fields = ["config", "opponents", "episode", "seed", "step", "action", "executed_action",
                      "latency_ms", "timeout", "audited", "unsafe", "bomb_dropped", "threat", "predicted_trap", "death_within_six"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            if episode_number == 0:
                writer.writeheader()
            for record in records:
                writer.writerow({key: record.get(key, False) for key in fields} |
                                {"config": config, "opponents": opponents, "episode": episode_number, "seed": seed})
        if (episode_number + 1) % 25 == 0:
            print(f"{config}/{opponents}: {episode_number + 1}/{episodes}, mean score={np.mean([r['score'] for r in rows]):.3f}", flush=True)
    metadata = {"config": config, "opponents": opponents, "opponent_agents": OPPONENTS[opponents], "episodes": episodes,
                "first_seed": first_seed, "audit_stride": audit_stride, "checkpoint": str(checkpoint) if learned else None,
                "checkpoint_sha256": digest_file(checkpoint) if learned else None, "checkpoint_metadata": checkpoint_metadata,
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "platform": platform.platform(), "cpu_threads": torch.get_num_threads(),
                "protocol": "Fixed per-episode layout seeds; separate action-order generator; isolated Python/NumPy streams per agent; episodic input history cleared; unmodified engine physics and timeout enforcement; observer excluded from callback timing; no training.",
                "audit_scope": f"Every {audit_stride} selected actions (zero disables); six-step conditional survival under existing bombs and explosions; current opponents block first movement only, future opponents/new bombs omitted.",
                "source_sha256": {str(p.relative_to(ROOT)): digest_file(p) for p in
                    [ROOT/"environment.py", ROOT/"items.py", ROOT/"settings.py", ROOT/"agents.py", ROOT/"agent_code/final_agent/callbacks.py",
                     ROOT/"agent_code/final_agent/train.py", ROOT/"agent_code/RUEHL_BASED_AGENT/callbacks.py", Path(__file__).resolve()]},
                "counterexamples": selected_examples}
    (output / f"{config}_{opponents}.json").write_text(json.dumps(
        metadata, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + "\n")
    print(f"Finished {config}/{opponents}: {len(rows)} episodes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--opponents", choices=OPPONENTS, default="strong")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--first-seed", type=int, default=270920260)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "agent_code/final_agent/Ultra_Network_agent_saved_model_v2.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "Writeup/experiments/results/frozen")
    parser.add_argument("--audit-stride", type=int, default=10)
    args = parser.parse_args()
    if args.episodes < 1 or args.audit_stride < 0:
        parser.error("episodes must be positive and audit stride nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    logging.disable(logging.CRITICAL)
    evaluate(args.config, args.opponents, args.episodes, args.first_seed,
             args.checkpoint.resolve(), args.out.resolve(), args.audit_stride)


if __name__ == "__main__":
    main()
