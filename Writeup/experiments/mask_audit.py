#!/usr/bin/env python3
"""Compare action masks with six-step engine reachability."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import settings as settings
from agent_code.RUEHL_BASED_AGENT import callbacks as ruehl
from agent_code.final_agent import callbacks as final
from environment import GenericWorld
from items import Bomb, Explosion

ACTIONS = ruehl.ACTIONS
DELTAS = dict(zip(ACTIONS[:5], ruehl.MOVES))
HORIZON = 6


def make_world(state, place_bomb=False):
    """Minimal world; bomb and explosion evolution use repository methods."""
    world = GenericWorld.__new__(GenericWorld)
    world.logger = logging.getLogger("mask-audit")
    world.logger.disabled = True
    world.arena = state["field"].copy()
    world.coins = []
    world.explosions = []
    owner = SimpleNamespace(name="fixed-hazard", bombs_left=False, add_event=lambda _: None)
    bombs = list(state["bombs"])
    if place_bomb:
        bombs.append((state["self"][-1], settings.BOMB_TIMER))
    world.bombs = [Bomb(pos, owner, timer, settings.BOMB_POWER, None) for pos, timer in bombs]
    # The generated suite starts without active explosions. Deterministic
    # regressions below cover both dangerous stages of an existing explosion.
    return world


def engine_timeline(state, place_bomb=False, world=None):
    world = make_world(state, place_bomb) if world is None else world
    timeline = []
    for _ in range(HORIZON):
        before = world.arena.copy()
        occupied = {(bomb.x, bomb.y) for bomb in world.bombs}
        # This is precisely the order in GenericWorld.do_step after actions.
        world.update_explosions()
        world.update_bombs()
        lethal = {tile for exp in world.explosions if exp.is_dangerous() for tile in exp.blast_coords}
        timeline.append((before, occupied, lethal))
    return timeline


def oracle_path(state, action, timeline=None):
    """An existential safe continuation, with dynamic crate destruction.

    No opponents, future new bombs, or coin collection are simulated. An
    intended move must execute legally; WAIT may stay on a bomb below self.
    """
    if action == "BOMB" and not state["self"][2]:
        return None
    if timeline is None:
        timeline = engine_timeline(state, action == "BOMB")
    start = tuple(state["self"][-1])
    frontier = {start: []}
    for step, (field, occupied, lethal) in enumerate(timeline):
        next_frontier = {}
        for position, path in frontier.items():
            options = ["WAIT" if action == "BOMB" else action] if step == 0 else ACTIONS[:5]
            for move in options:
                dx, dy = DELTAS[move]
                nxt = position[0] + dx, position[1] + dy
                x, y = nxt
                if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
                    continue
                if field[nxt] != 0 or (move != "WAIT" and nxt in occupied) or nxt in lethal:
                    continue
                next_frontier.setdefault(nxt, path + [action if step == 0 else move])
        frontier = next_frontier
        if not frontier:
            return None
    return next(iter(frontier.values()))


def replay_witness(state, path):
    """Replay selected oracle witnesses through the actual action method."""
    world = make_world(state)
    events = []
    actor = SimpleNamespace(name="witness", x=state["self"][-1][0], y=state["self"][-1][1],
                            bombs_left=state["self"][2], bomb_sprite=None, add_event=events.append)
    world.active_agents = [actor]
    for action in path:
        events.clear()
        world.perform_agent_action(actor, action)
        assert "INVALID_ACTION" not in events, (action, actor.x, actor.y)
        world.update_explosions()
        world.update_bombs()
        assert all((actor.x, actor.y) not in exp.blast_coords
                   for exp in world.explosions if exp.is_dangerous())


def state_from_field(field, position, bombs=(), available=True):
    return {"round": 1, "step": 1, "field": field, "self": ("audit", 0, available, position),
            "others": [], "coins": [], "bombs": list(bombs), "explosion_map": np.zeros_like(field)}


def wall_board():
    field = np.zeros((settings.COLS, settings.ROWS), dtype=np.int8)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    field[2::2, 2::2] = -1
    return field


def generated_state(rng, density):
    field = wall_board()
    field[(field == 0) & (rng.random(field.shape) < density)] = 1
    # Clear the same corner starting regions as BombeRLeWorld.build_arena.
    for x, y in ((1, 1), (1, 15), (15, 1), (15, 15)):
        for dx, dy in ((0, 0), *ruehl.MOVES[:4]):
            if field[x + dx, y + dy] == 1:
                field[x + dx, y + dy] = 0
    free = [tuple(map(int, p)) for p in np.argwhere(field == 0)]
    position = free[int(rng.integers(len(free)))]
    count = int(rng.integers(1, 4))
    near = [p for p in free if 0 < abs(p[0] - position[0]) + abs(p[1] - position[1]) <= 5]
    pool = [p for p in free if p != position]
    first = (near if near and rng.random() < 0.5 else pool)[0:]
    bombs = [(first[int(rng.integers(len(first)))], int(rng.integers(4)))]
    for _ in range(count - 1):
        candidates = [p for p in pool if p not in [b[0] for b in bombs]]
        bombs.append((candidates[int(rng.integers(len(candidates)))], int(rng.integers(4))))
    own_bomb = rng.random() < 0.25
    if own_bomb:
        bombs.append((position, int(rng.integers(4))))
    return state_from_field(field, position, bombs, available=not own_bomb)


def check_engine_contract():
    """Assert timing, persistence, crate opening, and WAIT-on-bomb semantics."""
    field = wall_board()
    field[3, 1] = 1
    state = state_from_field(field, (1, 3), [((1, 1), 1)])
    timeline = engine_timeline(state)
    assert (1, 1) not in timeline[0][2]
    assert (1, 1) in timeline[1][2] and (1, 1) in timeline[2][2]
    assert (1, 1) not in timeline[3][2]
    assert timeline[1][0][3, 1] == 1 and timeline[2][0][3, 1] == 0
    assert (4, 1) in timeline[1][2], "Crates must not stop the blast"
    state = state_from_field(wall_board(), (1, 1), [((1, 1), 3)], False)
    assert oracle_path(state, "WAIT") is not None
    assert not ruehl.action_mask(state)[ACTIONS.index("WAIT")]
    assert final.valid_action_mask(state)[ACTIONS.index("WAIT")]
    assert oracle_path(state, "BOMB") is None
    state = state_from_field(wall_board(), (1, 1))
    timeline = engine_timeline(state, place_bomb=True)
    assert all((1, 1) not in x[2] for x in timeline[:4])
    assert (1, 1) in timeline[4][2] and (1, 1) in timeline[5][2]
    # Verify exported explosion_map semantics against an existing explosion.
    world = make_world(state)
    owner = SimpleNamespace(name="existing", bombs_left=False)
    world.explosions = [Explosion([(1, 1)], [], owner, settings.EXPLOSION_TIMER)]
    timeline = engine_timeline(state, world=world)
    assert (1, 1) in timeline[0][2] and (1, 1) not in timeline[1][2]
    return "passed: timer=1, lingering blast, crate opening, blast behind crate, WAIT on own bomb, new-bomb timing, existing explosion"


def summarize(rows):
    summaries = []
    for density in sorted({r["density"] for r in rows}) + ["all"]:
        subset = [r for r in rows if density == "all" or r["density"] == density]
        for name in ("ruehl", "final"):
            admits = [r for r in subset if r["mask"] == name and r["action"] != "BOMB"]
            recoverable = [r for r in admits if r["recoverable"]]
            admitted = sum(r["allowed"] for r in recoverable)
            unsafe = sum(r["allowed"] and not r["viable"] for r in recoverable)
            safe = sum(r["viable"] for r in recoverable)
            withheld = sum(r["viable"] and not r["allowed"] for r in recoverable)
            bombs = [r for r in subset if r["mask"] == name and r["action"] == "BOMB" and r["recoverable"] and r["allowed"]]
            state_rows = [r for r in subset if r["mask"] == name and r["action"] == "UP"]
            times = [r["mask_us"] for r in state_rows]
            summaries.append({"density": density, "mask": name, "states": len(state_rows),
                              "recoverable_states": sum(r["recoverable"] for r in state_rows),
                              "admitted_movement_wait": admitted, "unsafe_admitted_movement_wait": unsafe,
                              "unsafe_fraction": unsafe / admitted if admitted else None,
                              "viable_movement_wait": safe, "viable_withheld": withheld,
                              "withheld_fraction": withheld / safe if safe else None,
                              "admitted_bombs": len(bombs), "unsafe_admitted_bombs": sum(not r["viable"] for r in bombs),
                              "median_mask_us": float(np.median(times)), "p95_mask_us": float(np.percentile(times, 95))})
    return summaries


def source_hashes():
    paths = ["environment.py", "items.py", "settings.py", "agent_code/RUEHL_BASED_AGENT/callbacks.py",
             "agent_code/final_agent/callbacks.py", "Writeup/experiments/mask_audit.py"]
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-density", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", type=Path, default=ROOT / "Writeup/experiments/results")
    args = parser.parse_args()
    if args.per_density < 1:
        parser.error("--per-density must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    checks = check_engine_contract()
    rng = np.random.default_rng(args.seed)
    rows = []
    examples = []
    witnesses_replayed = 0
    for density in (0.0, 0.3, 0.75):
        for index in range(args.per_density):
            state = generated_state(rng, density)
            timeline = engine_timeline(state)
            paths = {action: oracle_path(state, action, None if action == "BOMB" else timeline) for action in ACTIONS}
            recoverable = any(paths[a] is not None for a in ACTIONS[:5])
            if index < 30:
                for path in paths.values():
                    if path is not None:
                        replay_witness(state, path)
                        witnesses_replayed += 1
            for name, function in (("ruehl", ruehl.action_mask), ("final", final.valid_action_mask)):
                started = perf_counter_ns()
                mask = function(state)
                elapsed = (perf_counter_ns() - started) / 1000
                for action, allowed in zip(ACTIONS, mask):
                    rows.append({"density": density, "state_id": f"{density}:{index}", "mask": name,
                                 "action": action, "allowed": bool(allowed), "viable": paths[action] is not None,
                                 "recoverable": recoverable, "mask_us": elapsed})
                if recoverable:
                    errors = [a for a, allowed in zip(ACTIONS, mask) if allowed and paths[a] is None]
                    category = "unsafe_bomb" if "BOMB" in errors else "unsafe_movement"
                    saved = sum(x["density"] == density and x["mask"] == name and x["category"] == category for x in examples)
                    if errors and saved < 2:
                        examples.append({"density": density, "state_id": f"{density}:{index}",
                                         "mask": name, "category": category,
                                         "position": state["self"][-1], "bombs": state["bombs"],
                                         "bomb_available": state["self"][2], "field": state["field"].tolist(),
                                         "ruehl_mask": mask.tolist(), "final_mask": final.valid_action_mask(state).tolist(),
                                         "unsafe_allowed_actions": errors, "surviving_paths": paths})
    with (args.out / "mask_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"seed": args.seed, "per_density": args.per_density, "horizon": HORIZON,
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "platform": platform.platform(), "processor": platform.processor(), "checks": checks,
                "witnesses_replayed": witnesses_replayed,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": source_hashes(), "summaries": summarize(rows), "examples": examples,
                "scope": "Generated stress-test states; no opponents, no initial active explosions, no new future bombs; dynamic crate destruction; six-step existential survival; timings cover masks only."}
    (args.out / "mask_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(checks)
    print(json.dumps(metadata["summaries"][-2:], indent=2))


if __name__ == "__main__":
    main()
