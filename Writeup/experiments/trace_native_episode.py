#!/usr/bin/env python3
"""Record and verify the first native-policy episode."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import evaluate_frozen_policy as evaluator


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unexpected JSON value {type(value).__name__}")


def capture(output):
    output.mkdir(parents=True, exist_ok=True)
    observed = evaluator.observed_paths
    snapshots = []

    def recording_observation(state):
        paths = observed(state)
        snapshots.append({"step": state["step"], "state": evaluator.serialize_state(state),
                          "paths": paths, "native_mask": evaluator.NATIVE_FINAL_MASK(state).tolist(),
                          "ruehl_mask": evaluator.ruehl.action_mask(state).tolist()})
        return paths

    evaluator.observed_paths = recording_observation
    try:
        evaluator.evaluate("final_native", "strong", 1, 270920260,
                           evaluator.ROOT / "agent_code/final_agent/Ultra_Network_agent_saved_model_v2.pt",
                           output, 1)
    finally:
        evaluator.observed_paths = observed
    generator = np.random.default_rng(np.random.SeedSequence([270920260, 7201]))
    if [int(row["step"]) for row in snapshots] != list(range(1, len(snapshots) + 1)):
        raise ValueError("Expected one observed snapshot per focal action from step one")
    for row in snapshots:
        ordered = [row["state"]["self"][0]] + [agent[0] for agent in row["state"]["others"]]
        row["reconstructed_action_order"] = [ordered[int(index)]
                                             for index in generator.permutation(len(ordered))]
    record = {
        "scope": "Exploratory first native episode; all-action conditional audit and saved observations; no extra performance sample.",
        "selection_rule": "Primary native strong episode 0, fixed seed 270920260; selected before mechanism inspection",
        "action_order_reconstruction": "Declared action-order RNG default_rng(SeedSequence([seed,7201])), consuming one permutation of current active-agent count per step; current ordering is focal plus state.others ordering.",
        "capture_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "snapshots": snapshots,
    }
    (output / "observed_states.json").write_text(json.dumps(record, indent=2, default=json_scalar) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results/native_trace_capture")
    arguments = parser.parse_args()
    capture(arguments.out.resolve())
