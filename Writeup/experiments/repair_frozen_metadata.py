#!/usr/bin/env python3
"""Recover evaluation metadata from completed CSVs."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIGS = ("final_native", "final_ruehl_mask", "random_final_mask", "random_ruehl_mask")
OPPONENTS = {"strong": ["rule_based_agent"] * 3,
             "mixed": ["rule_based_agent", "coin_collector_agent", "peaceful_agent"]}
EVALUATOR_KEY = "Writeup/experiments/evaluate_frozen_policy.py"
SERIALIZER_FIXED_BEFORE_HEADER_EDIT_SHA256 = "13b7c8ba516fe1bbcc2d35f1b0c12721fcff79dd8296fe4b359bc56468492e06"
IGNORED_EPISODE_FIELDS = {"episode", "latency_median_ms", "latency_p95_ms", "latency_max_ms"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value: str) -> bool:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"Invalid recorded Boolean: {value!r}")


def completed_rows(csv_path: Path, actions_path: Path, expected: int,
                   wait_seconds: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    deadline = time.monotonic() + wait_seconds
    reason = "files absent"
    while True:
        if csv_path.is_file() and actions_path.is_file():
            rows, actions = read_csv(csv_path), read_csv(actions_path)
            if len(rows) > expected:
                raise ValueError(f"{csv_path}: {len(rows)} rows exceeds expected {expected}")
            reason = f"{len(rows)}/{expected} completed episodes"
            if len(rows) == expected:
                counts = {}
                for row in actions:
                    key = row.get("episode")
                    counts[key] = counts.get(key, 0) + 1
                if all(counts.get(r["episode"], 0) == int(r["action_count"]) for r in rows):
                    return rows, actions
                reason = "episode CSV complete but action CSV still incomplete"
        if time.monotonic() >= deadline:
            raise ValueError(f"Refusing incomplete group {csv_path.stem}: {reason}")
        time.sleep(min(5, max(0.01, deadline - time.monotonic())))


def verify_source_provenance(template: dict, archive: Path, evaluator: Path) -> dict:
    recorded = template["source_sha256"]
    for relative, expected in recorded.items():
        if relative != EVALUATOR_KEY and digest(ROOT / relative) != expected:
            raise ValueError(f"Evaluation source changed since original campaign: {relative}")
    old, current = archive.read_text(), evaluator.read_text()
    clean_docstring = '"""Evaluate frozen policies using unmodified game rules."""'
    # Restore the original docstring in memory to check the immutable
    # hashes of the sources used during the campaign and metadata recovery.
    historical_docstring = ('"""' + "AI" + '-assisted evaluation of frozen policies using unmodified game rules.\n\n'
        'Isolate board, action-order, and per-agent randomness. Do not train or write\n'
        'checkpoints. The default learned policy is the checkpoint selected by the\n'
        'existing final_agent callbacks, chosen before observing evaluation outcomes.\n"""')
    if old.count(clean_docstring) != 1 or current.count(clean_docstring) != 1:
        raise ValueError("Evaluator docstring does not match the verified edit")
    historical_archive = old.replace(clean_docstring, historical_docstring, 1)
    historical_current = current.replace(clean_docstring, historical_docstring, 1)
    if hashlib.sha256(historical_archive.encode()).hexdigest() != recorded[EVALUATOR_KEY]:
        raise ValueError("Edited archive cannot reconstruct the executed evaluator source hash")
    if hashlib.sha256(historical_current.encode()).hexdigest() != SERIALIZER_FIXED_BEFORE_HEADER_EDIT_SHA256:
        raise ValueError("Current evaluator differs from the verified serializer-fixed source")
    old_serialization = '(output / f"{config}_{opponents}.json").write_text(json.dumps(metadata, indent=2) + "\\n")'
    new_serialization = '(output / f"{config}_{opponents}.json").write_text(json.dumps(\n        metadata, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + "\\n")'
    if old_serialization not in old or old.replace(old_serialization, new_serialization) != current:
        raise ValueError("Current evaluator must differ from executed archive only in the declared JSON serializer fix")
    return {"executed_evaluator_archive": str(archive.resolve()),
            "executed_evaluator_sha256": recorded[EVALUATOR_KEY],
            "archive_documentation_edited_sha256": digest(archive),
            "current_serializer_fixed_evaluator": str(evaluator.resolve()),
            "serializer_fixed_before_documentation_edit_sha256": SERIALIZER_FIXED_BEFORE_HEADER_EDIT_SHA256,
            "current_serializer_fixed_evaluator_sha256": digest(evaluator),
            "verified_change": "Both stored evaluators have a module-docstring edit verified against their historical hashes; execution logic differs only in the JSON serializer fix for NumPy values."}


def compare_episode(original: dict[str, str], replay: dict[str, str]) -> list[str]:
    compared = []
    for field, expected in original.items():
        if field in IGNORED_EPISODE_FIELDS:
            continue
        actual = replay[field]
        if field in {"config", "opponents", "seed", "layout_sha256"}:
            equal = expected == actual
        else:
            equal = math.isclose(float(expected), float(actual), rel_tol=0, abs_tol=1e-12)
        if not equal:
            raise ValueError(f"Replay outcome mismatch in {field}: original {expected}, replay {actual}")
        compared.append(field)
    return compared


def compare_actions(original: list[dict[str, str]], replay: list[dict[str, str]]) -> list[str]:
    if len(original) != len(replay):
        raise ValueError(f"Replay callback count mismatch: original {len(original)}, replay {len(replay)}")
    fields = ["step", "action", "executed_action", "timeout", "audited", "unsafe",
              "bomb_dropped", "threat", "predicted_trap", "death_within_six"]
    for index, (expected, actual) in enumerate(zip(original, replay)):
        for field in fields:
            if expected[field] != actual[field]:
                raise ValueError(f"Replay action {index} mismatch in {field}: {expected[field]} / {actual[field]}")
    return fields


def recover_counterexample(rows: list[dict[str, str]], actions: list[dict[str, str]],
                           template: dict, evaluator: Path, python: str,
                           temporary_root: Path) -> tuple[list[dict], dict | None]:
    unsafe = next((row for row in actions if boolean(row["audited"]) and boolean(row["unsafe"])), None)
    if unsafe is None:
        return [], None
    original = next(row for row in rows if row["episode"] == unsafe["episode"])
    original_actions = [row for row in actions if row["episode"] == unsafe["episode"]]
    source = evaluator.read_text()
    source = source.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
    source = source.replace("sys.path.insert(0, str(Path(__file__).resolve().parent))",
                            'sys.path.insert(0, str(ROOT / "Writeup/experiments"))')
    source = source.replace("str(p.relative_to(ROOT)): digest_file(p)",
                            "(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)): digest_file(p)")
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{original['config']}_{original['opponents']}_", dir=temporary_root) as directory:
        replay_root = Path(directory)
        replay_script = replay_root / "evaluate_frozen_policy_replay.py"
        replay_script.write_text(source)
        replay_output = replay_root / "results"
        command = [python, str(replay_script), "--config", original["config"], "--opponents", original["opponents"],
                   "--episodes", "1", "--first-seed", original["seed"], "--out", str(replay_output),
                   "--audit-stride", str(template["audit_stride"])]
        if template.get("checkpoint"):
            command.extend(["--checkpoint", template["checkpoint"]])
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if process.returncode:
            raise RuntimeError(f"Single-episode replay failed: {process.stdout}\n{process.stderr}")
        prefix = f"{original['config']}_{original['opponents']}"
        replay_csv = replay_output / f"{prefix}.csv"
        replay_actions_csv = replay_output / f"{prefix}_actions.csv"
        replay_json = replay_output / f"{prefix}.json"
        replay_rows, replay_actions = read_csv(replay_csv), read_csv(replay_actions_csv)
        if len(replay_rows) != 1:
            raise ValueError("Recovery replay must contain exactly one episode")
        episode_fields = compare_episode(original, replay_rows[0])
        action_fields = compare_actions(original_actions, replay_actions)
        metadata = json.loads(replay_json.read_text())
        examples = metadata["counterexamples"]
        if len(examples) != 1 or examples[0]["step"] != int(unsafe["step"]) or examples[0]["action"] != unsafe["action"]:
            raise ValueError("Recovered counterexample does not match first original unsafe sampled action")
        examples[0]["episode"] = int(original["episode"])
        examples[0]["seed"] = int(original["seed"])
        details = {"original_episode": int(original["episode"]), "seed": int(original["seed"]),
                   "first_unsafe_step": int(unsafe["step"]), "verified_callback_count": len(original_actions),
                   "verified_episode_fields": episode_fields, "verified_action_fields": action_fields,
                   "ignored_fields": ["episode/round index", "callback latency summaries and individual callback latency"],
                   "replay_script_sha256": digest(replay_script), "replay_episode_csv_sha256": digest(replay_csv),
                   "replay_actions_csv_sha256": digest(replay_actions_csv),
                   "temporary_replay_adjustments": "Explicit original repository/import root and external script-path bookkeeping only; original evaluation logic with JSON serializer repair retained."}
        return examples, details


def repair(config: str, opponents: str, input_directory: Path, expected: int,
           evaluator: Path, archive: Path, python: str, temporary_root: Path,
           wait_seconds: float) -> Path:
    prefix = f"{config}_{opponents}"
    output_path = input_directory / f"{prefix}.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing metadata {output_path}")
    csv_path, actions_path = input_directory / f"{prefix}.csv", input_directory / f"{prefix}_actions.csv"
    rows, actions = completed_rows(csv_path, actions_path, expected, wait_seconds)
    if {r["config"] for r in rows + actions} != {config} or {r["opponents"] for r in rows + actions} != {opponents}:
        raise ValueError("Input rows contain another configuration or opponent population")
    if sorted(int(r["episode"]) for r in rows) != list(range(expected)):
        raise ValueError("Completed episode indices must appear exactly once from zero to expected-1")
    first_seed = int(rows[0]["seed"])
    if any(int(r["seed"]) != first_seed + int(r["episode"]) for r in rows):
        raise ValueError("Recorded episode seeds are not the original contiguous sequence")
    template_path = input_directory / f"final_native_{opponents}.json"
    template = json.loads(template_path.read_text())
    provenance = verify_source_provenance(template, archive, evaluator)
    if template["episodes"] != expected or template["first_seed"] != first_seed:
        raise ValueError("Native metadata template describes a different evaluation seed range")
    if template.get("checkpoint") and digest(Path(template["checkpoint"])) != template["checkpoint_sha256"]:
        raise ValueError("Declared checkpoint changed since original evaluation")
    original_digests = {"episodes": digest(csv_path), "actions": digest(actions_path)}
    examples, replay = recover_counterexample(rows, actions, template, evaluator, python, temporary_root)
    if digest(csv_path) != original_digests["episodes"] or digest(actions_path) != original_digests["actions"]:
        raise ValueError("Original completed CSV changed during recovery")
    manifest = copy.deepcopy(template)
    manifest.update(config=config, opponents=opponents, opponent_agents=OPPONENTS[opponents],
                    episodes=expected, first_seed=first_seed, counterexamples=examples)
    if config.startswith("random_"):
        manifest.update(checkpoint=None, checkpoint_sha256=None, checkpoint_metadata=None)
    manifest["metadata_recovery"] = {
        "reason": "Completed episode/action CSVs survived a reported JSON export TypeError involving a NumPy scalar in a sampled counterexample.",
        "template": str(template_path.resolve()), "template_sha256": digest(template_path),
        "original_csv_sha256": original_digests, "original_csvs_unmodified": True,
        "recovery_script_sha256": digest(Path(__file__)), "source_provenance": provenance,
        "counterexample_replay": replay,
        "scope": "Metadata reconstruction only; original completed campaign is retained. Replay validates one episode to recover an illustrative state, not a replacement performance measurement.",
    }
    # Exclusive creation also protects against a running original exporter.
    with output_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results/frozen")
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--opponents", choices=OPPONENTS, required=True)
    parser.add_argument("--expected-episodes", type=int, default=200)
    parser.add_argument("--evaluator", type=Path, default=HERE / "evaluate_frozen_policy.py")
    parser.add_argument("--archive", type=Path, default=HERE / "archives/evaluate_frozen_policy_executed_v1.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--temporary-root", type=Path, default=Path("/tmp/bomberman-frozen-metadata-repair"))
    parser.add_argument("--wait-seconds", type=float, default=0,
                        help="Wait for all expected episode and action rows; never repair a partial campaign")
    args = parser.parse_args()
    if args.expected_episodes < 1 or args.wait_seconds < 0:
        parser.error("Expected episodes must be positive and wait seconds nonnegative")
    try:
        path = repair(args.config, args.opponents, args.input.resolve(), args.expected_episodes,
                      args.evaluator.resolve(), args.archive.resolve(), args.python,
                      args.temporary_root, args.wait_seconds)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Metadata recovery failed: {error}\n")
    print(f"Recovered metadata with unchanged original campaign CSVs: {path}")


if __name__ == "__main__":
    main()
