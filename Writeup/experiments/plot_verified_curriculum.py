#!/usr/bin/env python3
"""Plot historical checkpoint summaries from recorded Git revisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "Writeup" / "figures"
RESULTS = ROOT / "Writeup" / "experiments" / "results"
SNAPSHOT_PATH = "agent_code/RUEHL_BASED_AGENT/Hyperparams.prm"
SNAPSHOTS = (
    ("Peaceful killer", "e664bca"),
    ("Classic self-play", "b66ec29"),
)


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def read_snapshot(label: str, revision: str) -> dict:
    commit = git("rev-parse", revision).decode().strip()
    source = git("show", f"{commit}:{SNAPSHOT_PATH}")
    values = {
        key: float(value)
        for line in source.decode().splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    required = (
        "BEST_MODEL_MEAN_GAME_SCORE",
        "BEST_MODEL_COINS_COLLECTED",
        "BEST_MODEL_MEAN_ENEMIES_KILLED",
        "BEST_MODEL_SUICIDE_RATE",
        "BEST_MODEL_SCORE",
        "EVAL_ROUNDS",
    )
    missing = set(required) - set(values)
    if missing:
        raise ValueError(f"{revision}: missing values {sorted(missing)}")

    official = values["BEST_MODEL_MEAN_GAME_SCORE"]
    coins = values["BEST_MODEL_COINS_COLLECTED"]
    kills = values["BEST_MODEL_MEAN_ENEMIES_KILLED"]
    suicide = values["BEST_MODEL_SUICIDE_RATE"]
    selection = values["BEST_MODEL_SCORE"]
    if not np.isclose(official, coins + 5 * kills, atol=1e-8):
        raise ValueError(f"{revision}: official-score decomposition inconsistent")
    if not np.isclose(selection, official + 5 * kills - 5 * suicide, atol=1e-8):
        raise ValueError(f"{revision}: killer-profile selection score inconsistent")
    if values["EVAL_ROUNDS"] != 20:
        raise ValueError(f"{revision}: expected a configured 20-round evaluation batch")

    return {
        "stage_label": label,
        "stage_label_origin": "existing report, Writeup/main.tex training-stage table",
        "git_commit": commit,
        "source_path": SNAPSHOT_PATH,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "configured_evaluation_rounds": int(values["EVAL_ROUNDS"]),
        "official_mean_score": official,
        "mean_coins": coins,
        "mean_credited_kills": kills,
        "suicide_fraction": suicide,
        "checkpoint_selection_score": selection,
        "recorded_environment_steps": int(values["STEPS_DONE"]),
        "snapshot_values": values,
    }


def main() -> None:
    records = [read_snapshot(label, revision) for label, revision in SNAPSHOTS]
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Verified historical selected-checkpoint summaries, not fresh evaluation",
        "source_git_revision": git("rev-parse", "HEAD").decode().strip(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_score_definition": "coins + 5 * credited kills",
        "killer_selection_definition": "official mean score + 5 * mean kills - 5 * suicide fraction",
        "uncertainty": "unavailable: per-round selection outcomes absent",
        "interpretation": "Distinct stages and training histories; no controlled superiority claim",
        "records": records,
    }
    (RESULTS / "verified_curriculum.json").write_text(json.dumps(manifest, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    x = np.arange(len(records), dtype=float)
    width = 0.31
    official = ax.bar(x - width / 2, [r["official_mean_score"] for r in records],
                      width, color="#315B86", label="Official mean game score")
    selection = ax.bar(x + width / 2, [r["checkpoint_selection_score"] for r in records],
                       width, color="#C98B40", label="Checkpoint-selection objective J")
    for bars in (official, selection):
        ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=11)
    ax.set_xticks(x, [f"{r['stage_label']}\n{r['git_commit'][:7]}" for r in records])
    ax.set_ylabel("Recorded summary value")
    ax.set_ylim(0, 13.4)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D5DBE0", alpha=0.7, linewidth=0.6)
    ax.set_title("Verified historical checkpoint summaries", loc="left", pad=35,
                 fontsize=14, weight="bold")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.005), ncol=2,
              frameon=False, fontsize=9)
    fig.text(0.12, 0.09, "Configured selection batches: 20 greedy rounds. Selected validation summaries;",
             fontsize=9, color="#46535F")
    fig.text(0.12, 0.05, "official game score and the checkpoint-selection objective are shown separately.",
             fontsize=9, color="#46535F")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.78)
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"verified_curriculum_summaries.{extension}", dpi=220)
    plt.close(fig)
    print(json.dumps({"figure_stem": str(FIGURES / "verified_curriculum_summaries"),
                      "records": [{k: r[k] for k in ("stage_label", "git_commit", "official_mean_score", "checkpoint_selection_score")}
                                  for r in records]}, indent=2))


if __name__ == "__main__":
    main()
