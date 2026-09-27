#!/usr/bin/env python3
"""Plot the first recorded unsafe action and a viable alternative."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1),
         "LEFT": (-1, 0), "WAIT": (0, 0), "BOMB": (0, 0)}
MOVEMENT_ORDER = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT")


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Expected recorded Boolean; found {value!r}")
    return value == "True"


def load_example(directory):
    prefix = "final_ruehl_mask_strong"
    metadata_path = directory / f"{prefix}.json"
    actions_path = directory / f"{prefix}_actions.csv"
    episodes_path = directory / f"{prefix}.csv"
    metadata = json.loads(metadata_path.read_text())
    if metadata["config"] != "final_ruehl_mask" or metadata["opponents"] != "strong":
        raise ValueError("Unexpected experiment configuration")
    examples = metadata.get("counterexamples", [])
    if not examples:
        raise ValueError("Completed experiment records no unsafe selected action; no failure figure can be made")
    example = examples[0]
    episode, step = int(example["episode"]), int(example["step"])
    with actions_path.open(newline="") as stream:
        measured = [row for row in csv.DictReader(stream) if int(row["episode"]) == episode]
    selected = next(row for row in measured if int(row["step"]) == step)
    if not boolean(selected["audited"]) or not boolean(selected["unsafe"]):
        raise ValueError("Saved counterexample does not match an audited unsafe measured action")
    if selected["action"] != example["action"]:
        raise ValueError("JSON and action CSV disagree on the selected action")
    with episodes_path.open(newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if len(episodes) != int(metadata["episodes"]):
        raise ValueError("Episode CSV is incomplete relative to the completed manifest")
    episode_row = next(row for row in episodes if int(row["episode"]) == episode)
    if int(episode_row["timeout_skips"]):
        raise ValueError("This figure requires zero skipped callbacks in the example episode")
    future = {int(row["step"]): row for row in measured if step <= int(row["step"]) < step + 6}
    reference_action = next(action for action in MOVEMENT_ORDER
                            if example["surviving_paths"][action] is not None)
    paths = example["surviving_paths"]
    if paths[example["action"]] is not None:
        raise ValueError("Selected action has a surviving reference continuation")
    return metadata, example, selected, episode_row, future, reference_action, (
        metadata_path, actions_path, episodes_path)


def blast_tiles(field, position):
    """Draw the current game's radius-three, wall-stopped blast geometry."""
    result = {tuple(position)}
    for dx, dy in list(MOVES.values())[:4]:
        for distance in range(1, 4):
            x, y = position[0] + dx * distance, position[1] + dy * distance
            if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]) or field[x, y] == -1:
                break
            result.add((x, y))
    return result


def draw_board(axis, example, reference_action):
    state = example["state"]
    field, explosions = np.asarray(state["field"]), np.asarray(state["explosion_map"])
    position = tuple(state["self"][-1])
    future_blast = set()
    for location, timer in state["bombs"]:
        future_blast.update(blast_tiles(field, location))
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            color = {-1: "#46515c", 0: "#fafafa", 1: "#dbb87d"}[int(field[x, y])]
            axis.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=color,
                                     edgecolor="#e2e3e5", linewidth=.35))
            if (x, y) in future_blast:
                axis.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#df8d43",
                                         edgecolor="none", alpha=.23))
            if explosions[x, y] > 0:
                axis.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#c84b52",
                                         edgecolor="none", alpha=.5))
    for location, timer in state["bombs"]:
        axis.add_patch(Circle(tuple(location), .31, color="#20272e", zorder=4))
        axis.text(*location, str(timer), color="white", ha="center", va="center", fontsize=8, zorder=5)
    for index, other in enumerate(state["others"], 1):
        location = tuple(other[-1])
        axis.add_patch(Circle(location, .30, color="#8d5d91", zorder=6))
        axis.text(*location, f"E{index}", color="white", ha="center", va="center", fontsize=7, zorder=7)
    axis.add_patch(Circle(position, .33, color="#246b8e", zorder=8))
    axis.text(*position, "F", color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=9)
    chosen = example["action"]
    if chosen in ("BOMB", "WAIT"):
        axis.add_patch(Circle(position, .49, fill=False, color="#c84b52", linewidth=2.5, zorder=10))
    else:
        delta = MOVES[chosen]
        axis.annotate("", xy=(position[0] + delta[0], position[1] + delta[1]), xytext=position,
                      arrowprops={"arrowstyle": "->", "color": "#c84b52", "lw": 2.7}, zorder=10)
    reference = example["surviving_paths"][reference_action]
    reference_locations = defaultdict(list)
    current = position
    for number, action in enumerate(reference, 1):
        delta = MOVES[action]
        nxt = current[0] + delta[0], current[1] + delta[1]
        if nxt != current:
            axis.annotate("", xy=nxt, xytext=current,
                          arrowprops={"arrowstyle": "->", "color": "#27845d", "lw": 1.8}, zorder=11)
        reference_locations[nxt].append(str(number))
        current = nxt
    for location, numbers in reference_locations.items():
        axis.text(location[0] + .33, location[1] - .29, ",".join(numbers), color="#176344",
                  fontsize=6.8, weight="bold", zorder=12,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .6})
    axis.set_aspect("equal")
    axis.set_xlim(-.5, field.shape[0] - .5)
    axis.set_ylim(field.shape[1] - .5, -.5)
    axis.set_xticks([0, 4, 8, 12, 16])
    axis.set_yticks([0, 4, 8, 12, 16])
    axis.tick_params(labelsize=8)
    axis.set_xlabel("x", fontsize=8)
    axis.set_ylabel("y", fontsize=8)
    axis.set_title(f"Observed board at step {example['step']}", fontsize=10)
    axis.legend(handles=[
        Line2D([0], [0], color="#df8d43", linewidth=5, alpha=.5, label="Existing-bomb blast area"),
        Line2D([0], [0], color="#c84b52", linewidth=2.3, label=f"Selected action: {chosen}"),
        Line2D([0], [0], color="#27845d", linewidth=2.0, label=f"Conditional alternative: {reference_action}"),
    ], loc="upper left", bbox_to_anchor=(-.01, -.13), fontsize=7.8, frameon=False)


def plot(directory, output):
    metadata, example, selected, episode, future, reference_action, sources = load_example(directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, (board, details) = plt.subplots(1, 2, figsize=(9.5, 5.2),
                                       gridspec_kw={"width_ratios": [1, 1.05]})
    fig.subplots_adjust(left=.06, right=.99, top=.86, bottom=.22, wspace=.24)
    draw_board(board, example, reference_action)
    details.axis("off")
    details.set_title("Measured next actions and conditional reference", fontsize=10, pad=14)
    reference = example["surviving_paths"][reference_action]
    first_step = int(example["step"])
    table_rows = []
    for offset in range(6):
        measured = future.get(first_step + offset)
        table_rows.append([str(first_step + offset),
                           measured["action"] if measured else "—",
                           measured["executed_action"] if measured else "—",
                           reference[offset]])
    table = details.table(cellText=table_rows,
                          colLabels=["Step", "Chosen", "After timeout", "Reference"],
                          cellLoc="center", bbox=[0, .47, 1, .50])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d8dc")
        if row == 0:
            cell.set_facecolor("#eef1f3")
            cell.set_text_props(weight="bold")
        elif col == 3:
            cell.set_facecolor("#e5f1e9")
        elif row == 1:
            cell.set_facecolor("#f7e4e6")
    death = boolean(selected["death_within_six"])
    details.text(0, .40, f"Death within six steps: {'yes' if death else 'no'}\n"
                 f"Episode score: {episode['score']}  ·  "
                 f"Episode survived: {'yes' if int(episode['survived']) else 'no'}",
                 transform=details.transAxes, va="top", fontsize=9, linespacing=1.5)
    details.text(0, .24, "The selected action has no six-step surviving\n"
                 "continuation under the saved conditional reference.\n"
                 "Green arrows show one available alternative;\n"
                 "the rightmost column is its full action sequence.",
                 transform=details.transAxes, va="top", fontsize=8.5, linespacing=1.5)
    details.text(0, .01, "F = focal agent; E = observed opponent; bomb digits = timer.\n"
                 "Orange shows blast geometry, without timing.\n"
                 "Red cell shading marks an initially active explosion.\n"
                 "A dash means no later recorded focal callback.\n"
                 "After timeout denotes engine input; movement legality\n"
                 "is not recorded separately for each action.",
                 transform=details.transAxes, va="top", fontsize=7.5, color="#505c67", linespacing=1.5)
    fig.suptitle("A safety failure selected during measured gameplay", fontsize=13, y=.97)
    fig.text(.5, .90, f"Frozen final-agent weights + RUEHL action mask; three rule-based opponents; "
             f"episode {example['episode']}, seed {selected['seed']}", ha="center", fontsize=8.7)
    fig.text(.06, .065, "Selection: first recorded unsafe audited action, independent of its outcome. "
             "Reference: existing bombs/explosions only;\n"
             "current opponents block the first move, and their future actions/new bombs are unknown. "
             "This is conditional viability, not proof of a winning counterfactual.",
             fontsize=8.1, color="#424e59", linespacing=1.5)
    output.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(output / f"gameplay_failure.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    provenance = {"selection_rule": "counterexamples[0]: first recorded unsafe audited selected action, independent of outcome",
                  "reference_selection_rule": "First surviving action in UP, RIGHT, DOWN, LEFT, WAIT order",
                  "config": metadata["config"], "opponents": metadata["opponents"],
                  "episode": int(example["episode"]), "step": first_step, "seed": int(selected["seed"]),
                  "selected_action": example["action"], "reference_action": reference_action,
                  "reference_path": reference, "death_within_six": death,
                  "scope": metadata["audit_scope"],
                  "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (directory / "gameplay_failure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "results/frozen")
    parser.add_argument("--out", type=Path, default=HERE.parent / "figures")
    args = parser.parse_args()
    plot(args.data.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
