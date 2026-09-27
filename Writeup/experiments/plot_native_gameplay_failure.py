#!/usr/bin/env python3
"""Draw the recorded observation/execution failure in native episode zero."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0),
         "WAIT": (0, 0), "BOMB": (0, 0)}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def local_board(axis, snapshot, chosen, reference=None, later_enemy=None):
    state = snapshot["state"]
    field = np.asarray(state["field"])
    origin = tuple(state["self"][-1])
    for x in range(6, 14):
        for y in range(12, 17):
            axis.add_patch(Rectangle((x - .5, y - .5), 1, 1,
                                     facecolor={-1: "#46515c", 0: "#fafafa", 1: "#dbb87d"}[int(field[x, y])],
                                     edgecolor="#dce0e4", lw=.5))
    for (bx, by), timer in state["bombs"]:
        blast = {(bx, by)}
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for distance in range(1, 4):
                x, y = bx + dx * distance, by + dy * distance
                if field[x, y] == -1:
                    break
                blast.add((x, y))
        for x, y in blast:
            if 6 <= x < 14 and 12 <= y < 17:
                axis.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#c84b52",
                                         edgecolor="none", alpha=.27 if timer == 0 else .13))
        axis.add_patch(Circle((bx, by), .30, color="#20272e", zorder=5))
        axis.text(bx, by, str(timer), color="white", ha="center", va="center", fontsize=12, zorder=6)
    for enemy in state["others"]:
        location = tuple(enemy[-1])
        axis.add_patch(Circle(location, .32, color="#8d5d91", zorder=6))
        axis.text(*location, "E", color="white", ha="center", va="center", fontsize=12, zorder=7)
    if reference:
        current = origin
        for number, action in enumerate(reference[:3], 1):
            dx, dy = MOVES[action]
            nxt = current[0] + dx, current[1] + dy
            if nxt != current:
                axis.annotate("", xy=nxt, xytext=current,
                              arrowprops={"arrowstyle": "->", "color": "#27845d", "lw": 1.8,
                                          "linestyle": "--"}, zorder=8)
            axis.text(nxt[0] + .29, nxt[1] - .26, str(number), color="#176344", fontsize=11,
                      weight="bold", zorder=9)
            current = nxt
    axis.add_patch(Circle(origin, .33, color="#246b8e", zorder=10))
    axis.text(*origin, "F", color="white", ha="center", va="center", fontsize=12, weight="bold", zorder=11)
    if chosen == "BOMB":
        axis.add_patch(Circle(origin, .46, fill=False, color="#20272e", lw=2, zorder=12))
    elif snapshot["step"] == 216:
        dx, dy = MOVES[chosen]
        nxt = origin[0] + dx, origin[1] + dy
        axis.annotate("", xy=nxt, xytext=origin,
                      arrowprops={"arrowstyle": "->", "color": "#c84b52", "lw": 2.8}, zorder=12)
        axis.plot(*nxt, marker="x", color="#ab303c", markersize=10, markeredgewidth=2.2, zorder=13)
    if later_enemy:
        old = tuple(state["others"][0][-1])
        axis.annotate("", xy=later_enemy, xytext=old,
                      arrowprops={"arrowstyle": "->", "color": "#8d5d91", "lw": 2.4}, zorder=12)
        axis.add_patch(Circle(later_enemy, .37, fill=False, color="#8d5d91", lw=1.8, linestyle="--", zorder=12))
    axis.set_aspect("equal")
    axis.set_xlim(5.5, 13.5)
    axis.set_ylim(16.5, 11.5)
    axis.set_xticks([6, 8, 10, 12])
    axis.set_yticks([12, 14, 16])
    axis.tick_params(labelsize=11)
    axis.set_xlabel("x", fontsize=11)
    axis.set_ylabel("y", fontsize=11)


def plot(trace_directory, primary_directory, output):
    observations_path = trace_directory / "observed_states.json"
    actions_path = trace_directory / "final_native_strong_actions.csv"
    primary_path = primary_directory / "final_native_strong_actions.csv"
    episode_path = trace_directory / "final_native_strong.csv"
    snapshots = {int(row["step"]): row for row in json.loads(observations_path.read_text())["snapshots"]}
    actions = read_csv(actions_path)
    primary = [row for row in read_csv(primary_path) if int(row["episode"]) == 0]
    keys = ("step", "action", "executed_action", "bomb_dropped", "threat", "predicted_trap", "death_within_six", "timeout")
    if len(actions) != len(primary) or any(tuple(a[key] for key in keys) != tuple(b[key] for key in keys)
                                         for a, b in zip(actions, primary)):
        raise ValueError("Exploratory trace does not replay primary actions and outcomes exactly")
    episode = read_csv(episode_path)[0]
    if int(episode["timeout_skips"]) or int(episode["suicides"]) != 1:
        raise ValueError("Unexpected episode outcome for this mechanism trace")
    by_step = {int(row["step"]): row for row in actions}
    at215, at216 = snapshots[215], snapshots[216]
    assert at215["state"]["self"][-1] == at216["state"]["self"][-1] == [10, 15]
    assert at215["state"]["others"][0][-1] == [11, 14]
    assert at216["state"]["others"][0][-1] == [11, 15]
    assert at215["reconstructed_action_order"] == ["opponent_2", "focal"]
    assert at215["paths"]["RIGHT"] is not None and not any(at216["paths"].values())
    assert by_step[215]["action"] == "RIGHT" and by_step[216]["action"] == "LEFT"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 4.25))
    fig.subplots_adjust(left=.06, right=.99, top=.94, bottom=.35, wspace=.25)
    specs = [
        (212, "BOMB permits an escape", "BOMB", snapshots[212]["paths"]["BOMB"], None,
         "F places BOMB at (8,15).\nA conditional escape exists."),
        (215, "RIGHT appears viable", "RIGHT", at215["paths"]["RIGHT"], (11, 15),
         "E moves first into (11,15).\nRIGHT fails; F stays at (10,15)."),
        (216, "Escape is blocked", "LEFT", None, None,
         "LEFT enters the own blast.\nF dies as timer 0 detonates."),
    ]
    for axis, (step, title, action, reference, later, caption) in zip(axes, specs):
        local_board(axis, snapshots[step], action, reference, later)
        axis.set_title(f"Step {step}\n{title}", fontsize=12, pad=10)
        axis.text(0, -.32, caption, transform=axis.transAxes, va="top", fontsize=11,
                  linespacing=1.6, color="#424e59")
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#246b8e", label="F: focal"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="#8d5d91", label="E: enemy"),
              Line2D([0], [0], color="#c84b52", lw=5, alpha=.3, label="Bomb blast; digit = timer")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .12), ncol=3,
               frameon=False, fontsize=11)
    fig.text(.5, .065, "Dashed path: conditional survival; future enemy actions are unknown.",
             ha="center", fontsize=11, color="#424e59")
    output.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(output / f"native_gameplay_failure.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    provenance = {
        "selection_rule": "Predetermined first native strong episode; inspect its actual first self-death, not a sample selected by outcome severity",
        "episode": 0, "seed": 270920260, "action_sequence_matches_primary": True,
        "steps_shown": [212, 215, 216], "conditional_reference_at215": at215["paths"]["RIGHT"],
        "action_order_at215": at215["reconstructed_action_order"],
        "mechanism": "Opponent moves DOWN from (11,14) into (11,15) before focal RIGHT; focal remains (10,15). The next observed state admits no surviving action. Focal LEFT executes into (9,15), and its timer-zero bomb at (8,15) detonates.",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                           (observations_path, actions_path, primary_path, episode_path, Path(__file__))},
    }
    (trace_directory / "native_gameplay_failure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=HERE / "results/native_trace_capture")
    parser.add_argument("--primary", type=Path, default=HERE / "results/frozen")
    parser.add_argument("--out", type=Path, default=HERE.parent / "figures")
    args = parser.parse_args()
    plot(args.trace.resolve(), args.primary.resolve(), args.out.resolve())
