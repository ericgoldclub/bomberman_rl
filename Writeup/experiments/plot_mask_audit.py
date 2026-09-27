#!/usr/bin/env python3
"""Create measured-only figures and a LaTeX table from mask_audit.json."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "figures"
DATA = HERE / "results/mask_audit.json"
COLORS = {"ruehl": "#246b8e", "final": "#bd642b"}
NAMES = {"ruehl": "RUEHL mask", "final": "Final-agent mask"}
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
MOVES = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0), "WAIT": (0, 0)}


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def summary_figure(data):
    summaries = data["summaries"][-2:]
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.1), layout="constrained")
    specs = [
        ("Unsafe movement / wait\namong admitted actions", "unsafe_admitted_movement_wait", "admitted_movement_wait"),
        ("Unsafe bomb placements\namong admitted bombs", "unsafe_admitted_bombs", "admitted_bombs"),
        ("Viable movement / wait\nexcluded by the mask", "viable_withheld", "viable_movement_wait"),
    ]
    for panel, (axis, (title, numerator, denominator)) in enumerate(zip(axes, specs)):
        percentages = [100 * s[numerator] / s[denominator] for s in summaries]
        axis.bar([0, 1], percentages, color=[COLORS[s["mask"]] for s in summaries], width=0.57)
        axis.set_xticks([0, 1], ["RUEHL", "Final"])
        axis.set_ylim(0, max(percentages) * 1.36)
        axis.set_ylabel("Admitted actions (%)" if panel < 2 else "Viable actions (%)")
        axis.set_title(title, fontsize=13, pad=12)
        axis.yaxis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        for index, (summary, percent) in enumerate(zip(summaries, percentages)):
            axis.text(index, percent + max(percentages) * 0.04,
                      f"{percent:.2f}%\n{summary[numerator]:,} / {summary[denominator]:,}",
                      ha="center", fontsize=12)
    fig.suptitle("Engine-based action-mask diagnostic", fontsize=16)
    fig.supxlabel("3,000 generated states; metrics restricted to 2,453 states with a surviving continuation", fontsize=12)
    save(fig, "mask_audit_comparison")


def counterexample_figure(data):
    example = next(e for e in data["examples"] if e["state_id"] == "0.3:21")
    field = np.asarray(example["field"])
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.85), gridspec_kw={"width_ratios": [1, 1.18]}, layout="constrained")
    board, details = axes
    board.set_aspect("equal")
    for x in range(7, 13):
        for y in range(5, 12):
            color = {-1: "#46515c", 0: "#fafafa", 1: "#dbb87d"}[field[x, y]]
            board.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=color, edgecolor="#e2e3e5", lw=.6))
    # The local existing bomb at (9,6), timer 1, detonates after action 2.
    for y in range(5, 10):
        board.add_patch(Rectangle((8.5, y - .5), 1, 1, facecolor="#c44d58", alpha=.22, edgecolor="none"))
    board.add_patch(Circle((9, 6), .25, facecolor="#20272e"))
    board.text(9.38, 6, "timer 1", va="center", fontsize=12)
    board.add_patch(Circle(tuple(example["position"]), .26, facecolor=COLORS["ruehl"], zorder=5))
    board.text(9.38, 8, "agent", va="center", fontsize=12)
    path = example["surviving_paths"]["DOWN"][:3]
    position = example["position"]
    for number, action in enumerate(path, 1):
        delta = MOVES[action]
        nxt = (position[0] + delta[0], position[1] + delta[1])
        board.annotate("", xy=nxt, xytext=position,
                       arrowprops={"arrowstyle": "->", "color": "#29845e", "lw": 2.2}, zorder=6)
        board.text(nxt[0] + .13, nxt[1] + .29, str(number), color="#176344", fontsize=12, weight="bold", zorder=7)
        position = nxt
    board.set_xlim(6.5, 12.5)
    board.set_ylim(11.5, 4.5)
    board.set_xticks(range(7, 13))
    board.set_yticks(range(5, 12))
    board.set_xlabel("x")
    board.set_ylabel("y")
    board.set_title("Generated state 0.3:21 (cropped)", fontsize=14)
    details.axis("off")
    details.set_title("Allowed actions versus engine viability", fontsize=14, pad=12)
    columns = ["Action", "RUEHL", "Final", "Viable"]
    rows = [[action, "yes" if example["ruehl_mask"][index] else "no",
             "yes" if example["final_mask"][index] else "no",
             "yes" if example["surviving_paths"][action] is not None else "no"]
            for index, action in enumerate(ACTIONS)]
    table = details.table(cellText=rows, colLabels=columns, cellLoc="center", bbox=[0.02, .33, .96, .63])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d8dc")
        if row == 0:
            cell.set_facecolor("#eef1f3")
            cell.set_text_props(weight="bold")
        if row == 3:
            cell.set_facecolor("#e3f1e9")
        if row == 6:
            cell.set_facecolor("#f5e4e5")
    details.text(.03, .25, "DOWN escapes the blast before it arrives.\nBOMB spends the first step on the agent's tile;\nno continuation then survives six steps.",
                 transform=details.transAxes, fontsize=12, va="top", linespacing=1.5)
    board.legend(handles=[Line2D([0], [0], color="#c44d58", lw=5, alpha=.45, label="Blast at steps 2 and 3"),
                          Line2D([0], [0], color="#29845e", lw=2, label="First three surviving moves")],
                 loc="upper left", bbox_to_anchor=(0, -.28), fontsize=11, frameon=False)
    save(fig, "mask_audit_counterexample")


def write_table(data):
    r, f = data["summaries"][-2:]
    values = [
        ("Unsafe admitted movement/wait", r["unsafe_admitted_movement_wait"], r["admitted_movement_wait"], f["unsafe_admitted_movement_wait"], f["admitted_movement_wait"]),
        ("Unsafe admitted bombs", r["unsafe_admitted_bombs"], r["admitted_bombs"], f["unsafe_admitted_bombs"], f["admitted_bombs"]),
        ("Viable movement/wait excluded", r["viable_withheld"], r["viable_movement_wait"], f["viable_withheld"], f["viable_movement_wait"]),
    ]
    lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabularx}{\textwidth}{@{}Yrr@{}}", r"\toprule",
             r"Diagnostic & RUEHL mask & Final-agent mask \\", r"\midrule"]
    for name, rn, rd, fn, fd in values:
        lines.append(f"{name} & ${rn}/{rd}$ ({100*rn/rd:.2f}\\%) & ${fn}/{fd}$ ({100*fn/fd:.2f}\\%) \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\caption{Measured diagnostics on 2,453 recoverable states from the 3,000-state suite. Each fraction uses its own candidate-action denominator.}",
              r"\label{tab:jesper-mask-audit}", r"\end{table}"]
    (HERE / "results/mask_audit_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    data = json.loads(DATA.read_text())
    summary_figure(data)
    counterexample_figure(data)
    write_table(data)
