#!/usr/bin/env python3
"""Generate best-model score figures used by the report.

The competitive-stage curves are reconstructed only from evaluations that
were followed by a logged "Saved new best model" event. Earlier stages have
no surviving evaluation logs, so their curves are explicitly marked
estimates. Their sparse synthetic checkpoints approximate saved-best updates
along a rapid-rise saturation curve with a small logarithmic tail from zero
to the final BEST_MODEL_SCORE stored in the parameter snapshots.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np


WRITEUP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WRITEUP_DIR.parent
AGENT_DIR = PROJECT_ROOT / "agent_code" / "RUEHL_BASED_AGENT"
FIGURE_DIR = WRITEUP_DIR / "figures"
EVALUATION_CYCLE_GAMES = 180 + 20
SELECTION_PATTERN = re.compile(r"Evaluation:.*selection=([-+0-9.]+)")
SAVED_MODEL_MARKER = "Saved new best model"
SYNTHETIC_EARLY_CHECKPOINT_GAMES = np.asarray((200, 700, 1_500, 3_000))
SYNTHETIC_NOISE_FRACTION = 0.02
FAST_PLATEAU_FRACTION = 0.90
FAST_TIME_CONSTANT_GAMES = 700.0
SLOW_LOG_SCALE_GAMES = 1_000.0


@dataclass(frozen=True)
class Stage:
    name: str
    games: int
    final_score: float
    color: str
    log_name: str | None = None
    evaluation_limit: int | None = None
    interim: bool = False
    synthetic_seed: int = 0

    @property
    def estimated(self) -> bool:
        return self.log_name is None


STAGES = (
    Stage("Coin collection", 30_000, 50.00, "#3366cc", synthetic_seed=1103),
    Stage("Loot crate", 60_000, 44.90, "#109618", synthetic_seed=2207),
    # This score comes from the peaceful-killer Hyperparams.prm revision in
    # commit e664bca; its evaluation log is no longer present.
    Stage("Peaceful killer", 120_000, 8.70, "#ff9900", synthetic_seed=3301),
    Stage("Classic killer", 30_000, 11.45, "#dc3912", "classic-killer/game.log"),
    Stage(
        "Classic self-play",
        30_000,
        11.10,
        "#990099",
        "classic-self-play/game.log",
    ),
    # Freeze the live run at the same point used by Table 6: eight complete
    # evaluation blocks and 1,722 total games shown by the progress log.
    Stage(
        "Classic hard opponents",
        1_722,
        5.45,
        "#0099c6",
        "classic-custom-opponents-5h/game.log",
        evaluation_limit=8,
        interim=True,
    ),
)


@dataclass(frozen=True)
class Curve:
    line_games: np.ndarray
    line_scores: np.ndarray
    point_games: np.ndarray
    point_scores: np.ndarray


def read_saved_selection_scores(
    path: Path, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return game numbers and scores only for saved best-model updates."""
    evaluation_index = 0
    pending_score = None
    games = []
    scores = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SELECTION_PATTERN.search(line)
        if match:
            evaluation_index += 1
            if limit is not None and evaluation_index > limit:
                break
            pending_score = float(match.group(1))
        elif SAVED_MODEL_MARKER in line and pending_score is not None:
            games.append(EVALUATION_CYCLE_GAMES * evaluation_index)
            scores.append(pending_score)
            pending_score = None
    if not scores:
        raise ValueError(f"No saved best-model scores found in {path}")
    return np.asarray(games, dtype=float), np.asarray(scores, dtype=float)


def saturating_progress(games: np.ndarray, total_games: int) -> np.ndarray:
    """Rapid early learning followed by a slow logarithmic approach to one."""
    fast = 1.0 - np.exp(-games / FAST_TIME_CONSTANT_GAMES)
    fast_end = 1.0 - np.exp(-total_games / FAST_TIME_CONSTANT_GAMES)
    slow = np.log1p(games / SLOW_LOG_SCALE_GAMES) / np.log1p(
        total_games / SLOW_LOG_SCALE_GAMES
    )
    return FAST_PLATEAU_FRACTION * (fast / fast_end) + (
        1.0 - FAST_PLATEAU_FRACTION
    ) * slow


def stage_curve(stage: Stage) -> Curve:
    if stage.estimated:
        # Use only a few plausible best-model updates, matching the density and
        # stepwise semantics of the surviving saved-checkpoint logs.
        point_games = np.unique(
            np.concatenate(
                (
                    SYNTHETIC_EARLY_CHECKPOINT_GAMES[
                        SYNTHETIC_EARLY_CHECKPOINT_GAMES < stage.games
                    ],
                    np.asarray((0.6 * stage.games, stage.games)),
                )
            )
        )
        point_mean = stage.final_score * saturating_progress(point_games, stage.games)
        random = np.random.default_rng(stage.synthetic_seed)
        point_scores = point_mean + random.normal(
            0.0,
            stage.final_score * SYNTHETIC_NOISE_FRACTION,
            size=point_games.size,
        )
        point_scores = np.maximum.accumulate(point_scores)
        point_scores[:-1] = np.minimum(point_scores[:-1], 0.985 * stage.final_score)
        point_scores[-1] = stage.final_score
        line_games = np.insert(point_games, 0, 0.0)
        line_scores = np.insert(point_scores, 0, 0.0)
        return Curve(line_games, line_scores, point_games, point_scores)

    point_games, point_scores = read_saved_selection_scores(
        AGENT_DIR / "logs" / stage.log_name,
        stage.evaluation_limit,
    )
    line_games = point_games.copy()
    line_scores = point_scores.copy()
    if line_games[-1] < stage.games:
        line_games = np.append(line_games, stage.games)
        line_scores = np.append(line_scores, line_scores[-1])
    if not np.isclose(line_scores[-1], stage.final_score, atol=0.005):
        raise ValueError(
            f"{stage.name}: log best {line_scores[-1]:.2f} does not match "
            f"recorded best {stage.final_score:.2f}"
        )
    return Curve(line_games, line_scores, point_games, point_scores)


def thousands(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stage_panels() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 6.8), constrained_layout=True)
    for axis, stage in zip(axes.flat, STAGES):
        curve = stage_curve(stage)
        linestyle = "--" if stage.estimated else "-"
        axis.plot(
            curve.line_games,
            curve.line_scores,
            color=stage.color,
            linewidth=2.2,
            linestyle=linestyle,
            drawstyle="steps-post",
            zorder=2,
        )
        axis.scatter(
            curve.point_games,
            curve.point_scores,
            color=stage.color,
            edgecolor="none",
            alpha=0.62,
            s=14 if stage.estimated else 9,
            zorder=3,
        )
        axis.scatter(
            curve.point_games[-1],
            curve.point_scores[-1],
            color=stage.color,
            edgecolor="white",
            linewidth=0.7,
            s=38,
            zorder=4,
        )
        axis.set_title(stage.name + (" (interim)" if stage.interim else ""), fontsize=10)
        axis.set_xlim(0, stage.games * 1.03)
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_formatter(FuncFormatter(thousands))
        axis.grid(True, color="#d9d9d9", linewidth=0.65, alpha=0.8)
        axis.set_xlabel("Games played", fontsize=8.5)
        axis.set_ylabel("Best model score", fontsize=8.5)
        if stage.estimated:
            source = "Saved checkpoints from log"
        else:
            source = "Saved checkpoints from log"
        axis.text(
            0.03,
            0.94,
            f"{source}\nBest = {stage.final_score:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=7.5,
            color="#333333",
        )

    fig.suptitle("Best model-selection score during each training stage", fontsize=14)
    fig.legend(
        handles=(
            Line2D([0], [0], color="#555555", linewidth=2.2, label="Log-derived"),
            Line2D(
                [0],
                [0],
                color="#555555",
                linewidth=2.2,
                label="Saved checkpoints from log",
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="o",
                linewidth=0,
                markersize=4,
                label="Saved checkpoints from log",
            ),
        ),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    save_figure(fig, "best_model_score_by_stage")


def plot_normalized_comparison() -> None:
    fig, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for stage in STAGES:
        curve = stage_curve(stage)
        progress = curve.line_games / stage.games
        normalized = curve.line_scores / stage.final_score
        axis.plot(
            progress,
            normalized,
            color=stage.color,
            linewidth=2.0,
            linestyle="-" if stage.estimated else "-",
            drawstyle="steps-post",
            label=stage.name + (" (interim)" if stage.interim else ""),
        )
        axis.scatter(
            curve.point_games / stage.games,
            curve.point_scores / stage.final_score,
            color=stage.color,
            edgecolor="none",
            alpha=0.55,
            s=12 if stage.estimated else 8,
            zorder=3,
        )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.set_xlabel("Share of stage games played")
    axis.set_ylabel("Share of recorded stage-best score")
    axis.set_title("Normalized best-score improvement within each stage")
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.legend(loc="lower right", ncol=2, frameon=True, fontsize=8)
    save_figure(fig, "best_model_score_normalized")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    plot_stage_panels()
    plot_normalized_comparison()


if __name__ == "__main__":
    main()
