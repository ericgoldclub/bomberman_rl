#!/usr/bin/env python3
"""Analyze frozen-policy results with episode-level bootstrap intervals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_LABELS = {
    "final_native": "Final Q + temporal mask",
    "final_ruehl_mask": "Final Q + RUEHL mask",
    "random_final_mask": "Random + temporal mask",
    "random_ruehl_mask": "Random + RUEHL mask",
    "ruehl_native": "RUEHL Q + RUEHL mask",
    "ruehl_final_mask": "RUEHL Q + temporal mask",
}
OPPONENT_LABELS = {
    "strong": "Three rule-based opponents",
    "mixed": "Mixed opponent population",
}
OPPONENT_COLORS = {"strong": "#246b8e", "mixed": "#bd642b"}
METRICS = (
    "score", "coins", "kills", "suicides", "survived", "win_share",
    "episode_steps", "bombs", "invalid", "timeout_events", "timeout_skips", "action_count",
    "unsafe_selected", "checked_actions", "latency_median_ms",
    "latency_p95_ms", "latency_max_ms",
    "threat_bombs", "trapped_threat_bombs", "audited_deaths", "audited_self_deaths",
    "unsafe_selected_deaths",
)
REQUIRED = ("config", "opponents", "episode", "seed", "layout_sha256", "score")
PRIMARY_CONTRASTS = (
    ("final_native", "ruehl_native", "Native final minus native RUEHL"),
    ("final_native", "final_ruehl_mask", "Temporal mask effect, final Q"),
    ("random_final_mask", "random_ruehl_mask", "Temporal mask effect, random"),
    ("final_native", "random_final_mask", "Final Q effect, temporal mask"),
    ("final_ruehl_mask", "random_ruehl_mask", "Final Q effect, RUEHL mask"),
    ("ruehl_final_mask", "ruehl_native", "Temporal mask effect, RUEHL Q"),
)


def label(config: str) -> str:
    return CONFIG_LABELS.get(config, config.replace("_", " "))


def opponent_sort_key(opponents: str) -> tuple[int, str]:
    return ({"strong": 0, "mixed": 1}.get(opponents, 2), opponents)


def set_plot_style() -> None:
    """Keep text legible when a 10.4-inch figure is placed at 16 cm width."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.labelsize": 13, "axes.titlesize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 11.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng_for(seed: int, key: str) -> np.random.Generator:
    # Stable across process hash randomization and addition/removal of groups.
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def bootstrap_means(values: np.ndarray, samples: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Use the same complete-episode resamples for all columns in a group."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n = len(values)
    output = np.empty((samples, values.shape[1]))
    for begin in range(0, samples, 1000):
        end = min(samples, begin + 1000)
        draws = rng.integers(0, n, size=(end - begin, n))
        output[begin:end] = values[draws].mean(axis=1)
    return output


def percentile_interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def wilson_interval(successes: float, n: int) -> list[float]:
    if n < 1:
        raise ValueError("A Wilson interval needs at least one episode")
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    combined = path / "frozen_evaluation.csv"
    if combined.is_file():
        return [combined]
    # Schema identifies evaluation files; unrelated diagnostic CSVs are skipped.
    paths = []
    for candidate in sorted(path.glob("*.csv")):
        if candidate.name.endswith("_actions.csv"):
            continue
        with candidate.open(newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        if all(column in fields for column in REQUIRED):
            paths.append(candidate)
    if not paths:
        raise ValueError(f"No episode-result CSVs found in {path}")
    return paths


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(REQUIRED) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for line, source in enumerate(reader, 2):
                row: dict[str, Any] = {}
                for key in REQUIRED:
                    value = (source.get(key) or "").strip()
                    if not value:
                        raise ValueError(f"{path}:{line}: empty {key}")
                    row[key] = value
                key = (row["config"], row["opponents"], row["seed"])
                if key in seen:
                    raise ValueError(f"Duplicate (config, opponents, seed): {key}")
                seen.add(key)
                for metric in METRICS:
                    raw = source.get(metric)
                    if raw is None or not raw.strip():
                        continue
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError(f"{path}:{line}: non-finite {metric}")
                    row[metric] = value
                if "survived" in row and row["survived"] not in (0, 1):
                    raise ValueError(f"{path}:{line}: survived must be binary")
                if "win_share" in row and not 0 <= row["win_share"] <= 1:
                    raise ValueError(f"{path}:{line}: win_share must lie in [0, 1]")
                for metric in ("coins", "kills", "suicides", "episode_steps", "bombs",
                               "invalid", "timeout_events", "timeout_skips", "action_count", "unsafe_selected",
                               "checked_actions", "latency_median_ms", "latency_p95_ms",
                               "latency_max_ms", "threat_bombs", "trapped_threat_bombs",
                               "audited_deaths", "audited_self_deaths", "unsafe_selected_deaths"):
                    if row.get(metric, 0) < 0:
                        raise ValueError(f"{path}:{line}: negative {metric}")
                if ("unsafe_selected" in row and "checked_actions" in row
                        and row["unsafe_selected"] > row["checked_actions"]):
                    raise ValueError(f"{path}:{line}: unsafe_selected exceeds checked_actions")
                rows.append(row)
    if not rows:
        raise ValueError("Input contains no episodes")
    # Optional metrics must be complete within each configuration/opponent group.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["config"], row["opponents"]), []).append(row)
    for key, group in groups.items():
        for metric in METRICS:
            present = sum(metric in row for row in group)
            if present not in (0, len(group)):
                raise ValueError(f"Metric {metric} present in only {present}/{len(group)} rows of {key}")
    return sorted(rows, key=lambda r: (r["opponents"], r["config"], r["seed"]))


def summarize_group(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    config, opponents = rows[0]["config"], rows[0]["opponents"]
    metrics = [metric for metric in METRICS if metric in rows[0]]
    values = np.asarray([[row[metric] for metric in metrics] for row in rows])
    draws = bootstrap_means(values, samples, rng_for(seed, f"group:{config}:{opponents}"))
    summary: dict[str, Any] = {
        "config": config, "label": label(config), "opponents": opponents,
        "episodes": len(rows), "seeds": [row["seed"] for row in rows], "metrics": {},
    }
    for column, metric in enumerate(metrics):
        data = values[:, column]
        binary = metric in ("survived", "suicides") and set(data).issubset({0.0, 1.0})
        summary["metrics"][metric] = {
            "mean": float(data.mean()), "total": float(data.sum()),
            "sd": float(data.std(ddof=1)) if len(data) > 1 else None,
            "ci95": wilson_interval(float(data.sum()), len(data)) if binary else percentile_interval(draws[:, column]),
            "interval_method": "Wilson" if binary else "episode percentile bootstrap",
        }
    if "win_share" in metrics:
        shares = values[:, metrics.index("win_share")]
        leading = shares > 0
        summary["metrics"]["score_leader_presence"] = {
            "mean": float(leading.mean()), "total": int(leading.sum()),
            "ci95": wilson_interval(float(leading.sum()), len(rows)),
            "interval_method": "Wilson on Bernoulli indicator win_share > 0",
            "definition": "Episode has positive score-leader credit, including tied leaders.",
        }
        summary["metrics"]["win_share"]["boundary_degenerate"] = bool(np.all(shares == 0))
        if np.all(shares == 0):
            summary["metrics"]["win_share"]["boundary_note"] = (
                "Raw empirical bootstrap interval is degenerate because no positive share was observed; "
                "this does not exclude rare score-leading episodes. The Wilson upper bound for score_leader_presence "
                "also conservatively bounds expected fractional win share."
            )
    if "unsafe_selected" in metrics and "checked_actions" in metrics:
        num_col, den_col = metrics.index("unsafe_selected"), metrics.index("checked_actions")
        denominator = values[:, den_col].sum()
        if denominator > 0:
            valid = draws[:, den_col] > 0
            summary["metrics"]["unsafe_selection_rate"] = {
                "mean": float(values[:, num_col].sum() / denominator),
                "numerator": int(values[:, num_col].sum()), "denominator": int(denominator),
                "ci95": percentile_interval(draws[valid, num_col] / draws[valid, den_col]),
                "interval_method": "episode cluster percentile bootstrap of ratio of totals",
                "zero_denominator_resamples": int((~valid).sum()),
            }
    return summary


def paired_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_seeds, right_seeds = {r["seed"]: r for r in left}, {r["seed"]: r for r in right}
    seeds = sorted(left_seeds.keys() & right_seeds.keys())
    for seed in seeds:
        if left_seeds[seed]["layout_sha256"] != right_seeds[seed]["layout_sha256"]:
            raise ValueError(f"Paired-layout mismatch for seed {seed}: "
                             f"{left_seeds[seed]['config']} / {right_seeds[seed]['config']}")
    return [left_seeds[s] for s in seeds], [right_seeds[s] for s in seeds]


def contrasts(groups: dict[tuple[str, str], list[dict[str, Any]]], samples: int,
              seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results, omitted = [], []
    opponents_all = sorted({key[1] for key in groups}, key=opponent_sort_key)
    for opponents in opponents_all:
        for left_config, right_config, name in PRIMARY_CONTRASTS:
            if (left_config, opponents) not in groups or (right_config, opponents) not in groups:
                continue
            left_group, right_group = groups[left_config, opponents], groups[right_config, opponents]
            left, right = paired_rows(left_group, right_group)
            if not left:
                omitted.append({"name": name, "opponents": opponents, "reason": "no common seeds"})
                continue
            metrics = [m for m in METRICS if m in left[0] and m in right[0]]
            differences = np.asarray([[l[m] - r[m] for m in metrics] for l, r in zip(left, right)])
            draws = bootstrap_means(differences, samples, rng_for(seed, f"contrast:{left_config}:{right_config}:{opponents}"))
            result: dict[str, Any] = {
                "name": name, "left": left_config, "right": right_config,
                "opponents": opponents, "paired_episodes": len(left),
                "seeds": [r["seed"] for r in left],
                "unpaired_left_episodes": len(left_group) - len(left),
                "unpaired_right_episodes": len(right_group) - len(right),
                "metrics": {},
            }
            for index, metric in enumerate(metrics):
                result["metrics"][metric] = {
                    "mean_difference": float(differences[:, index].mean()),
                    "ci95": percentile_interval(draws[:, index]),
                    "interval_method": "paired episode percentile bootstrap",
                }
            results.append(result)
    return results, omitted


def factorial_interactions(groups: dict[tuple[str, str], list[dict[str, Any]]],
                           samples: int, seed: int) -> list[dict[str, Any]]:
    """Mask effect under learned Q-values minus its effect under random actions."""
    configs = ("final_native", "final_ruehl_mask", "random_final_mask", "random_ruehl_mask")
    signs = np.asarray([1, -1, -1, 1])
    results = []
    for opponents in sorted({key[1] for key in groups}, key=opponent_sort_key):
        if not all((config, opponents) in groups for config in configs):
            continue
        by_seed = [{r["seed"]: r for r in groups[config, opponents]} for config in configs]
        shared = sorted(set.intersection(*(set(g) for g in by_seed)))
        if not shared:
            continue
        metrics = [m for m in METRICS if all(m in g[shared[0]] for g in by_seed)]
        values = []
        for game_seed in shared:
            rows = [g[game_seed] for g in by_seed]
            if len({r["layout_sha256"] for r in rows}) != 1:
                raise ValueError(f"Factorial layout mismatch for seed {game_seed}")
            values.append(np.asarray([[r[m] for m in metrics] for r in rows]).T @ signs)
        values = np.asarray(values)
        draws = bootstrap_means(values, samples, rng_for(seed, f"interaction:{opponents}"))
        results.append({
            "name": "Learned-selection by mask interaction", "opponents": opponents,
            "definition": "(final_native - final_ruehl_mask) - (random_final_mask - random_ruehl_mask)",
            "paired_episodes": len(shared), "seeds": shared,
            "metrics": {m: {"mean_difference": float(values[:, i].mean()),
                            "ci95": percentile_interval(draws[:, i]),
                            "interval_method": "paired four-configuration episode percentile bootstrap"}
                        for i, m in enumerate(metrics)},
        })
    return results


def action_input_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Actions input does not exist: {path}")
    paths = sorted(path.glob("*_actions.csv"))
    if not paths:
        raise ValueError(f"No *_actions.csv files found in {path}")
    return paths


def read_actions(paths: list[Path], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_episodes = {(r["config"], r["opponents"], r["seed"]) for r in episodes}
    seen = set()
    rows = []
    required = {"config", "opponents", "episode", "seed", "step", "latency_ms"}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing action columns {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                for field in required:
                    row[field] = (row[field] or "").strip()
                    if not row[field]:
                        raise ValueError(f"{path}:{line}: empty action {field}")
                key = (row["config"], row["opponents"], row["seed"])
                if key not in observed_episodes:
                    raise ValueError(f"{path}:{line}: action episode is absent from episode CSV: {key}")
                action_key = (*key, row["step"])
                if action_key in seen:
                    raise ValueError(f"Duplicate callback action: {action_key}")
                seen.add(action_key)
                row["latency_ms"] = float(row["latency_ms"])
                if not math.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
                    raise ValueError(f"{path}:{line}: invalid latency_ms")
                rows.append({field: row[field] for field in required})
    if not rows:
        raise ValueError("Actions input contains no callback timings")
    return rows


def bootstrap_pooled_quantiles(values: list[np.ndarray], samples: int,
                              rng: np.random.Generator) -> np.ndarray:
    """Exact pooled linear quantiles after resampling entire episode clusters.

    Resampling weights implement repeated whole episodes without expanding the
    underlying latency records. This is identical to np.quantile on an expanded
    episode bootstrap, including interpolation between adjacent observations.
    """
    pooled = np.concatenate(values)
    cluster = np.concatenate([np.full(len(v), index, dtype=int) for index, v in enumerate(values)])
    order = np.argsort(pooled)
    pooled, cluster = pooled[order], cluster[order]
    output = np.empty((samples, 2))
    n = len(values)
    for begin in range(0, samples, 100):
        end = min(samples, begin + 100)
        draws = rng.integers(0, n, size=(end - begin, n))
        counts = np.zeros((len(draws), n), dtype=np.int32)
        for index, draw in enumerate(draws):
            counts[index] = np.bincount(draw, minlength=n)
        cumulative = np.cumsum(counts[:, cluster], axis=1)
        totals = cumulative[:, -1]
        for column, quantile in enumerate((0.5, 0.95)):
            rank = (totals - 1) * quantile
            floor, ceil = np.floor(rank).astype(int), np.ceil(rank).astype(int)
            lower = np.argmax(cumulative > floor[:, None], axis=1)
            upper = np.argmax(cumulative > ceil[:, None], axis=1)
            output[begin:end, column] = pooled[lower] + (rank - floor) * (pooled[upper] - pooled[lower])
    return output


def summarize_actions(rows: list[dict[str, Any]], episodes: list[dict[str, Any]],
                      samples: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["config"], row["opponents"]), []).append(row)
    output = []
    for (config, opponents), group in sorted(groups.items(), key=lambda item: (opponent_sort_key(item[0][1]), item[0][0])):
        by_seed: dict[str, list[float]] = {}
        for row in group:
            by_seed.setdefault(row["seed"], []).append(row["latency_ms"])
        # All timed callbacks are expected, not an arbitrary latency sample.
        group_episodes = [r for r in episodes if r["config"] == config and r["opponents"] == opponents]
        for episode in group_episodes:
            count = len(by_seed.get(episode["seed"], []))
            if "action_count" in episode and count != int(episode["action_count"]):
                raise ValueError(f"Action timing count {count} differs from action_count "
                                 f"{episode['action_count']} in {(config, opponents, episode['seed'])}")
        clusters = [np.asarray(by_seed[s]) for s in sorted(by_seed)]
        values = np.concatenate(clusters)
        draws = bootstrap_pooled_quantiles(clusters, samples, rng_for(seed, f"latency:{config}:{opponents}"))
        output.append({
            "config": config, "label": label(config), "opponents": opponents,
            "timed_actions": len(values), "episodes_with_actions": len(clusters),
            "bootstrap_samples": samples, "interval_method": "episode cluster percentile bootstrap of pooled action quantiles",
            "latency_median_ms": {"value": float(np.quantile(values, 0.5)), "ci95": percentile_interval(draws[:, 0])},
            "latency_p95_ms": {"value": float(np.quantile(values, 0.95)), "ci95": percentile_interval(draws[:, 1])},
            "latency_max_ms": float(values.max()),
        })
    return output


def latency_figure(summaries: list[dict[str, Any]], figures: Path) -> list[str]:
    configs = sorted({s["config"] for s in summaries}, key=lambda c: (list(CONFIG_LABELS).index(c) if c in CONFIG_LABELS else 99, c))
    opponents = sorted({s["opponents"] for s in summaries}, key=opponent_sort_key)
    positions = {c: i for i, c in enumerate(configs)}
    offsets = np.linspace(-0.15, 0.15, len(opponents)) if len(opponents) > 1 else [0]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, max(3.6, 0.62 * len(configs) + 1.8)), layout="constrained")
    for axis_index, (axis, metric) in enumerate(zip(axes, ("latency_median_ms", "latency_p95_ms"))):
        for index, opponents_name in enumerate(opponents):
            group = [s for s in summaries if s["opponents"] == opponents_name]
            values = np.asarray([s[metric]["value"] for s in group])
            bounds = np.asarray([s[metric]["ci95"] for s in group])
            y = [positions[s["config"]] + offsets[index] for s in group]
            axis.errorbar(values, y, xerr=np.maximum(0, np.vstack((values - bounds[:, 0], bounds[:, 1] - values))),
                          fmt="o", capsize=3, color=OPPONENT_COLORS.get(opponents_name, "#29845e"),
                          label=OPPONENT_LABELS.get(opponents_name, opponents_name))
        axis.set_yticks(range(len(configs)), [label(c) for c in configs] if axis_index == 0 else [""] * len(configs))
        axis.invert_yaxis()
        axis.set_xscale("log")
        low, high = axis.get_xlim()
        shift = 5 if metric == "latency_median_ms" else 1
        powers = np.arange(math.floor(math.log10(low)) - 2, math.ceil(math.log10(high)) + 2)
        ticks = shift * 10.0 ** powers
        ticks = ticks[(ticks >= low / 1.5) & (ticks <= high * 1.5)]
        if len(ticks) >= 3:
            axis.set_xticks(ticks)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xlabel("Complete callback latency (ms)\nLogarithmic axis")
        axis.set_title("Pooled median" if axis_index == 0 else "Pooled 95th percentile")
        axis.xaxis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
    axes[-1].legend(loc="lower right", fontsize=11.5, frameon=False)
    fig.suptitle("Measured complete decision latency on local CPU", fontsize=15)
    fig.supxlabel("95% episode-cluster bootstrap intervals; observer instrumentation excluded", fontsize=12)
    return save_figure(fig, figures, "frozen_evaluation_latency")


def tex_escape(value: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
                    "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in value)


def tex_metric(summary: dict[str, Any], metric: str, percent: bool = False,
               decimals: int = 1, include_interval: bool = True) -> str:
    item = summary["metrics"].get(metric)
    if item is None:
        return "--"
    scale = 100 if percent else 1
    value, (low, high) = item["mean"] * scale, np.asarray(item["ci95"]) * scale
    if metric == "win_share" and item.get("boundary_degenerate"):
        return f"{value:.{decimals}f}" + r"$^{\dagger}$"
    if not include_interval:
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


def write_table(summaries: list[dict[str, Any]], out: Path) -> None:
    zeros = [s for s in summaries if s["metrics"].get("win_share", {}).get("boundary_degenerate")]
    boundary_caption = ""
    if zeros:
        limits = sorted({(s["episodes"], 100 * s["metrics"]["score_leader_presence"]["ci95"][1]) for s in zeros})
        descriptions = "; ".join(f"zero score-leading episodes among {n} games gives a Wilson upper bound of {upper:.1f}\\%"
                                 for n, upper in limits)
        boundary_caption = (r" $^{\dagger}$The degenerate all-zero bootstrap interval is suppressed; "
                            + descriptions + " on leader presence, which also bounds expected fractional share.")
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabularx}{\textwidth}{@{}Yrrrrrrr@{}}", r"\toprule",
             r"Configuration & Games & Score [95\% CI] & Coins & Kills & \shortstack{Self-\\destruction (\%)} & \shortstack{Survived\\(\%)} & \shortstack{Leader\\share (\%)} \\", r"\midrule"]
    for opponents, group_iter in itertools.groupby(summaries, key=lambda s: s["opponents"]):
        lines.append(r"\multicolumn{8}{@{}l}{\textit{" + tex_escape(OPPONENT_LABELS.get(opponents, opponents)) + r"}} \\")
        for summary in group_iter:
            lines.append(f"{tex_escape(summary['label'])} & {summary['episodes']} & "
                         f"{tex_metric(summary, 'score', decimals=2)} & "
                         f"{tex_metric(summary, 'coins', decimals=3, include_interval=False)} & "
                         f"{tex_metric(summary, 'kills', decimals=3, include_interval=False)} & "
                         f"{tex_metric(summary, 'suicides', True, include_interval=False)} & "
                         f"{tex_metric(summary, 'survived', True, include_interval=False)} & "
                         f"{tex_metric(summary, 'win_share', True, include_interval=False)}" + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabularx}",
                  r"\caption{Episode means from 200 games per row; score brackets are 95\% episode-bootstrap intervals. Other columns are point estimates, with intervals retained in the supplementary JSON. Score-leader share splits credit among tied highest final scores, including dead agents. Inference is conditional on one checkpoint and the stated opponents."
                  + boundary_caption + "}",
                  r"\label{tab:jesper-frozen-evaluation}", r"\end{table}"])
    (out / "frozen_evaluation_table.tex").write_text("\n".join(lines) + "\n")


def save_figure(fig: plt.Figure, figures: Path, basename: str) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = figures / f"{basename}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def performance_figure(summaries: list[dict[str, Any]], figures: Path) -> list[str]:
    opponents = sorted({s["opponents"] for s in summaries}, key=opponent_sort_key)
    configs = sorted({s["config"] for s in summaries}, key=lambda c: (list(CONFIG_LABELS).index(c) if c in CONFIG_LABELS else 99, c))
    available = [m for m in ("score", "win_share") if any(m in s["metrics"] for s in summaries)]
    fig, axes = plt.subplots(1, len(available), figsize=(10.4, max(3.6, 0.62 * len(configs) + 1.8)),
                             squeeze=False, layout="constrained")
    colors = ["#246b8e", "#bd642b", "#29845e", "#805f98"]
    offsets = np.linspace(-0.15, 0.15, len(opponents)) if len(opponents) > 1 else [0]
    positions = {c: i for i, c in enumerate(configs)}
    for axis_index, (axis, metric) in enumerate(zip(axes[0], available)):
        scale = 100 if metric == "win_share" else 1
        for index, opponent in enumerate(opponents):
            group = [s for s in summaries if s["opponents"] == opponent and metric in s["metrics"]]
            y = np.asarray([positions[s["config"]] + offsets[index] for s in group])
            x = np.asarray([s["metrics"][metric]["mean"] * scale for s in group])
            bounds = np.asarray([s["metrics"][metric]["ci95"] for s in group]) * scale
            intervals = np.maximum(0, np.vstack((x - bounds[:, 0], bounds[:, 1] - x)))
            boundary = np.asarray([metric == "win_share" and s["metrics"][metric].get("boundary_degenerate", False) for s in group])
            color = OPPONENT_COLORS.get(opponent, colors[index % len(colors)])
            axis.errorbar(x[~boundary], y[~boundary], xerr=intervals[:, ~boundary],
                          fmt="o", capsize=3, markersize=5, color=color,
                          label=OPPONENT_LABELS.get(opponent, opponent))
            if boundary.any():
                axis.plot(x[boundary], y[boundary], "o", markersize=5, color=color)
        axis.set_yticks(range(len(configs)), [label(c) for c in configs] if axis_index == 0 else [""] * len(configs))
        axis.invert_yaxis()
        axis.set_xlabel("Official score per episode" if metric == "score" else "Score-leader share (%)")
        axis.set_title("Competitive return" if metric == "score" else "Relative rank")
        axis.xaxis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        if metric == "win_share":
            upper = max(s["metrics"][metric]["ci95"][1] * 100 for s in summaries)
            axis.set_xlim(-0.5, min(101, max(5, upper * 1.2)))
    axes[0, -1].legend(loc="lower right", fontsize=11.5, frameon=False)
    fig.suptitle("Frozen-policy performance on measured episodes", fontsize=15)
    subtitle = "95% intervals over evaluation episodes; fixed checkpoints, no training-seed replication"
    if any(s["metrics"].get("win_share", {}).get("boundary_degenerate") for s in summaries):
        subtitle += "\nAll-zero leader-share bootstrap intervals suppressed; Wilson bounds are reported in the table"
    fig.supxlabel(subtitle, fontsize=12)
    return save_figure(fig, figures, "frozen_evaluation_performance")


def components_figure(results: list[dict[str, Any]], figures: Path) -> list[str]:
    if not results:
        return []
    available = [m for m in ("score", "survived") if any(m in r["metrics"] for r in results)]
    fig, axes = plt.subplots(1, len(available), figsize=(10.4, max(3.7, len(results) * 0.53 + 1.6)),
                             squeeze=False, layout="constrained")
    labels = [f"{r['name']}\n{OPPONENT_LABELS.get(r['opponents'], r['opponents'])}, {r['paired_episodes']} paired games" for r in results]
    for axis_index, (axis, metric) in enumerate(zip(axes[0], available)):
        scale = 100 if metric == "survived" else 1
        for index, result in enumerate(results):
            if metric not in result["metrics"]:
                continue
            value = result["metrics"][metric]["mean_difference"] * scale
            low, high = np.asarray(result["metrics"][metric]["ci95"]) * scale
            color = OPPONENT_COLORS.get(result["opponents"], "#29845e")
            axis.errorbar(value, index, xerr=[[max(0, value - low)], [max(0, high - value)]],
                          fmt="o", capsize=3, color=color)
        axis.axvline(0, color="#737b83", lw=1)
        axis.set_yticks(range(len(results)), labels if axis_index == 0 else [""] * len(results))
        axis.invert_yaxis()
        axis.xaxis.grid(True, alpha=0.2)
        axis.set_axisbelow(True)
        axis.set_xlabel("Difference in score" if metric == "score" else "Difference in survival\n(percentage points)")
        axis.set_title("Change in competitive return" if metric == "score" else "Change in survival")
    fig.suptitle("Controlled inference-time comparisons", fontsize=15)
    fig.supxlabel("Mask: temporal minus RUEHL; ranking: Final Q minus random\n95% paired episode-bootstrap intervals", fontsize=12)
    return save_figure(fig, figures, "frozen_evaluation_components")


def analyze(input_path: Path, out: Path, figures: Path, manifest: Path | None,
            samples: int, seed: int, actions_input: Path | None = None,
            latency_samples: int = 2000) -> dict[str, Any]:
    paths = input_paths(input_path)
    rows = read_rows(paths)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["config"], row["opponents"]), []).append(row)
    # Pair eligibility is checked for every observed pair, not just displayed contrasts.
    for opponents in sorted({r["opponents"] for r in rows}):
        observed = [g for (c, o), g in groups.items() if o == opponents]
        for left, right in itertools.combinations(observed, 2):
            paired_rows(left, right)
    summaries = [summarize_group(groups[key], samples, seed)
                 for key in sorted(groups, key=lambda k: (opponent_sort_key(k[1]), k[0]))]
    differences, omitted = contrasts(groups, samples, seed)
    result: dict[str, Any] = {
        "source_files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in paths],
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "bootstrap_seed": seed, "bootstrap_samples": samples, "confidence_level": 0.95,
        "episode_count": len(rows), "summaries": summaries,
        "paired_contrasts": differences, "omitted_contrasts": omitted,
        "factorial_interactions": factorial_interactions(groups, samples, seed),
        "limitations": [
            "Intervals describe evaluation-game variability conditional on these fixed checkpoints, not training-seed variability.",
            "Pairing matches initial layouts and game seeds, not identical trajectories or adversary actions after policies diverge.",
            "Displayed contrast intervals are pointwise and are not corrected for multiple comparisons.",
            "Episode-table latency summaries are means of per-episode quantiles; action_latency_summaries and the figure use pooled action quantiles with episode-cluster intervals.",
            "A zero observed event count is not evidence that its probability is zero; Wilson intervals are used for binary survival and suicide indicators.",
        ],
        "diagnostic_semantics": {
            "trapped_threat_bombs": "Model-predicted traps among dropped bombs with an opponent in the blast; not observed kills.",
            "audited_deaths": "Sampled recoverable action opportunities followed by focal death within six engine steps; multiple opportunities can precede one death.",
            "unsafe_selected_deaths": "Unsafe selections among audited_deaths; a descriptive fraction, not a causal attribution or death probability.",
        },
    }
    if manifest is not None:
        result["manifest"] = {"path": str(manifest.resolve()), "sha256": sha256_file(manifest),
                              "content": json.loads(manifest.read_text())}
    out.mkdir(parents=True, exist_ok=True)
    write_table(summaries, out)
    result["figures"] = performance_figure(summaries, figures) + components_figure(differences, figures)
    if actions_input is not None:
        actions_paths = action_input_paths(actions_input)
        actions = read_actions(actions_paths, rows)
        result["action_source_files"] = [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in actions_paths]
        result["action_latency_summaries"] = summarize_actions(actions, rows, latency_samples, seed)
        result["latency_measurement"] = (
            "perf_counter around each complete callback; learned callbacks include features, mask and neural inference, while random callbacks include mask and random selection only; "
            "observer instrumentation excluded. One PyTorch CPU thread per evaluator process; concurrently running "
            "evaluator processes do not imply exclusive CPU-core access. This is current local evaluation hardware, "
            "not reconstructed historical training hardware."
        )
        result["figures"].extend(latency_figure(result["action_latency_summaries"], figures))
    (out / "frozen_evaluation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Combined episode CSV or directory of configuration CSVs")
    parser.add_argument("--out", type=Path, required=True, help="Directory for summary JSON and LaTeX table")
    parser.add_argument("--figures-dir", type=Path, default=HERE.parent / "figures")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--actions-input", type=Path, help="All-callback *_actions.csv file or directory")
    parser.add_argument("--latency-bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=240927)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.latency_bootstrap_samples < 100:
        parser.error("--latency-bootstrap-samples must be at least 100")
    set_plot_style()
    try:
        result = analyze(args.input, args.out, args.figures_dir, args.manifest,
                         args.bootstrap_samples, args.bootstrap_seed, args.actions_input,
                         args.latency_bootstrap_samples)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Analysis error: {error}\n")
    print(f"Analyzed {result['episode_count']} episodes across {len(result['summaries'])} groups; "
          f"{len(result['paired_contrasts'])} paired contrasts. Outputs: {args.out}")


if __name__ == "__main__":
    main()
