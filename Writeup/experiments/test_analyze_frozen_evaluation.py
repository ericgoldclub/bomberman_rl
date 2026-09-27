"""Small statistical-invariant tests; all fixtures are temporary synthetic data."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("analyze_frozen_evaluation.py")
SPEC = importlib.util.spec_from_file_location("frozen_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def row(config="final_native", seed="1", score=1, **extras):
    result = dict(config=config, opponents="strong", episode=seed, seed=seed,
                  layout_sha256="same-layout-" + seed, score=float(score))
    result.update(extras)
    return result


def write_csv(directory, rows):
    path = Path(directory) / "episodes.csv"
    fields = sorted({key for r in rows for key in r})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


class FrozenAnalysisTests(unittest.TestCase):
    def test_zero_events_wilson_upper_limit_is_positive(self):
        low, high = analysis.wilson_interval(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0.03)
        self.assertLess(high, 0.04)

    def test_all_zero_win_share_has_nonzero_score_leader_upper_bound(self):
        rows = [row(seed=str(i), win_share=0) for i in range(200)]
        summary = analysis.summarize_group(rows, 500, 3)
        share = summary["metrics"]["win_share"]
        leader = summary["metrics"]["score_leader_presence"]
        self.assertEqual(share["ci95"], [0, 0])  # Preserve the raw descriptive bootstrap.
        self.assertTrue(share["boundary_degenerate"])
        self.assertGreater(leader["ci95"][1], 0.018)
        self.assertLess(leader["ci95"][1], 0.02)
        self.assertNotIn("[0.0, 0.0]", analysis.tex_metric(summary, "win_share", True))
        with tempfile.TemporaryDirectory() as directory:
            analysis.write_table([summary], Path(directory))
            table = (Path(directory) / "frozen_evaluation_table.tex").read_text()
            self.assertIn("Wilson upper bound of 1.9", table)
            self.assertIn("degenerate all-zero bootstrap interval is suppressed", table)

    def test_pairing_preserves_constant_effect_under_large_game_variation(self):
        left = [row(seed=str(i), score=value + 5) for i, value in enumerate([0, 100, 3, 44])]
        right = [row("final_ruehl_mask", str(i), value) for i, value in enumerate([0, 100, 3, 44])]
        groups = {("final_native", "strong"): left, ("final_ruehl_mask", "strong"): right}
        result, omitted = analysis.contrasts(groups, 1000, 12)
        self.assertEqual(len(result), 1)
        self.assertEqual(omitted, [])
        self.assertEqual(result[0]["metrics"]["score"]["mean_difference"], 5)
        self.assertEqual(result[0]["metrics"]["score"]["ci95"], [5, 5])

    def test_mismatched_layout_cannot_be_called_paired(self):
        left, right = [row()], [row("final_ruehl_mask")]
        right[0]["layout_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Paired-layout mismatch"):
            analysis.paired_rows(left, right)

    def test_factorial_interaction_removes_additive_game_effect(self):
        effects = {"final_native": 15, "final_ruehl_mask": 10,
                   "random_final_mask": 4, "random_ruehl_mask": 2}
        groups = {(config, "strong"): [row(config, str(i), game + effect)
                                      for i, game in enumerate([0, 100, 3, 44])]
                  for config, effect in effects.items()}
        result = analysis.factorial_interactions(groups, 1000, 12)
        self.assertEqual(result[0]["metrics"]["score"]["mean_difference"], 3)
        self.assertEqual(result[0]["metrics"]["score"]["ci95"], [3, 3])

    def test_duplicate_seed_is_rejected_even_if_episode_labels_differ(self):
        duplicate = row(episode="2")
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, [row(), duplicate])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                analysis.read_rows([path])

    def test_optional_metric_cannot_silently_drop_missing_games(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, [row(survived=1), row(seed="2")])
            with self.assertRaisesRegex(ValueError, "present in only"):
                analysis.read_rows([path])

    def test_episode_reader_retains_all_metrics(self):
        source = row(**{metric: 1 for metric in analysis.METRICS})
        source.update(checked_actions=2, unsafe_selected=1)
        with tempfile.TemporaryDirectory() as directory:
            result = analysis.read_rows([write_csv(directory, [source])])[0]
            for metric in analysis.METRICS:
                self.assertEqual(result[metric], source[metric], metric)

    def test_action_rate_weights_actions_but_resamples_episodes(self):
        rows = [row(seed="1", unsafe_selected=1, checked_actions=2),
                row(seed="2", unsafe_selected=0, checked_actions=100)]
        summary = analysis.summarize_group(rows, 2000, 1)
        rate = summary["metrics"]["unsafe_selection_rate"]
        self.assertAlmostEqual(rate["mean"], 1 / 102)
        self.assertEqual(rate["ci95"], [0, 0.5])
        self.assertIn("cluster", rate["interval_method"])

    def test_bootstrap_reproducibility(self):
        values = np.array([[0, 1], [5, 4], [2, 8]])
        a = analysis.bootstrap_means(values, 500, analysis.rng_for(3, "same"))
        b = analysis.bootstrap_means(values, 500, analysis.rng_for(3, "same"))
        np.testing.assert_array_equal(a, b)

    def test_pooled_latency_bootstrap_matches_expanded_episode_resampling(self):
        clusters = [np.array([1.0, 5.0]), np.array([4.0, 10.0, 20.0])]
        actual = analysis.bootstrap_pooled_quantiles(clusters, 100, analysis.rng_for(5, "latency"))
        draws = analysis.rng_for(5, "latency").integers(0, 2, size=(100, 2))
        expected = np.asarray([np.quantile(np.concatenate([clusters[i] for i in draw]), [0.5, 0.95])
                               for draw in draws])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_action_files_excluded_from_episode_directory_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_path = write_csv(directory, [row()])
            (Path(directory) / "synthetic_actions.csv").write_text(episode_path.read_text())
            self.assertEqual(analysis.input_paths(Path(directory)), [episode_path])

    def test_missing_action_timings_rejected_by_count(self):
        actions = [dict(config="final_native", opponents="strong", seed="1", latency_ms=1.0)]
        with self.assertRaisesRegex(ValueError, "differs from action_count"):
            analysis.summarize_actions(actions, [row(action_count=2)], 100, 1)

    def test_action_reader_and_complete_output(self):
        episodes = [row(seed="1", action_count=2), row(seed="2", action_count=2)]
        actions = [dict(config="final_native", opponents="strong", episode=seed,
                        seed=seed, step=step, latency_ms=value)
                   for seed, step, value in [("1", 1, 1.3), ("1", 2, 2.4), ("2", 1, 3.5), ("2", 2, 4.6)]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_path = write_csv(root, episodes)
            (root / "actions").mkdir()
            actions_path = write_csv(root / "actions", actions)
            result = analysis.analyze(episode_path, root / "out", root / "figures", None,
                                      100, 7, actions_path, 100)
            latency = result["action_latency_summaries"][0]
            self.assertEqual(latency["timed_actions"], 4)
            self.assertAlmostEqual(latency["latency_median_ms"]["value"], 2.95)
            self.assertAlmostEqual(latency["latency_p95_ms"]["value"], 4.435)
            self.assertEqual(latency["latency_max_ms"], 4.6)
            self.assertEqual(len(result["figures"]), 4)

    def test_single_configuration_outputs_no_invented_comparisons(self):
        rows = [row(seed=str(i), score=i, survived=i % 2, win_share=0.25) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(directory, rows)
            result = analysis.analyze(path, Path(directory) / "out", Path(directory) / "figures", None, 500, 7)
            self.assertEqual(len(result["summaries"]), 1)
            self.assertEqual(result["paired_contrasts"], [])
            self.assertEqual(len(result["figures"]), 2)
            table = (Path(directory) / "out/frozen_evaluation_table.tex").read_text()
            self.assertIn("Final Q + temporal mask", table)
            self.assertNotIn("RUEHL Q", table)
            self.assertIn("25.0", table)  # Fractional win share stays a mean.
            self.assertIn("Score [95", table)
            self.assertTrue(any(line.endswith(r"\\") and "Final Q" in line for line in table.splitlines()))


if __name__ == "__main__":
    unittest.main()
