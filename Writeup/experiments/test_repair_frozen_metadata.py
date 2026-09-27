"""Guards around recovery; temporary fixtures never become report evidence."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("metadata_repair", Path(__file__).with_name("repair_frozen_metadata.py"))
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class MetadataRepairTests(unittest.TestCase):
    def test_edited_archive_reconstructs_historical_executed_hash(self):
        root = Path(__file__).resolve().parents[2]
        template = json.loads((root / "Writeup/experiments/results/frozen/final_native_strong.json").read_text())
        archive = root / "Writeup/experiments/archives/evaluate_frozen_policy_executed_v1.py"
        evaluator = root / "Writeup/experiments/evaluate_frozen_policy.py"
        provenance = repair.verify_source_provenance(template, archive, evaluator)
        self.assertEqual(provenance["executed_evaluator_sha256"],
                         template["source_sha256"][repair.EVALUATOR_KEY])
        self.assertNotEqual(provenance["archive_documentation_edited_sha256"],
                            provenance["executed_evaluator_sha256"])
        self.assertEqual(provenance["current_serializer_fixed_evaluator_sha256"], repair.digest(evaluator))

    def test_archive_change_cannot_pass_historical_hash_check(self):
        root = Path(__file__).resolve().parents[2]
        template = json.loads((root / "Writeup/experiments/results/frozen/final_native_strong.json").read_text())
        archive = root / "Writeup/experiments/archives/evaluate_frozen_policy_executed_v1.py"
        evaluator = root / "Writeup/experiments/evaluate_frozen_policy.py"
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / archive.name
            altered.write_text(archive.read_text().replace("parents[2]", "parents[1]", 1))
            with self.assertRaisesRegex(ValueError, "cannot reconstruct the executed evaluator"):
                repair.verify_source_provenance(template, altered, evaluator)

    def test_partial_campaign_cannot_be_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.csv"
            episodes.write_text("episode,action_count\n0,1\n")
            actions = root / "actions.csv"
            actions.write_text("episode\n0\n")
            with self.assertRaisesRegex(ValueError, "Refusing incomplete group"):
                repair.completed_rows(episodes, actions, 200, 0)

    def test_completed_episode_with_missing_action_records_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes = root / "episodes.csv"
            episodes.write_text("episode,action_count\n0,2\n")
            actions = root / "actions.csv"
            actions.write_text("episode\n0\n")
            with self.assertRaisesRegex(ValueError, "action CSV still incomplete"):
                repair.completed_rows(episodes, actions, 1, 0)

    def test_replay_latency_can_change_but_official_outcome_cannot(self):
        original = dict(config="final_native", opponents="strong", seed="10", layout_sha256="abc",
                        episode="7", score="4", latency_p95_ms="1.2")
        replay = original | dict(episode="0", latency_p95_ms="1.7")
        self.assertIn("score", repair.compare_episode(original, replay))
        with self.assertRaisesRegex(ValueError, "Replay outcome mismatch in score"):
            repair.compare_episode(original, replay | dict(score="3"))

    def test_recovery_cannot_overwrite_an_existing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "final_native_strong.json"
            manifest.write_text('{"preserve": true}\n')
            with self.assertRaises(FileExistsError):
                repair.repair("final_native", "strong", root, 200, Path("unused"),
                              Path("unused"), "unused", root / "temporary", 0)
            self.assertEqual(manifest.read_text(), '{"preserve": true}\n')


if __name__ == "__main__":
    unittest.main()
