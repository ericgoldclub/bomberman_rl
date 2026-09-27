# Reproducing the report experiments

These scripts reproduce the report's diagnostic and frozen-policy experiments.
They do not modify agent code, weights, hyperparameters, or training state.
The mask suite is generated rather than sampled from learned-policy trajectories.
The separate frozen-policy campaign uses the actual engine for 1,600 games.
Experiment JSONs retain the source hashes used for measurement. Current source
hashes and later docstring edits are recorded in `results/report_artifact_manifest.json`.

From the repository root, create an environment with Python 3.12 and install:

```sh
python3.12 -m venv /tmp/bomberman-report-py312
/tmp/bomberman-report-py312/bin/python -m pip install -r Writeup/experiments/requirements.txt
/tmp/bomberman-report-py312/bin/python Writeup/experiments/mask_audit.py
/tmp/bomberman-report-py312/bin/python Writeup/experiments/plot_mask_audit.py
```

The checked run used Python 3.12.14 on an Apple M4, NumPy 2.2.6 and PyTorch
2.8.0. `mask_audit.json` records all measured-source SHA-256 hashes and Git
revision `8138269` (the full revision is in the JSON). Counts are deterministic
for the recorded generator and versions; microsecond timings are not.

The audit directly imports both sets of callbacks, uses actual engine methods
for hazard/terrain evolution, and computes existential reachability separately
from either agent's safety helper. It checks bomb timing, lingering danger,
crate opening, blast propagation behind crates, waiting on an own bomb, newly
placed bombs, and an initially active explosion. It also replays 265 witness
paths through the engine's actual action method in the default 3,000-state run.

Raw rows are retained for every candidate action for both masks, including
states with no surviving continuation. Summary error fractions are restricted
to recoverable states, avoiding penalties for unavoidable death. Useful-bomb
constraints are not scored as erroneous rejections of movement. Per-density
counts and mask-only latency measurements are in the JSON. The suite does
not include initial active explosions in its random states, moving opponents,
or future new bombs. It is not an estimate of tournament death rates.

The figures and LaTeX table are derived from the JSON, without fitted or
invented learning curves. The illustrative state is `0.3:21` from the default
seed, and the final-mask fallback counterexample is `0.75:428`.

For a fast code smoke check without replacing the report measurements:

```sh
/tmp/bomberman-report-py312/bin/python Writeup/experiments/mask_audit.py --per-density 30 --out /tmp/bomberman-mask-smoke
```

The existing relevant tests can be run with:

```sh
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy OMP_NUM_THREADS=1 /tmp/bomberman-report-py312/bin/python -m unittest tests.test_ruehl_based_agent tests.test_ultra_network_agent -v
```

All 17 of these tests passed. The Ultra callback's safety implementation is
the same as the final agent's; the audit nevertheless imports and tests the
final callbacks directly. Unrestricted test discovery additionally tries
`test_dqn_agent_v3_killer.py`, which imports the absent
`agent_code.dqn_agent_v3` module and fails during collection. This is an
existing repository issue, not a diagnostic result.

## Frozen-policy campaign

The four configurations cross the same frozen final-agent Q-values or
uniform random selection with the temporal or RUEHL mask. Both learned
configurations retain final-agent features and weights: `final_ruehl_mask`
does **not** mean the trained RUEHL agent.
The primary checkpoint was the existing callbacks' default v2 artifact,
selected before campaign outcomes; its full SHA-256 is
`c268c3fcb5df349117f3a975dd9e158df7439dc25cd6c0be8f4d3fae80c9471e`.

Run from the repository root into a fresh output directory; the evaluator
refuses to overwrite raw CSVs. The following complete reproduction runs
sequentially (the original campaign ran concurrent independent processes):

```sh
for config in final_native final_ruehl_mask random_final_mask random_ruehl_mask; do
  for opponents in strong mixed; do
    CUDA_VISIBLE_DEVICES="" /tmp/bomberman-report-py312/bin/python Writeup/experiments/evaluate_frozen_policy.py --config "$config" --opponents "$opponents" --episodes 200 --out /tmp/bomberman-frozen-reproduction
  done
done
/tmp/bomberman-report-py312/bin/python Writeup/experiments/analyze_frozen_evaluation.py --input Writeup/experiments/results/frozen --actions-input Writeup/experiments/results/frozen --out Writeup/experiments/results --manifest Writeup/experiments/results/evaluation_manifest.json
```

The last command regenerates statistics/figures from the retained campaign.
For a newly reproduced campaign, change both input paths and use a separate
output/figures directory; omit the old manifest rather than attributing new
timings to it. Seeds 270920260–270920459 reset layouts every episode, followed
by a separate action-order generator and isolated per-agent Python/NumPy
streams. All four configurations have identical initial-layout hashes within
each opponent group. No model updates occur; strict tensor checks and a
warning-to-error logger prohibit silent inference fallback. Episodic focal
position history is cleared explicitly.

Per-episode and all-callback CSVs contain **1,600 games and 266,007 timed
callbacks**. Official score equals coins+5×credited kills in every episode.
There are zero timeout events and zero timeout skips. `executed_action` in
the callback CSV denotes the action passed to the engine after timeout
handling; a subsequent physical collision can still make it invalid.
`win_share` is fractional credit for the highest final score, including dead
agents; it is not a last-survivor statistic. `audited_deaths` counts sampled
recoverable action opportunities followed by death within six steps, not
distinct deaths. Predicted trap counts are predictions, not credited kills.
Callback timing excludes reference instrumentation and includes features,
masking and neural inference. The Apple M4 measurements do not reproduce
the tournament AMD hardware or historical training resources.

The analyzer bootstraps whole episodes, paired configurations together,
and reports the 2×2 interaction. Binary survival/self-destruction use Wilson
intervals; fractional score-leader share uses episode bootstrap intervals.
All-zero share groups have no displayed degenerate bootstrap bar: their
binary nonzero-share Wilson upper bound is about 1.9%. Pooled callback
median/p95 intervals resample whole episode clusters using 2,000 draws;
score intervals use 10,000 draws. These are pointwise intervals over game
seeds, conditional on one trained artifact, not training-seed uncertainty.

### Metadata export repair

The executed v1 evaluator saved every CSV row, but four JSON exports failed on
NumPy scalars. The current evaluator changes execution logic only for JSON
conversion. Both stored evaluators have shorter docstrings; the recovery verifier
reconstructs their historical source in memory and checks its SHA-256.
`evaluation_manifest.json` records historical and current hashes separately.

Four deterministic single-episode replays recovered the counterexamples and
matched the recorded actions and official outcomes. Campaign CSVs and performance
results are unchanged. Group JSONs retain recovery details; the global manifest
also explains two exports whose on-disk source hash changed after their code loaded.

### Competitive failure trace

The first native strong episode is replayed with every action audited:

```sh
/tmp/bomberman-report-py312/bin/python Writeup/experiments/trace_native_episode.py --out /tmp/bomberman-native-trace-reproduction
/tmp/bomberman-report-py312/bin/python Writeup/experiments/plot_native_gameplay_failure.py
```

`native_trace_capture` retains all 216 observed states, primary replay
verification and figure provenance. An enemy executes first at step215,
blocks the requested RIGHT, and leaves the focal agent trapped at step216.
The trace explains a mechanism; it does not estimate its frequency or prove
that a conditional alternative would win. The optional
`plot_gameplay_failure.py` visualizes the first unsafe RUEHL-mask selection
from the repaired metadata; this second failure figure is outside the report.

### Relevant verification

```sh
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy OMP_NUM_THREADS=1 /tmp/bomberman-report-py312/bin/python -m unittest tests.test_ruehl_based_agent tests.test_ultra_network_agent Writeup/experiments/test_analyze_frozen_evaluation.py Writeup/experiments/test_evaluate_frozen_policy.py Writeup/experiments/test_repair_frozen_metadata.py -v
```

All **45 relevant tests passed**, including RNG isolation, actual engine
timing/timeout rules, exact deterministic replay, layout pairing, boundary
intervals, bootstrap invariants and metadata-recovery guards. Core agent
source and checkpoint bytes remain unchanged.

## Building the complete report

Run from the repository root:

```sh
python3 Writeup/experiments/build_final_report.py
```

The builder applies `../scientific_corrections.patch` in a temporary directory,
preserving the original `main.tex`. It assembles all seven chapters and uses the
verified author assignments in `results/report_author_provenance.json`.
`results/integration_manifest.json` records the source and assembly hashes.

With LaTeX and BibTeX installed, run these commands from `Writeup`:

```sh
mkdir -p ../output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf final_report.tex
(cd ../output/pdf && BIBINPUTS=../../Writeup: bibtex final_report)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf final_report.tex
```

The combined PDF is written to `output/pdf/final_report.pdf`.
`results/report_artifact_manifest.json` records its hash and the report sources.
`plot_verified_curriculum.py` regenerates the historical summary figure directly
from the two recorded Git snapshots.
