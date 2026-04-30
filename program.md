# autoresearch for chess-to-pgn

This repo is configured for an autoresearch-style loop inspired by
`karpathy/autoresearch`: make one focused research change, run a fixed
evaluation command, record the result, keep the change only if the metric
improves enough to justify the complexity.

## Setup

Current run branch: `autoresearch/apr30`.

Before a new run:

1. Read `AGENTS.md` and `graphify-out/GRAPH_REPORT.md`.
2. Read `results/evaluation_report.md` for the current research priorities.
3. Check `git status --short` and do not overwrite unrelated user changes.
4. Ensure `autoresearch/results.tsv` exists with a header row.

The current high-priority research targets from the evaluation report are:

- Multi-frame softmax fusion before decoding.
- Temporal tracker confirmation relaxed by legal-move-filtered tolerance.
- Full-board plus per-square classifier fusion.
- OOD/domain-randomized evaluation and training data.

## Fixed Evaluation

Use `autoresearch_runner.py` as the harness. It runs the selected command,
stores logs under `autoresearch/runs/`, parses the JSON metric, and appends a
tab-separated row to `autoresearch/results.tsv`.

Default baseline command:

```powershell
python autoresearch_runner.py --tag baseline --description "baseline temporal tracker eval"
```

The default score is `tr_correctness_pct` from `scripts/eval_temporal_tracker.py`.
Higher is better. For classifier-focused experiments, add a new fixed runner
mode before comparing results; do not hand-edit the score after a run.

## Experiment Loop

1. Inspect the current best score in `autoresearch/results.tsv`.
2. Make one small, reviewable code change.
3. Run:

```powershell
python autoresearch_runner.py --tag short_name --description "what changed"
```

4. If the run crashes, inspect the generated log and fix only simple mistakes.
5. If the score improves, keep the change and run focused tests.
6. If the score is worse or complexity is not justified, revert only your own
   experiment change.
7. After modifying code files, run `graphify update .`.

## Guardrails

- Do not modify data files, checkpoints, or generated reports as part of an
  experiment unless the experiment explicitly owns that artifact.
- Keep `autoresearch_runner.py` stable while comparing runs.
- Do not commit `autoresearch/results.tsv` or run logs; they are local lab
  notebook artifacts.
- Prefer focused tests such as `python -m pytest tests/test_move_detector.py`
  before broader suites.
