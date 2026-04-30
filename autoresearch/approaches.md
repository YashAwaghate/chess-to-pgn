# Autoresearch Approaches

Local notes from the medium sweep. Keep this file as the memory of what worked
and what should not be repeated blindly.

## Working Approaches

### Complete-flow corner source: `auto_prefer`

Use ML corner detector outputs from `data/auto_corners.json`, falling back to
official/manual corners only where auto corners are missing.

Observed 20-game test split:

- Current/manual fallback path: `64.7149%` per-square, `8.7%` exact in the older eval path.
- `auto`: `97.6250%` per-square.
- `auto_prefer`: `97.8107%` per-square at TTA1.
- `auto_prefer`: `97.8555%` per-square and `51.8553%` exact at TTA4.

Conclusion: all complete-flow evals should default to `auto_prefer`.

### Legal-state smoothing / Bayesian board projection

Added experimental `state_exact_pct` metric in `scripts/eval_temporal_tracker.py`.
It evaluates board exactness after projecting the frame sequence through legal
Bayesian move choices.

Observed 20-game test split:

- Raw exact board, `auto_prefer`, TTA1: `50.4462%`
- Raw exact board, `auto_prefer`, TTA4: `51.8553%`
- Legal-state exact, `auto_prefer`, TTA1: `71.4420%`
- Legal-state exact, `auto_prefer`, TTA4: `70.3147%`

Conclusion: legal-state smoothing is the strongest exact-board improvement so
far. It should be turned from metric-only into an actual pipeline output mode.

### TTA4 for raw exact boards

Observed 20-game test split:

- TTA1 exact: `50.4462%`
- TTA4 exact: `51.8553%`

Conclusion: TTA4 helps raw exact-board accuracy, but it is slower and did not
beat legal-state smoothing.

## Do Not Retry Blindly

### GT-FEN fusion in this eval path

Observed:

- `exact20_auto_prefer_fusion_tta1`: `50.4462%`, same as no fusion.
- Earlier per-square fusion runs also tied baseline.

Reason: ChessReD test frames in this script mostly behave like one frame per
board state, so contiguous same-FEN segments offer little or no averaging.

### Auto-only corners

Observed:

- `auto_prefer` TTA4 exact: `51.8553%`
- `auto` only TTA4 exact: `49.6983%`

Reason: `auto_corners.json` does not cover all test frames; fallback frames are
useful.

### TTA7 full 20-game exact run with default timeout

Observed:

- `exact20_auto_prefer_tta7` timed out at 900 seconds.

Reason: too slow for the current medium loop. Retry only with a longer timeout
or on a smaller diagnostic subset.

### TTA4 for legal-state exact

Observed:

- TTA1 legal-state exact: `71.4420%`
- TTA4 legal-state exact: `70.3147%`

Reason: heavier TTA improves raw board exactness slightly, but appears to hurt
the legal-state sequence metric on this split.

## Promising Next Experiments

- Implement legal-state projection as a real decoder output and PGN/FEN stream,
  not just an eval metric.
- Add piece-count constrained repair for per-frame argmax boards.
- Add a no-change score/margin so the legal-state decoder can carry forward a
  board unless a candidate move is sufficiently better.
- Add crop `--pad` sweep in `scripts/eval_temporal_tracker.py` for bad games
  `4`, `21`, `75`, and `97`.
- Add corner quality scores: heatmap peak/entropy, polygon sanity, and
  downstream classifier confidence after warp.
