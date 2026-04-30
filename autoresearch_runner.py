#!/usr/bin/env python3
"""Small autoresearch harness for this chess-to-pgn repo.

The original karpathy/autoresearch loop compares fixed-time nanochat runs.
This project compares chess-pipeline experiments, so the stable metric is read
from the JSON emitted by the existing evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "autoresearch" / "runs"
RESULTS_PATH = ROOT / "autoresearch" / "results.tsv"
DEFAULT_GAMES = ["0", "33", "76"]


def _safe_tag(tag: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag.strip())
    return safe or time.strftime("%Y%m%d_%H%M%S")


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _ensure_results_file() -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        return
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "timestamp",
                "commit",
                "tag",
                "score",
                "metric",
                "status",
                "duration_s",
                "log_path",
                "description",
            ]
        )


def _default_command(args: argparse.Namespace, out_json: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/eval_temporal_tracker.py",
        "--games",
        *[str(g) for g in args.games],
        "--tta",
        str(args.tta),
        "--checkpoint",
        str(args.checkpoint),
        "--fusion",
        str(args.fusion),
        "--corner_source",
        str(args.corner_source),
        "--out",
        str(out_json),
    ]


def _read_score(out_json: Path, metric: str) -> float:
    with out_json.open(encoding="utf-8") as f:
        payload = json.load(f)
    totals = payload.get("totals", {})
    if metric not in totals:
        raise KeyError(f"Metric {metric!r} not present in {out_json}")
    return float(totals[metric])


def _append_result(row: dict[str, str | float]) -> None:
    _ensure_results_file()
    with RESULTS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "commit",
                "tag",
                "score",
                "metric",
                "status",
                "duration_s",
                "log_path",
                "description",
            ],
            delimiter="\t",
        )
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a chess-to-pgn autoresearch evaluation")
    parser.add_argument("--tag", default="baseline", help="Short run tag for logs/results")
    parser.add_argument("--description", default="", help="One-line experiment description")
    parser.add_argument("--metric", default="tr_correctness_pct", help="Metric key under JSON totals")
    parser.add_argument("--games", nargs="*", default=DEFAULT_GAMES, help="ChessReD game ids to evaluate")
    parser.add_argument("--tta", type=int, default=1, help="TTA views for faster autoresearch loops")
    parser.add_argument("--checkpoint", default="models/chess_piece_classifier_v2.pth", help="Classifier checkpoint")
    parser.add_argument("--fusion", default="none", choices=["none", "gt_fen"], help="Softmax fusion mode")
    parser.add_argument("--corner-source", dest="corner_source", default="auto_prefer",
                        choices=["current", "auto", "auto_prefer"], help="Corner source for pipeline eval")
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds before marking crash")
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help="Override command. Use {out_json} as placeholder for the JSON output path.",
    )
    args = parser.parse_args()

    tag = _safe_tag(args.tag)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RUN_DIR / f"{stamp}_{tag}.json"
    log_path = RUN_DIR / f"{stamp}_{tag}.log"

    if args.command:
        command = [part.format(out_json=str(out_json)) for part in args.command]
    else:
        command = _default_command(args, out_json)

    started = time.time()
    status = "crash"
    score = 0.0

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
            )
            if completed.returncode == 0:
                score = _read_score(out_json, args.metric)
                status = "ok"
            else:
                status = f"crash:{completed.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
        except Exception as exc:
            log.write(f"\n[autoresearch_runner] {type(exc).__name__}: {exc}\n")
            status = "crash"

    duration = time.time() - started
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git_short_hash(),
        "tag": tag,
        "score": f"{score:.6f}",
        "metric": args.metric,
        "status": status,
        "duration_s": f"{duration:.1f}",
        "log_path": str(log_path.relative_to(ROOT)),
        "description": args.description,
    }
    _append_result(row)

    print(f"status: {status}")
    print(f"{args.metric}: {score:.6f}")
    print(f"duration_s: {duration:.1f}")
    print(f"log: {log_path}")
    print(f"results: {RESULTS_PATH}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
