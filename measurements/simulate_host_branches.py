#!/usr/bin/env python3
"""
Simulate Engine A host branch routing before spending model time.

Fixture mode proves every label is reachable. Grid mode sweeps synthetic metric
packets to keep the decision tree inclusive.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Metrics:
    name: str
    baseline_ok: bool
    output_acceptable: bool
    calibration_monotonic: float
    high_bucket_accuracy: float
    fidelity: float
    exit_rate: float
    false_exit_rate: float
    boundary_recall: float
    avg_layers_skipped: float
    easy_exit_rate: float
    hard_exit_rate: float
    layer_tap_sane: bool
    mid_layer_best: bool
    top1_agreement: float
    topk_agreement: float
    short_context_fidelity: float
    long_context_fidelity: float
    wall_clock_delta: float


@dataclass
class Route:
    label: str
    branch: str
    next_change: str
    meaning: str


def route(m: Metrics) -> Route:
    if not m.baseline_ok:
        if m.output_acceptable:
            return Route("parser/eval failure", "Tier 0 baseline", "Fix parser.", "The model output is acceptable but evaluation rejected it.")
        return Route("baseline instability", "Tier 0 baseline", "Fix prompt/model before Engine A.", "Full-depth behavior is not stable enough to test skipping.")

    if not m.layer_tap_sane:
        return Route("layer tap mismatch", "Tier 1 harness", "Verify hidden-state index and final norm before lm_head.", "The intermediate readout is probably wired wrong.")

    if m.calibration_monotonic < 0.60:
        return Route("no calibration", "Tier 1 calibration", "Try later exit layer, logit gap, entropy, and norm fixes.", "Confidence does not predict full-depth agreement.")

    if m.exit_rate >= 0.20 and m.fidelity < 0.90:
        if m.high_bucket_accuracy < 0.85:
            return Route("false plateau", "Tier 1 calibration", "Add stability/margin guard.", "The intermediate layer is confidently wrong.")
        if m.boundary_recall < 0.75 or m.hard_exit_rate > 0.20:
            return Route("boundary blindness", "Tier 1 calibration", "Add entropy/moment/velocity hard-token detector.", "Boundary tokens are being treated as laminar.")
        return Route("threshold too loose", "Tier 1 threshold", "Raise threshold or add guard.", "The gate accepts too many unsafe tokens.")

    if m.fidelity >= 0.97 and m.exit_rate < 0.02:
        if m.easy_exit_rate < 0.10:
            return Route("laminar under-skip", "Tier 2 utility", "Add cluster-conditioned thresholds.", "Easy tokens are still not exiting.")
        return Route("threshold too strict", "Tier 2 utility", "Lower threshold or try later exit layer.", "The gate is safe but not useful.")

    if m.avg_layers_skipped < 1.0 and m.exit_rate >= 0.02:
        return Route("late-only signal", "Tier 3 layer sweep", "Measure if late exit still has enough speedup ceiling.", "The gate works only near the end of the stack.")

    if not m.mid_layer_best and m.fidelity < 0.90 and m.exit_rate < 0.05:
        return Route("layer signal absent", "Tier 3 layer sweep", "Try trained per-layer classifier.", "No tested layer has a useful early-exit signal.")

    if m.top1_agreement >= 0.95 and m.topk_agreement < 0.80:
        return Route("distribution drift", "Tier 4 distribution", "Add KL/top-k guard.", "Exact match passes but distribution changes too much.")

    if m.short_context_fidelity - m.long_context_fidelity > 0.08:
        return Route("context drift", "Tier 4 context", "Calibrate thresholds by context length.", "The gate shifts under long-context distribution.")

    if m.fidelity >= 0.95 and m.exit_rate >= 0.05 and m.wall_clock_delta <= 0.0:
        return Route("feature overhead", "Tier 5 systems", "Move feature extraction into C++ or drop expensive sensors.", "Logical skip works but wall-clock does not.")

    if m.fidelity >= 0.95 and m.exit_rate >= 0.05 and m.avg_layers_skipped < 2.0:
        return Route(
            "shallow signal",
            "Tier 3 layer sweep",
            "Try earlier exit layers or accept lower speedup ceiling.",
            "The gate is safe, but the expected depth saved is small.",
        )

    if m.fidelity >= 0.95 and m.exit_rate >= 0.05 and m.avg_layers_skipped >= 2.0:
        return Route("first-order signal", "Tier 2 utility", "Sweep thresholds/layers, then long-context drift.", "Engine A is skipping safely.")

    if m.fidelity < 0.95 and m.exit_rate >= 0.05:
        return Route(
            "partial unsafe signal",
            "Tier 2 utility",
            "Add guards or raise threshold, then rerun calibration.",
            "The gate has useful coverage, but fidelity is not deployable yet.",
        )

    if m.fidelity < 0.95 and m.exit_rate < 0.05:
        return Route("weak signal", "Tier 2 utility", "Try later layers or fused features.", "There is some structure, but not enough utility yet.")

    return Route("ambiguous", "manual review", "Inspect calibration buckets and examples.", "Metrics do not match a clean branch.")


def fixtures() -> List[Tuple[Metrics, str]]:
    base = dict(
        baseline_ok=True,
        output_acceptable=False,
        calibration_monotonic=0.90,
        high_bucket_accuracy=0.96,
        fidelity=0.96,
        exit_rate=0.10,
        false_exit_rate=0.002,
        boundary_recall=0.95,
        avg_layers_skipped=4.0,
        easy_exit_rate=0.30,
        hard_exit_rate=0.02,
        layer_tap_sane=True,
        mid_layer_best=True,
        top1_agreement=0.96,
        topk_agreement=0.92,
        short_context_fidelity=0.96,
        long_context_fidelity=0.95,
        wall_clock_delta=0.10,
    )
    cases = [
        (Metrics("parser", False, True, **{k: v for k, v in base.items() if k not in {"baseline_ok", "output_acceptable"}}), "parser/eval failure"),
        (Metrics("baseline", False, False, **{k: v for k, v in base.items() if k not in {"baseline_ok", "output_acceptable"}}), "baseline instability"),
        (Metrics("tap", **{**base, "layer_tap_sane": False}), "layer tap mismatch"),
        (Metrics("calibration", **{**base, "calibration_monotonic": 0.30}), "no calibration"),
        (Metrics("plateau", **{**base, "fidelity": 0.70, "exit_rate": 0.40, "high_bucket_accuracy": 0.70}), "false plateau"),
        (Metrics("boundary", **{**base, "fidelity": 0.85, "exit_rate": 0.30, "boundary_recall": 0.40, "hard_exit_rate": 0.50}), "boundary blindness"),
        (Metrics("loose", **{**base, "fidelity": 0.88, "exit_rate": 0.35}), "threshold too loose"),
        (Metrics("under_skip", **{**base, "fidelity": 0.99, "exit_rate": 0.01, "easy_exit_rate": 0.02}), "laminar under-skip"),
        (Metrics("strict", **{**base, "fidelity": 0.99, "exit_rate": 0.01, "easy_exit_rate": 0.30}), "threshold too strict"),
        (Metrics("late", **{**base, "avg_layers_skipped": 0.5, "exit_rate": 0.10}), "late-only signal"),
        (Metrics("absent", **{**base, "fidelity": 0.80, "exit_rate": 0.01, "mid_layer_best": False}), "layer signal absent"),
        (Metrics("dist", **{**base, "top1_agreement": 0.97, "topk_agreement": 0.60}), "distribution drift"),
        (Metrics("context", **{**base, "short_context_fidelity": 0.98, "long_context_fidelity": 0.86}), "context drift"),
        (Metrics("overhead", **{**base, "wall_clock_delta": -0.05}), "feature overhead"),
        (Metrics("shallow", **{**base, "avg_layers_skipped": 1.4}), "shallow signal"),
        (Metrics("success", **base), "first-order signal"),
        (Metrics("partial", **{**base, "fidelity": 0.92, "exit_rate": 0.10}), "partial unsafe signal"),
        (Metrics("weak", **{**base, "fidelity": 0.90, "exit_rate": 0.02, "avg_layers_skipped": 1.5}), "weak signal"),
    ]
    return cases


def run_fixtures() -> bool:
    print("Engine A branch fixtures")
    print("-" * 92)
    print(f"{'scenario':>20} {'expected':>24} {'actual':>24} {'ok':>4}")
    ok_all = True
    for metrics, expected in fixtures():
        actual = route(metrics)
        ok = actual.label == expected
        ok_all = ok_all and ok
        print(f"{metrics.name:>20} {expected:>24} {actual.label:>24} {str(ok):>4}")
        if not ok:
            print(f"  next={actual.next_change}")
    print()
    return ok_all


def run_grid(limit_print: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    interesting: List[Tuple[Metrics, Route]] = []
    total = 0
    for calibration, fidelity, exit_rate, boundary, topk, context_drop, wall in itertools.product(
        [0.40, 0.75, 0.95],
        [0.82, 0.92, 0.97, 0.99],
        [0.005, 0.03, 0.10, 0.35],
        [0.50, 0.85, 0.98],
        [0.65, 0.85, 0.95],
        [0.00, 0.04, 0.12],
        [-0.05, 0.08],
    ):
        metrics = Metrics(
            name=f"cal={calibration:.2f}:fid={fidelity:.2f}:exit={exit_rate:.3f}",
            baseline_ok=True,
            output_acceptable=False,
            calibration_monotonic=calibration,
            high_bucket_accuracy=max(0.50, fidelity - 0.03),
            fidelity=fidelity,
            exit_rate=exit_rate,
            false_exit_rate=max(0.0, exit_rate * (1.0 - fidelity)),
            boundary_recall=boundary,
            avg_layers_skipped=exit_rate * 16,
            easy_exit_rate=min(1.0, exit_rate * 4.0),
            hard_exit_rate=max(0.0, exit_rate * (1.0 - boundary)),
            layer_tap_sane=True,
            mid_layer_best=True,
            top1_agreement=max(fidelity, 0.95 if topk < 0.80 else fidelity),
            topk_agreement=topk,
            short_context_fidelity=fidelity,
            long_context_fidelity=max(0.0, fidelity - context_drop),
            wall_clock_delta=wall,
        )
        actual = route(metrics)
        counts[actual.label] = counts.get(actual.label, 0) + 1
        total += 1
        if actual.label != "first-order signal" and len(interesting) < limit_print:
            interesting.append((metrics, actual))

    print("Engine A branch grid")
    print("-" * 92)
    print(f"grid_runs={total}")
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"{label:>28}: {count}")
    print()
    if interesting:
        print("First non-success examples")
        print("-" * 92)
        for metrics, actual in interesting:
            print(
                f"{actual.label:>24} cal={metrics.calibration_monotonic:4.0%} "
                f"fid={metrics.fidelity:4.0%} exit={metrics.exit_rate:5.1%} "
                f"boundary={metrics.boundary_recall:4.0%} topk={metrics.topk_agreement:4.0%} "
                f"ctx_drop={metrics.short_context_fidelity - metrics.long_context_fidelity:+.2f} "
                f"wall={metrics.wall_clock_delta:+.2f}"
            )
            print(f"  next: {actual.next_change}")
        print()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fixtures", "grid", "all"], default="all")
    parser.add_argument("--limit_print", type=int, default=18)
    args = parser.parse_args()

    ok = True
    if args.mode in {"fixtures", "all"}:
        ok = run_fixtures() and ok
    if args.mode in {"grid", "all"}:
        counts = run_grid(args.limit_print)
        if counts.get("ambiguous", 0):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
