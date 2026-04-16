#!/usr/bin/env python3
"""
Simulate the cheapest Engine A signal budget.

This asks a practical systems question:

Which useful gate signals are free or nearly free once we already have an
intermediate readout, and does an Engine B-style memory-tension veto help even
when it is weakly correlated with Engine A confidence?
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

try:
    from .synthetic_engine_a import GateResult, TokenState, build_sequence, summarize
except ImportError:  # Allows direct execution: python measurements/free_signal_budget.py
    from synthetic_engine_a import GateResult, TokenState, build_sequence, summarize


@dataclass
class BudgetResult:
    name: str
    cost_units: float
    metrics: Dict[str, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / max(1, len(xs))
    mean_y = sum(ys) / max(1, len(ys))
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return numerator / ((denom_x * denom_y) ** 0.5)


def engine_b_tension(token: TokenState, rng: random.Random) -> float:
    """A weakly correlated memory ambiguity proxy.

    This is intentionally not just difficulty again. It mainly lights up on
    false plateaus and boundary-like ambiguity, with enough noise that the
    global correlation with Engine A confidence stays weak.
    """
    return clamp(
        0.32
        + 0.10 * float(token.boundary)
        + 0.50 * float(token.false_plateau)
        + rng.gauss(0.0, 0.25),
        0.0,
        1.0,
    )


def accept(mode: str, token: TokenState, tension: float) -> bool:
    if mode == "confidence":
        return token.mid_conf >= 0.90
    if mode == "free_sharpness":
        return token.mid_conf >= 0.86 and token.logit_gap >= 0.18 and token.entropy <= 2.40
    if mode == "free_plus_b_veto":
        return (
            token.mid_conf >= 0.82
            and token.logit_gap >= 0.12
            and token.entropy <= 2.60
            and tension <= 0.50
        )
    if mode == "stability_guard":
        return (
            token.mid_conf >= 0.86
            and token.logit_gap >= 0.18
            and token.entropy <= 2.40
            and token.stability >= 0.55
        )
    if mode == "a_b_stability":
        return (
            token.mid_conf >= 0.82
            and token.logit_gap >= 0.10
            and token.entropy <= 2.80
            and token.stability >= 0.50
            and tension <= 0.58
        )
    raise ValueError(f"unknown mode: {mode}")


def cost_units(mode: str) -> float:
    # Relative units, not wall-clock. Free means reused from intermediate logits.
    return {
        "confidence": 0.00,
        "free_sharpness": 0.05,
        "free_plus_b_veto": 0.05,
        "stability_guard": 0.20,
        "a_b_stability": 0.20,
    }[mode]


def run_mode(mode: str, seeds: int, tokens: int, exit_layer: int, total_layers: int) -> BudgetResult:
    rows = []
    for seed in range(seeds):
        rng = random.Random(seed + 10_000)
        results: List[GateResult] = []
        for token in build_sequence(seed, tokens):
            tension = engine_b_tension(token, rng)
            accepted = accept(mode, token, tension)
            correct = token.mid_token == token.final_token
            results.append(
                GateResult(
                    accepted=accepted,
                    correct=correct,
                    boundary=token.boundary,
                    false_plateau=token.false_plateau,
                    layers_used=exit_layer if accepted else total_layers,
                    confidence=token.mid_conf,
                )
            )
        rows.append(summarize(results, total_layers=total_layers, exit_layer=exit_layer))
    return BudgetResult(name=mode, cost_units=cost_units(mode), metrics=mean_dict(rows))


def mean_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(items)
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def correlation_probe(seeds: int, tokens: int) -> Dict[str, float]:
    confidences: List[float] = []
    tensions: List[float] = []
    difficulties: List[float] = []
    false_plateaus: List[float] = []
    for seed in range(seeds):
        rng = random.Random(seed + 10_000)
        for token in build_sequence(seed, tokens):
            confidences.append(token.mid_conf)
            difficulties.append(token.difficulty)
            false_plateaus.append(float(token.false_plateau))
            tensions.append(engine_b_tension(token, rng))
    return {
        "corr_tension_confidence": pearson(tensions, confidences),
        "corr_tension_difficulty": pearson(tensions, difficulties),
        "corr_tension_false_plateau": pearson(tensions, false_plateaus),
    }


def pct(value: float) -> str:
    return f"{value:6.2%}" if -1.0 <= value <= 1.0 else f"{value:+.3f}"


def print_results(results: Sequence[BudgetResult], correlations: Dict[str, float]) -> None:
    print("Engine A cheap/free signal budget")
    print("This validates simulation logic only; it does not validate a real model.")
    print()
    print("correlations")
    for key, value in correlations.items():
        print(f"  {key:>32}: {value:+.3f}")
    print()
    header = (
        f"{'mode':>18} {'cost':>6} {'exit':>8} {'fid':>8} {'wrong':>8} "
        f"{'boundary':>9} {'plateau':>9} {'skip':>8} {'utility':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        m = result.metrics
        utility = m["skip_gain"] - result.cost_units * 0.01
        print(
            f"{result.name:>18} {result.cost_units:6.2f} "
            f"{m['exit_rate']:8.2%} {m['fidelity']:8.2%} "
            f"{m['false_exit_rate']:8.2%} {m['boundary_recall']:9.2%} "
            f"{m['false_plateau_block']:9.2%} {m['skip_gain']:8.2%} "
            f"{utility:9.2%}"
        )
    print()


def validate(results: Sequence[BudgetResult], correlations: Dict[str, float]) -> Dict[str, bool]:
    by_name = {result.name: result for result in results}
    sharp = by_name["free_sharpness"].metrics
    b_veto = by_name["free_plus_b_veto"].metrics
    confidence = by_name["confidence"].metrics
    return {
        "weak_a_b_correlation": abs(correlations["corr_tension_confidence"]) < 0.20,
        "sharpness_beats_confidence_fidelity": sharp["fidelity"] > confidence["fidelity"],
        "b_veto_reduces_false_exits": b_veto["false_exit_rate"] < sharp["false_exit_rate"],
        "b_veto_blocks_false_plateaus": b_veto["false_plateau_block"] > sharp["false_plateau_block"],
        "b_veto_keeps_useful_exit_rate": b_veto["exit_rate"] >= 0.25,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--exit_layer", type=int, default=16)
    parser.add_argument("--total_layers", type=int, default=32)
    args = parser.parse_args()

    modes = [
        "confidence",
        "free_sharpness",
        "free_plus_b_veto",
        "stability_guard",
        "a_b_stability",
    ]
    results = [
        run_mode(
            mode=mode,
            seeds=args.seeds,
            tokens=args.tokens,
            exit_layer=args.exit_layer,
            total_layers=args.total_layers,
        )
        for mode in modes
    ]
    correlations = correlation_probe(seeds=args.seeds, tokens=args.tokens)
    print_results(results, correlations)
    checks = validate(results, correlations)
    print("validation")
    for key, ok in checks.items():
        print(f"  {key:>32}: {'PASS' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
