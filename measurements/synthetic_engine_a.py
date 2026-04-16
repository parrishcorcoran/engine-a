#!/usr/bin/env python3
"""
Synthetic Engine A simulator.

This is not a model test. It validates the test logic before host runs:

- Easy tokens should collapse early and skip layers.
- Boundary tokens should refuse early exit.
- Confidence should be calibrated.
- Stability guards should reduce false plateaus.
- Fused signals should beat confidence-only gates at matched fidelity.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class TokenState:
    index: int
    difficulty: float
    boundary: bool
    false_plateau: bool
    final_token: int
    mid_token: int
    mid_conf: float
    final_conf: float
    logit_gap: float
    stability: float
    cluster_stability: float
    entropy: float


@dataclass
class GateResult:
    accepted: bool
    correct: bool
    boundary: bool
    false_plateau: bool
    layers_used: int
    confidence: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def make_token(rng: random.Random, index: int, total: int) -> TokenState:
    # Later tokens are usually more deterministic. Boundary spikes remain.
    progress = index / max(1, total - 1)
    base_difficulty = rng.betavariate(2.4, 3.2) * (1.0 - 0.35 * progress)
    boundary = rng.random() < (0.10 + 0.22 * base_difficulty)
    if boundary:
        difficulty = clamp(base_difficulty + rng.uniform(0.25, 0.55), 0.0, 1.0)
    else:
        difficulty = clamp(base_difficulty * rng.uniform(0.35, 0.85), 0.0, 1.0)

    false_plateau = boundary and rng.random() < 0.30
    final_token = rng.randrange(50_000)
    mid_wrong = rng.random() < (0.04 + 0.78 * difficulty)
    mid_token = rng.randrange(50_000) if mid_wrong else final_token

    collapse = 1.0 - difficulty
    mid_conf = clamp(0.22 + 0.76 * collapse + rng.gauss(0.0, 0.06), 0.01, 0.995)
    if false_plateau:
        mid_conf = clamp(0.88 + rng.random() * 0.10, 0.01, 0.995)
        mid_token = rng.randrange(50_000)

    final_conf = clamp(0.35 + 0.62 * collapse + rng.gauss(0.0, 0.04), 0.01, 0.999)
    logit_gap = clamp(mid_conf - (0.18 + 0.45 * difficulty) + rng.gauss(0.0, 0.06), -0.50, 0.95)
    stability = clamp(1.0 - difficulty + rng.gauss(0.0, 0.08), 0.0, 1.0)
    if false_plateau:
        stability = clamp(stability - 0.45, 0.0, 1.0)
    cluster_stability = clamp(1.0 - difficulty + rng.gauss(0.0, 0.10), 0.0, 1.0)
    entropy = clamp(0.10 + 3.0 * difficulty + rng.gauss(0.0, 0.18), 0.0, 4.0)

    return TokenState(
        index=index,
        difficulty=difficulty,
        boundary=boundary,
        false_plateau=false_plateau,
        final_token=final_token,
        mid_token=mid_token,
        mid_conf=mid_conf,
        final_conf=final_conf,
        logit_gap=logit_gap,
        stability=stability,
        cluster_stability=cluster_stability,
        entropy=entropy,
    )


def build_sequence(seed: int, tokens: int) -> List[TokenState]:
    rng = random.Random(seed)
    return [make_token(rng, index=i, total=tokens) for i in range(tokens)]


def confidence_gate(token: TokenState, threshold: float) -> bool:
    return token.mid_conf >= threshold


def guarded_gate(token: TokenState, threshold: float) -> bool:
    return (
        token.mid_conf >= threshold
        and token.logit_gap >= 0.22
        and token.stability >= 0.58
    )


def fused_score(token: TokenState) -> float:
    # Seven-dimension proxy: sharpness, gap, stability, cluster, entropy,
    # boundary inverse, and final-confidence proxy.
    raw = (
        2.2 * token.mid_conf
        + 1.1 * token.logit_gap
        + 1.2 * token.stability
        + 0.8 * token.cluster_stability
        - 0.55 * token.entropy
        - 1.4 * float(token.boundary)
        + 0.4 * token.final_conf
        - 2.0
    )
    return sigmoid(raw)


def fused_gate(token: TokenState, threshold: float) -> bool:
    return fused_score(token) >= threshold


def run_gate(
    sequence: Sequence[TokenState],
    gate_name: str,
    threshold: float,
    exit_layer: int,
    total_layers: int,
) -> List[GateResult]:
    results: List[GateResult] = []
    for token in sequence:
        if gate_name == "confidence":
            accepted = confidence_gate(token, threshold)
            confidence = token.mid_conf
        elif gate_name == "guarded":
            accepted = guarded_gate(token, threshold)
            confidence = token.mid_conf
        elif gate_name == "fused":
            accepted = fused_gate(token, threshold)
            confidence = fused_score(token)
        else:
            raise ValueError(f"unknown gate: {gate_name}")

        correct = token.mid_token == token.final_token
        results.append(
            GateResult(
                accepted=accepted,
                correct=correct,
                boundary=token.boundary,
                false_plateau=token.false_plateau,
                layers_used=exit_layer if accepted else total_layers,
                confidence=confidence,
            )
        )
    return results


def summarize(results: Sequence[GateResult], total_layers: int, exit_layer: int) -> Dict[str, float]:
    n = max(1, len(results))
    accepted = [r for r in results if r.accepted]
    boundaries = [r for r in results if r.boundary]
    false_plateaus = [r for r in results if r.false_plateau]
    wrong_exits = [r for r in accepted if not r.correct]
    return {
        "exit_rate": len(accepted) / n,
        "fidelity": sum(r.correct for r in accepted) / max(1, len(accepted)),
        "false_exit_rate": len(wrong_exits) / n,
        "boundary_recall": sum(not r.accepted for r in boundaries) / max(1, len(boundaries)),
        "false_plateau_block": sum(not r.accepted for r in false_plateaus) / max(1, len(false_plateaus)),
        "avg_layers_used": sum(r.layers_used for r in results) / n,
        "avg_layers_skipped": sum(total_layers - r.layers_used for r in results) / n,
        "skip_gain": len(accepted) * (total_layers - exit_layer) / max(1, n * total_layers),
    }


def mean_dict(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(items)
    if not rows:
        return {}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def calibration_monotonic(sequences: Sequence[Sequence[TokenState]]) -> Dict[str, float]:
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.93), (0.93, 1.01)]
    accuracies: List[float] = []
    for low, high in buckets:
        tokens = [t for seq in sequences for t in seq if low <= t.mid_conf < high]
        if tokens:
            accuracies.append(sum(t.mid_token == t.final_token for t in tokens) / len(tokens))
    monotonic_pairs = sum(
        later + 0.03 >= earlier for earlier, later in zip(accuracies, accuracies[1:])
    )
    return {
        "calibration_monotonic": monotonic_pairs / max(1, len(accuracies) - 1),
        "low_bucket_acc": accuracies[0] if accuracies else 0.0,
        "high_bucket_acc": accuracies[-1] if accuracies else 0.0,
    }


def entropy_compute_inversion(sequences: Sequence[Sequence[TokenState]]) -> Dict[str, float]:
    early = [t for seq in sequences for t in seq[: len(seq) // 3]]
    late = [t for seq in sequences for t in seq[(2 * len(seq)) // 3 :]]
    early_easy = sum(t.difficulty < 0.35 for t in early) / max(1, len(early))
    late_easy = sum(t.difficulty < 0.35 for t in late) / max(1, len(late))
    return {
        "early_easy_rate": early_easy,
        "late_easy_rate": late_easy,
        "inversion_strength": late_easy - early_easy,
    }


def layer_collapse(sequences: Sequence[Sequence[TokenState]]) -> Dict[str, float]:
    easy = [t for seq in sequences for t in seq if t.difficulty < 0.30]
    hard = [t for seq in sequences for t in seq if t.difficulty >= 0.70]
    easy_match = sum(t.mid_token == t.final_token for t in easy) / max(1, len(easy))
    hard_match = sum(t.mid_token == t.final_token for t in hard) / max(1, len(hard))
    return {
        "easy_mid_match": easy_match,
        "hard_mid_match": hard_match,
        "collapse_gap": easy_match - hard_match,
    }


def run_invariants(seeds: int, tokens: int, exit_layer: int, total_layers: int) -> Dict[str, Dict[str, float]]:
    sequences = [build_sequence(seed, tokens) for seed in range(seeds)]
    confidence_rows = [
        summarize(run_gate(seq, "confidence", 0.90, exit_layer, total_layers), total_layers, exit_layer)
        for seq in sequences
    ]
    guarded_rows = [
        summarize(run_gate(seq, "guarded", 0.90, exit_layer, total_layers), total_layers, exit_layer)
        for seq in sequences
    ]
    fused_rows = [
        summarize(run_gate(seq, "fused", 0.92, exit_layer, total_layers), total_layers, exit_layer)
        for seq in sequences
    ]
    confidence = mean_dict(confidence_rows)
    guarded = mean_dict(guarded_rows)
    fused = mean_dict(fused_rows)
    return {
        "calibration": calibration_monotonic(sequences),
        "entropy_compute_inversion": entropy_compute_inversion(sequences),
        "layer_collapse": layer_collapse(sequences),
        "confidence_gate": confidence,
        "guarded_gate": guarded,
        "fused_gate": fused,
        "false_plateau_guard": {
            "confidence_false_exit": confidence["false_exit_rate"],
            "guarded_false_exit": guarded["false_exit_rate"],
            "confidence_plateau_block": confidence["false_plateau_block"],
            "guarded_plateau_block": guarded["false_plateau_block"],
        },
        "fusion_gain": {
            "guarded_exit_rate": guarded["exit_rate"],
            "fused_exit_rate": fused["exit_rate"],
            "guarded_fidelity": guarded["fidelity"],
            "fused_fidelity": fused["fidelity"],
            "fused_skip_gain": fused["skip_gain"],
        },
    }


def pct(value: float) -> str:
    return f"{value:6.2%}" if 0.0 <= value <= 1.0 else f"{value:+.3f}"


def print_block(name: str, metrics: Dict[str, float]) -> None:
    print(name)
    for key, value in metrics.items():
        print(f"  {key:>28}: {pct(value)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--exit_layer", type=int, default=16)
    parser.add_argument("--total_layers", type=int, default=32)
    args = parser.parse_args()

    metrics = run_invariants(
        seeds=args.seeds,
        tokens=args.tokens,
        exit_layer=args.exit_layer,
        total_layers=args.total_layers,
    )
    print("Synthetic Engine A invariant check")
    print("This validates harness logic only; it does not validate a real model.")
    print()
    for name, block in metrics.items():
        print_block(name, block)

    passed = {
        "calibration": metrics["calibration"]["calibration_monotonic"] >= 0.75,
        "entropy_compute_inversion": metrics["entropy_compute_inversion"]["inversion_strength"] > 0.05,
        "layer_collapse": metrics["layer_collapse"]["collapse_gap"] > 0.30,
        "false_plateau_guard": (
            metrics["false_plateau_guard"]["guarded_false_exit"]
            < metrics["false_plateau_guard"]["confidence_false_exit"]
        ),
        "fusion_gain": (
            metrics["fusion_gain"]["fused_exit_rate"] > metrics["fusion_gain"]["guarded_exit_rate"]
            and metrics["fusion_gain"]["fused_fidelity"] >= metrics["fusion_gain"]["guarded_fidelity"] - 0.03
        ),
    }
    print("Invariant summary")
    for key, ok in passed.items():
        print(f"  {key:>28}: {'PASS' if ok else 'FAIL'}")
    if not all(passed.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
