#!/usr/bin/env python3
"""
Physics-informed Monte Carlo planner for Engine A host tests.

This is not a claim that quantum mechanics literally predicts transformer
behavior. It uses physics-style equations as conservative priors:

- Roofline timing: wall time is bounded by compute throughput and memory
  bandwidth.
- Landau-Zener transition: small logit gaps plus fast boundary drift produce
  false plateaus.
- Spiked random matrix separation: an overparameterized model is more likely to
  expose a clean low-dimensional boundary signal.
- Localization/diffusion: long context can either localize into a simple
  trajectory or diffuse into memory tension.

The output is a ranked test plan. Calibrate it with one real host timing before
trusting absolute wall-clock predictions.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    non_embed_b: float
    layers: int
    hidden: int
    q_heads: int
    kv_heads: int
    vocab: int
    family: str

    @property
    def head_dim(self) -> int:
        return max(1, self.hidden // self.q_heads)


@dataclass(frozen=True)
class HardwareSpec:
    ram_per_socket_gb: float
    sockets: int
    mem_bandwidth_gbs: float
    compute_tflops: float
    numa_penalty: float

    @property
    def total_ram_gb(self) -> float:
        return self.ram_per_socket_gb * self.sockets


@dataclass(frozen=True)
class Trial:
    model: str
    dtype: str
    fits: bool
    peak_gb: float
    wall_seconds: float
    fidelity: float
    exit_rate: float
    false_plateau: float
    success: bool


MODEL_SPECS: Tuple[ModelSpec, ...] = (
    ModelSpec(
        name="Qwen/Qwen3-8B",
        params_b=8.1907,
        non_embed_b=6.95,
        layers=36,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        vocab=151_936,
        family="qwen3",
    ),
    ModelSpec(
        name="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.6156,
        non_embed_b=6.55,
        layers=28,
        hidden=3584,
        q_heads=28,
        kv_heads=4,
        vocab=152_064,
        family="qwen2",
    ),
    ModelSpec(
        name="meta-llama/Llama-3.1-8B-Instruct",
        params_b=8.03,
        non_embed_b=7.50,
        layers=32,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        vocab=128_256,
        family="llama",
    ),
    ModelSpec(
        name="mistralai/Mistral-7B-Instruct-v0.3",
        params_b=7.25,
        non_embed_b=7.00,
        layers=32,
        hidden=4096,
        q_heads=32,
        kv_heads=8,
        vocab=32_768,
        family="mistral",
    ),
    ModelSpec(
        name="google/gemma-2-9b-it",
        params_b=9.24,
        non_embed_b=8.45,
        layers=42,
        hidden=3584,
        q_heads=16,
        kv_heads=8,
        vocab=256_000,
        family="gemma",
    ),
)


DTYPE_BYTES = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.1,
    "int4": 0.60,
}


DTYPE_SIGNAL_PENALTY = {
    "float32": 0.00,
    "bfloat16": 0.01,
    "float16": 0.015,
    "int8": 0.035,
    "int4": 0.075,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return ordered[clamp(index, 0, len(ordered) - 1)]  # type: ignore[arg-type]


def bytes_per_param(dtype: str) -> float:
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype: {dtype}")
    return DTYPE_BYTES[dtype]


def estimate_memory_gb(
    model: ModelSpec,
    dtype: str,
    context_tokens: int,
    eval_tail_tokens: int,
    exit_layers: int,
    hidden_states: bool = True,
    current_hf_logits: bool = True,
) -> float:
    b = bytes_per_param(dtype)
    weight_gb = model.params_b * b
    kv_gb = (
        2.0
        * model.layers
        * model.kv_heads
        * model.head_dim
        * context_tokens
        * b
        / 1e9
    )
    hidden_gb = 0.0
    if hidden_states:
        hidden_gb = (model.layers + 1) * context_tokens * model.hidden * b / 1e9

    # Current AutoModelForCausalLM path returns final logits for all positions.
    # Tail scoring only avoids repeated intermediate lm_head projections.
    final_logits_tokens = context_tokens if current_hf_logits else eval_tail_tokens
    final_logits_gb = final_logits_tokens * model.vocab * b / 1e9
    tail_logits_gb = eval_tail_tokens * model.vocab * b * max(1, exit_layers * 2) / 1e9

    allocator_overhead = 1.20
    return allocator_overhead * (weight_gb + kv_gb + hidden_gb + final_logits_gb + tail_logits_gb)


def estimate_flops(
    model: ModelSpec,
    context_tokens: int,
    eval_tail_tokens: int,
    prompts: int,
    exit_layers: int,
    current_hf_logits: bool = True,
) -> float:
    dense = 2.0 * model.non_embed_b * 1e9 * context_tokens
    attention = 4.0 * model.layers * (context_tokens**2) * model.hidden
    final_logits_tokens = context_tokens if current_hf_logits else eval_tail_tokens
    final_head = 2.0 * model.hidden * model.vocab * final_logits_tokens
    intermediate_heads = (
        2.0
        * model.hidden
        * model.vocab
        * eval_tail_tokens
        * max(1, exit_layers * 2)
    )
    return prompts * (dense + attention + final_head + intermediate_heads)


def estimate_streamed_bytes(
    model: ModelSpec,
    dtype: str,
    context_tokens: int,
    prompts: int,
) -> float:
    b = bytes_per_param(dtype)
    weights = model.params_b * 1e9 * b
    activations = model.layers * context_tokens * model.hidden * b * 10.0
    return prompts * (weights + activations)


def landau_zener_false_plateau(gap: float, drift: float) -> float:
    # P(transition) ~= exp(-2*pi*gap^2 / velocity). This is used as a
    # structure-preserving proxy, not a literal quantum transition.
    return math.exp(-2.0 * math.pi * gap * gap / max(0.03, drift))


def spiked_matrix_signal(model: ModelSpec, context_tokens: int, dtype: str, rng: random.Random) -> float:
    overparam = (model.params_b / 7.0) ** 0.25
    depth = (model.layers / 32.0) ** 0.35
    width = (model.hidden / 4096.0) ** 0.20
    context_noise = 1.0 + 0.055 * math.log2(max(1.0, context_tokens / 1024.0))
    quant_noise = 1.0 + 2.0 * DTYPE_SIGNAL_PENALTY[dtype]
    family_bias = {
        "qwen3": 1.04,
        "qwen2": 1.00,
        "llama": 1.02,
        "mistral": 0.98,
        "gemma": 0.96,
    }.get(model.family, 1.0)
    return family_bias * overparam * depth * width / (context_noise * quant_noise) + rng.gauss(0.0, 0.08)


def localization_bonus(context_tokens: int, rng: random.Random) -> float:
    # Short deterministic continuations localize; very long context diffuses.
    localized = math.exp(-context_tokens / 90_000.0)
    return clamp(0.03 * localized + rng.gauss(0.0, 0.01), -0.02, 0.05)


def run_trial(
    model: ModelSpec,
    dtype: str,
    hardware: HardwareSpec,
    context_tokens: int,
    eval_tail_tokens: int,
    prompts: int,
    exit_layers: int,
    target_fidelity: float,
    min_exit_rate: float,
    engine_b_veto: bool,
    rng: random.Random,
) -> Trial:
    peak_gb = estimate_memory_gb(
        model=model,
        dtype=dtype,
        context_tokens=context_tokens,
        eval_tail_tokens=eval_tail_tokens,
        exit_layers=exit_layers,
    )
    fits_one_socket = peak_gb <= hardware.ram_per_socket_gb
    fits = peak_gb <= hardware.total_ram_gb

    bw = hardware.mem_bandwidth_gbs * rng.lognormvariate(0.0, 0.18)
    compute = hardware.compute_tflops * rng.lognormvariate(0.0, 0.28)
    if fits and not fits_one_socket:
        bw /= hardware.numa_penalty
        compute /= math.sqrt(hardware.numa_penalty)

    flops = estimate_flops(
        model=model,
        context_tokens=context_tokens,
        eval_tail_tokens=eval_tail_tokens,
        prompts=prompts,
        exit_layers=exit_layers,
    )
    streamed = estimate_streamed_bytes(model, dtype=dtype, context_tokens=context_tokens, prompts=prompts)
    compute_time = flops / max(1e9, compute * 1e12)
    bandwidth_time = streamed / max(1e6, bw * 1e9)
    python_overhead = rng.lognormvariate(math.log(1.35), 0.25)
    wall_seconds = max(compute_time, bandwidth_time) * python_overhead
    if not fits:
        wall_seconds = float("inf")

    snr = spiked_matrix_signal(model, context_tokens, dtype, rng)
    gap = clamp(0.26 + 0.14 * (snr - 1.0) + rng.gauss(0.0, 0.06), 0.04, 0.75)
    drift = clamp(0.18 + 0.06 * math.log2(max(1.0, context_tokens / 1024.0)) + rng.gauss(0.0, 0.05), 0.04, 0.60)
    false_plateau = landau_zener_false_plateau(gap=gap, drift=drift)
    tension = clamp(0.30 + 0.65 * false_plateau + rng.gauss(0.0, 0.12), 0.0, 1.0)

    b_bonus = 0.0
    b_exit_tax = 0.0
    if engine_b_veto:
        b_bonus = 0.035 * sigmoid(6.0 * (tension - 0.45))
        b_exit_tax = 0.035 * sigmoid(6.0 * (tension - 0.45))

    quant_penalty = DTYPE_SIGNAL_PENALTY[dtype]
    base_quality = sigmoid(4.0 * (snr - 0.94))
    fidelity = clamp(
        0.885
        + 0.105 * base_quality
        + localization_bonus(context_tokens, rng)
        + b_bonus
        - 0.060 * false_plateau
        - quant_penalty
        + rng.gauss(0.0, 0.012),
        0.0,
        0.999,
    )
    exit_rate = clamp(
        0.05
        + 0.43 * sigmoid(3.5 * (snr - 0.95))
        - 0.12 * false_plateau
        - b_exit_tax
        - 0.5 * quant_penalty
        + rng.gauss(0.0, 0.035),
        0.0,
        0.70,
    )
    success = fits and fidelity >= target_fidelity and exit_rate >= min_exit_rate
    return Trial(
        model=model.name,
        dtype=dtype,
        fits=fits,
        peak_gb=peak_gb,
        wall_seconds=wall_seconds,
        fidelity=fidelity,
        exit_rate=exit_rate,
        false_plateau=false_plateau,
        success=success,
    )


def summarize(trials: Sequence[Trial]) -> Dict[str, float]:
    finite_wall = [t.wall_seconds for t in trials if math.isfinite(t.wall_seconds)]
    return {
        "fit_rate": sum(t.fits for t in trials) / max(1, len(trials)),
        "success_rate": sum(t.success for t in trials) / max(1, len(trials)),
        "peak_gb_p50": percentile([t.peak_gb for t in trials], 0.50),
        "wall_p50": percentile(finite_wall, 0.50),
        "wall_p90": percentile(finite_wall, 0.90),
        "fidelity_p50": percentile([t.fidelity for t in trials], 0.50),
        "exit_rate_p50": percentile([t.exit_rate for t in trials], 0.50),
        "false_plateau_p50": percentile([t.false_plateau for t in trials], 0.50),
    }


def run_monte_carlo(
    models: Sequence[ModelSpec],
    dtypes: Sequence[str],
    hardware: HardwareSpec,
    context_tokens: int,
    eval_tail_tokens: int,
    prompts: int,
    exit_layers: int,
    target_fidelity: float,
    min_exit_rate: float,
    engine_b_veto: bool,
    samples: int,
    seed: int,
) -> List[Tuple[str, str, Dict[str, float]]]:
    rows: List[Tuple[str, str, Dict[str, float]]] = []
    rng = random.Random(seed)
    for model in models:
        for dtype in dtypes:
            trials = [
                run_trial(
                    model=model,
                    dtype=dtype,
                    hardware=hardware,
                    context_tokens=context_tokens,
                    eval_tail_tokens=eval_tail_tokens,
                    prompts=prompts,
                    exit_layers=exit_layers,
                    target_fidelity=target_fidelity,
                    min_exit_rate=min_exit_rate,
                    engine_b_veto=engine_b_veto,
                    rng=rng,
                )
                for _ in range(samples)
            ]
            rows.append((model.name, dtype, summarize(trials)))
    rows.sort(key=lambda row: (-row[2]["success_rate"], row[2]["wall_p50"], -row[2]["fidelity_p50"]))
    return rows


def select_models(raw: str) -> List[ModelSpec]:
    if raw == "all":
        return list(MODEL_SPECS)
    wanted = {item.strip().lower() for item in raw.split(",") if item.strip()}
    selected = [model for model in MODEL_SPECS if model.name.lower() in wanted or model.family.lower() in wanted]
    if not selected:
        known = ", ".join(model.name for model in MODEL_SPECS)
        raise SystemExit(f"No models selected. Known models: {known}")
    return selected


def print_rows(rows: Sequence[Tuple[str, str, Dict[str, float]]], context_tokens: int, prompts: int) -> None:
    print("Physics-informed Engine A Monte Carlo")
    print("Calibrate absolute wall-clock with one real host timing.")
    print(f"context_tokens={context_tokens} prompts={prompts}")
    print()
    header = (
        f"{'rank':>4} {'model':<36} {'dtype':>8} {'fit':>7} {'work':>7} "
        f"{'wall50':>9} {'wall90':>9} {'peakGB':>8} {'fid50':>8} {'exit50':>8} {'LZflip':>8}"
    )
    print(header)
    print("-" * len(header))
    for index, (model, dtype, metrics) in enumerate(rows, start=1):
        print(
            f"{index:4d} {model:<36} {dtype:>8} "
            f"{metrics['fit_rate']:7.1%} {metrics['success_rate']:7.1%} "
            f"{metrics['wall_p50']:9.1f} {metrics['wall_p90']:9.1f} "
            f"{metrics['peak_gb_p50']:8.1f} {metrics['fidelity_p50']:8.2%} "
            f"{metrics['exit_rate_p50']:8.2%} {metrics['false_plateau_p50']:8.2%}"
        )
    print()
    print("Legend: work = P(fits and meets target fidelity/exit-rate in the prior).")
    print("LZflip = Landau-Zener-style false-plateau transition prior.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all, family names, or exact model ids.")
    parser.add_argument("--dtypes", default="float32,bfloat16,int8")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context_tokens", type=int, default=2048)
    parser.add_argument("--eval_tail_tokens", type=int, default=128)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--exit_layers", type=int, default=1, help="Number of exit-layer points in this run.")
    parser.add_argument("--target_fidelity", type=float, default=0.97)
    parser.add_argument("--min_exit_rate", type=float, default=0.10)
    parser.add_argument("--engine_b_veto", action="store_true")
    parser.add_argument("--ram_per_socket_gb", type=float, default=350.0)
    parser.add_argument("--sockets", type=int, default=2)
    parser.add_argument("--mem_bandwidth_gbs", type=float, default=110.0)
    parser.add_argument("--compute_tflops", type=float, default=2.0)
    parser.add_argument("--numa_penalty", type=float, default=1.7)
    args = parser.parse_args()

    models = select_models(args.models)
    dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    hardware = HardwareSpec(
        ram_per_socket_gb=args.ram_per_socket_gb,
        sockets=args.sockets,
        mem_bandwidth_gbs=args.mem_bandwidth_gbs,
        compute_tflops=args.compute_tflops,
        numa_penalty=args.numa_penalty,
    )
    rows = run_monte_carlo(
        models=models,
        dtypes=dtypes,
        hardware=hardware,
        context_tokens=args.context_tokens,
        eval_tail_tokens=args.eval_tail_tokens,
        prompts=args.prompts,
        exit_layers=args.exit_layers,
        target_fidelity=args.target_fidelity,
        min_exit_rate=args.min_exit_rate,
        engine_b_veto=args.engine_b_veto,
        samples=args.samples,
        seed=args.seed,
    )
    print_rows(rows, context_tokens=args.context_tokens, prompts=args.prompts)


if __name__ == "__main__":
    main()
