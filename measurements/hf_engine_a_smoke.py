#!/usr/bin/env python3
"""
Hugging Face logical early-exit smoke test for Engine A.

This does not physically skip layers. It measures whether an intermediate layer
could have exited safely by projecting that layer through the model's final norm
and lm_head, then comparing against the full-depth distribution.

The harness is deliberately diagnostic:

- Sweep multiple exit layers and thresholds in one run.
- Compare gate modes: confidence, margin, stability, fused.
- Report top-1 fidelity, top-k agreement, false exits, and calibration buckets.
- Route the result through docs/inclusive_branch_tree.md before doing C++ work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "The capital of France is",
    "In Python, a list comprehension is useful because it",
    "The secret password is Supernova. Based on the text, the secret password is",
    "Complete the sequence: Monday, Tuesday, Wednesday,",
    "A careful scientific answer should distinguish evidence from",
    "The opposite of hot is",
    "In a JSON object, keys are usually written as",
    "A triangle has three sides. A square has",
]


@dataclass
class PromptCache:
    prompt: str
    final_stats: Dict[str, torch.Tensor]
    layer_stats: Dict[int, Dict[str, torch.Tensor]]


@dataclass
class RunConfig:
    exit_layer: int
    threshold: float
    gate: str
    gap_threshold: float
    entropy_threshold: float


def parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(value)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def top_stats(logits: torch.Tensor, top_k: int) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)
    return {
        "top_ids": top_ids,
        "top_probs": top_probs,
        "top_id": top_ids[..., 0],
        "top_prob": top_probs[..., 0],
        "logit_gap": torch.log(top_probs[..., 0].clamp_min(1e-12))
        - torch.log(top_probs[..., 1].clamp_min(1e-12)),
        "topk_cov": top_probs.sum(dim=-1),
        "entropy": entropy_from_logits(logits),
    }


def model_norm(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def load_prompts(prompt_file: str | None, max_prompts: int | None) -> List[str]:
    if prompt_file:
        prompts = [
            line.strip()
            for line in Path(prompt_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        prompts = list(DEFAULT_PROMPTS)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts


def build_prompt_cache(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    exit_layers: Sequence[int],
    max_tokens: int,
    top_k: int,
) -> Tuple[List[PromptCache], int]:
    caches: List[PromptCache] = []
    needed_layers = set(exit_layers)
    needed_layers.update(max(1, layer - 1) for layer in exit_layers)

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        encoded = encoded.to(model.device)
        prompt_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, use_cache=False)

        num_layers = len(out.hidden_states) - 1
        for layer in needed_layers:
            if layer < 1 or layer > num_layers:
                raise ValueError(f"exit layer {layer} is outside valid range [1, {num_layers}]")

        final_logits = out.logits[:, :-1, :]
        final_stats = top_stats(final_logits, top_k=top_k)
        layer_stats: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer in sorted(needed_layers):
            hidden = model_norm(model, out.hidden_states[layer])[:, :-1, :]
            logits = model.lm_head(hidden)
            layer_stats[layer] = top_stats(logits, top_k=top_k)
        caches.append(PromptCache(prompt=prompt_text, final_stats=final_stats, layer_stats=layer_stats))

    return caches, num_layers


def topk_contains(final_ids: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return (final_ids == candidate.unsqueeze(-1)).any(dim=-1)


def topk_overlap(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Small top-k sets; explicit loop keeps this readable and robust.
    flat_a = a.reshape(-1, a.shape[-1]).tolist()
    flat_b = b.reshape(-1, b.shape[-1]).tolist()
    overlaps = []
    for left, right in zip(flat_a, flat_b):
        overlaps.append(len(set(left) & set(right)) / max(1, len(set(left) | set(right))))
    return torch.tensor(overlaps, dtype=torch.float32, device=a.device).reshape(a.shape[:-1])


def gate_score(mid: Dict[str, torch.Tensor], prev: Dict[str, torch.Tensor]) -> torch.Tensor:
    stable = (mid["top_id"] == prev["top_id"]).float()
    raw = (
        5.0 * mid["top_prob"]
        + 0.35 * mid["logit_gap"]
        + 0.75 * mid["topk_cov"]
        - 0.16 * mid["entropy"]
        + 0.75 * stable
        - 4.2
    )
    return sigmoid(raw)


def gate_accept(config: RunConfig, mid: Dict[str, torch.Tensor], prev: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    stable = mid["top_id"] == prev["top_id"]
    if config.gate == "confidence":
        score = mid["top_prob"]
        accept = score >= config.threshold
    elif config.gate == "margin":
        score = mid["top_prob"]
        accept = (score >= config.threshold) & (mid["logit_gap"] >= config.gap_threshold)
    elif config.gate == "stability":
        score = mid["top_prob"]
        accept = (
            (score >= config.threshold)
            & (mid["logit_gap"] >= config.gap_threshold)
            & (mid["entropy"] <= config.entropy_threshold)
            & stable
        )
    elif config.gate == "fused":
        score = gate_score(mid, prev)
        accept = score >= config.threshold
    else:
        raise ValueError(f"unknown gate: {config.gate}")
    return accept, score


def evaluate_config(caches: Sequence[PromptCache], config: RunConfig, num_layers: int, top_k: int) -> Dict[str, float]:
    totals = {
        "tokens": 0.0,
        "accepted": 0.0,
        "safe": 0.0,
        "wrong": 0.0,
        "top1_agree": 0.0,
        "topk_agree": 0.0,
        "topk_overlap": 0.0,
        "accepted_topk_agree": 0.0,
        "score_sum": 0.0,
        "entropy_sum": 0.0,
    }
    skipped_layers = max(0, num_layers - config.exit_layer)

    for cache in caches:
        mid = cache.layer_stats[config.exit_layer]
        prev = cache.layer_stats[max(1, config.exit_layer - 1)]
        final = cache.final_stats
        accept, score = gate_accept(config, mid=mid, prev=prev)

        top1 = mid["top_id"] == final["top_id"]
        topk = topk_contains(final["top_ids"], mid["top_id"])
        overlap = topk_overlap(final["top_ids"], mid["top_ids"])
        safe = accept & top1
        wrong = accept & ~top1

        n = float(top1.numel())
        totals["tokens"] += n
        totals["accepted"] += float(accept.sum().item())
        totals["safe"] += float(safe.sum().item())
        totals["wrong"] += float(wrong.sum().item())
        totals["top1_agree"] += float(top1.float().sum().item())
        totals["topk_agree"] += float(topk.float().sum().item())
        totals["topk_overlap"] += float(overlap.sum().item())
        totals["accepted_topk_agree"] += float((accept & topk).float().sum().item())
        totals["score_sum"] += float(score.float().sum().item())
        totals["entropy_sum"] += float(mid["entropy"].float().sum().item())

    tokens = max(1.0, totals["tokens"])
    accepted = max(1.0, totals["accepted"])
    return {
        "exit_layer": float(config.exit_layer),
        "threshold": config.threshold,
        "tokens": tokens,
        "exit_rate": totals["accepted"] / tokens,
        "fidelity": totals["safe"] / accepted,
        "false_exit_rate": totals["wrong"] / tokens,
        "top1_agreement": totals["top1_agree"] / tokens,
        "topk_agreement": totals["topk_agree"] / tokens,
        "accepted_topk_agreement": totals["accepted_topk_agree"] / accepted,
        "topk_overlap": totals["topk_overlap"] / tokens,
        "avg_layers_skipped": skipped_layers * totals["accepted"] / tokens,
        "skip_gain": skipped_layers * totals["accepted"] / max(1.0, tokens * num_layers),
        "avg_score": totals["score_sum"] / tokens,
        "avg_entropy": totals["entropy_sum"] / tokens,
    }


def bucketize(caches: Sequence[PromptCache], exit_layer: int) -> List[Tuple[str, int, float]]:
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.93), (0.93, 1.01)]
    rows: List[Tuple[float, bool]] = []
    for cache in caches:
        mid = cache.layer_stats[exit_layer]
        final = cache.final_stats
        confs = mid["top_prob"].flatten().tolist()
        agreement = (mid["top_id"] == final["top_id"]).flatten().tolist()
        rows.extend((float(conf), bool(ok)) for conf, ok in zip(confs, agreement))

    output = []
    for low, high in buckets:
        members = [ok for conf, ok in rows if low <= conf < high]
        if members:
            output.append((f"[{low:.2f},{high:.2f})", len(members), sum(members) / len(members)))
    return output


def print_table(rows: Sequence[Dict[str, float]], gate: str) -> None:
    print(f"gate: {gate}")
    header = (
        f"{'layer':>5} {'thr':>6} {'exit':>8} {'fid':>8} {'wrong':>8} "
        f"{'top1':>8} {'topk':>8} {'a_topk':>8} {'skipL':>8} {'gain':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{int(row['exit_layer']):5d} {row['threshold']:6.2f} "
            f"{row['exit_rate']:8.2%} {row['fidelity']:8.2%} "
            f"{row['false_exit_rate']:8.2%} {row['top1_agreement']:8.2%} "
            f"{row['topk_agreement']:8.2%} {row['accepted_topk_agreement']:8.2%} "
            f"{row['avg_layers_skipped']:8.2f} {row['skip_gain']:8.2%}"
        )
    print()


def write_outputs(rows: Sequence[Dict[str, float]], output_jsonl: str | None, output_csv: str | None) -> None:
    if output_jsonl:
        with Path(output_jsonl).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    if output_csv:
        with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exit_layers", default="16", help="Comma-separated layers, e.g. 8,12,16,20,24.")
    parser.add_argument("--thresholds", default="0.90", help="Comma-separated thresholds.")
    parser.add_argument("--gates", default="confidence,margin,stability,fused")
    parser.add_argument("--gap_threshold", type=float, default=0.25)
    parser.add_argument("--entropy_threshold", type=float, default=2.50)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--prompt_file")
    parser.add_argument("--max_prompts", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--output_jsonl")
    parser.add_argument("--output_csv")
    args = parser.parse_args()

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device_map = "auto" if args.device == "auto" else None
    manual_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        model.to(manual_device)
    model.eval()

    prompts = load_prompts(args.prompt_file, args.max_prompts)
    exit_layers = parse_ints(args.exit_layers)
    thresholds = parse_floats(args.thresholds)
    gates = [gate.strip() for gate in args.gates.split(",") if gate.strip()]
    caches, num_layers = build_prompt_cache(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        exit_layers=exit_layers,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
    )

    all_rows: List[Dict[str, float]] = []
    print("Engine A HF logical early-exit smoke")
    print(f"model: {args.model}")
    print(f"num_layers: {num_layers}")
    print(f"prompts: {len(prompts)}")
    print(f"exit_layers: {exit_layers}")
    print(f"thresholds: {thresholds}")
    print()

    for gate in gates:
        gate_rows: List[Dict[str, float]] = []
        for exit_layer in exit_layers:
            for threshold in thresholds:
                config = RunConfig(
                    exit_layer=exit_layer,
                    threshold=threshold,
                    gate=gate,
                    gap_threshold=args.gap_threshold,
                    entropy_threshold=args.entropy_threshold,
                )
                row = evaluate_config(caches=caches, config=config, num_layers=num_layers, top_k=args.top_k)
                row["gate"] = gate
                row["gap_threshold"] = args.gap_threshold
                row["entropy_threshold"] = args.entropy_threshold
                gate_rows.append(row)
                all_rows.append(row)
        print_table(gate_rows, gate=gate)

    print("calibration buckets by exit layer")
    for exit_layer in exit_layers:
        print(f"layer {exit_layer}")
        for bucket, count, acc in bucketize(caches, exit_layer):
            print(f"  {bucket:>12} n={count:>5} acc={acc:.2%}")
    print()

    if all_rows:
        write_outputs(all_rows, output_jsonl=args.output_jsonl, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
