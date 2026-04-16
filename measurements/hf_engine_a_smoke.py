#!/usr/bin/env python3
"""
Hugging Face logical early-exit smoke test for Engine A.

This does not physically skip layers. It measures whether an intermediate layer
could have exited safely by projecting that layer through the model's final norm
and lm_head, then comparing against the full-depth top token.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "The capital of France is",
    "In Python, a list comprehension is useful because it",
    "The secret password is Supernova. Based on the text, the secret password is",
    "Complete the sequence: Monday, Tuesday, Wednesday,",
    "A careful scientific answer should distinguish evidence from",
]


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def top_stats(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=10, dim=-1)
    top2 = top_probs[..., :2]
    return {
        "top_id": top_ids[..., 0],
        "top_prob": top_probs[..., 0],
        "logit_gap": torch.log(top2[..., 0].clamp_min(1e-12)) - torch.log(top2[..., 1].clamp_min(1e-12)),
        "top10_cov": top_probs.sum(dim=-1),
        "entropy": entropy_from_logits(logits),
    }


def bucketize(rows: Iterable[Tuple[float, bool]]) -> List[Tuple[str, int, float]]:
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.93), (0.93, 1.01)]
    output = []
    data = list(rows)
    for low, high in buckets:
        members = [ok for conf, ok in data if low <= conf < high]
        if members:
            output.append((f"[{low:.2f},{high:.2f})", len(members), sum(members) / len(members)))
    return output


def model_norm(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden)
    return hidden


def run_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    exit_layer: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True, use_cache=False)

    hidden_states = out.hidden_states
    if exit_layer < 1 or exit_layer >= len(hidden_states):
        raise ValueError(f"exit_layer must be in [1, {len(hidden_states) - 1}], got {exit_layer}")

    final_logits = out.logits[:, :-1, :]
    mid_hidden = model_norm(model, hidden_states[exit_layer])[:, :-1, :]
    mid_logits = model.lm_head(mid_hidden)

    final = top_stats(final_logits)
    mid = top_stats(mid_logits)
    agreement = mid["top_id"] == final["top_id"]
    accepted = mid["top_prob"] >= threshold
    safe = accepted & agreement
    wrong = accepted & ~agreement

    total = max(1, agreement.numel())
    accepted_count = int(accepted.sum().item())
    wrong_count = int(wrong.sum().item())
    num_layers = len(hidden_states) - 1
    skipped = max(0, num_layers - exit_layer)

    return {
        "tokens": float(total),
        "accepted": float(accepted_count),
        "safe": float(safe.sum().item()),
        "wrong": float(wrong_count),
        "top1_agreement": float(agreement.float().mean().item()),
        "exit_rate": accepted_count / total,
        "fidelity": float(safe.sum().item()) / max(1, accepted_count),
        "false_exit_rate": wrong_count / total,
        "avg_layers_skipped": skipped * accepted_count / total,
        "avg_mid_conf": float(mid["top_prob"].mean().item()),
        "avg_mid_entropy": float(mid["entropy"].mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exit_layer", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = parser.parse_args()

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device_map = "auto" if args.device == "auto" else None
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if device_map is None:
        model.to(device)
    model.eval()

    rows = []
    calibration_rows: List[Tuple[float, bool]] = []
    for prompt in PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        prompt_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
        stats = run_prompt(model, tokenizer, prompt_text, args.exit_layer, args.threshold, device)
        rows.append(stats)

        # Re-run lightweight for calibration rows. Kept separate for clarity.
        batch = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)
        final_logits = out.logits[:, :-1, :]
        mid_hidden = model_norm(model, out.hidden_states[args.exit_layer])[:, :-1, :]
        mid_logits = model.lm_head(mid_hidden)
        final = top_stats(final_logits)
        mid = top_stats(mid_logits)
        agreement = (mid["top_id"] == final["top_id"]).flatten().tolist()
        confs = mid["top_prob"].flatten().tolist()
        calibration_rows.extend((float(conf), bool(ok)) for conf, ok in zip(confs, agreement))

    totals: Dict[str, float] = {}
    for key in rows[0]:
        totals[key] = sum(row[key] for row in rows)
    tokens = max(1.0, totals["tokens"])
    accepted = max(1.0, totals["accepted"])

    print("Engine A HF logical early-exit smoke")
    print(f"model: {args.model}")
    print(f"exit_layer: {args.exit_layer}")
    print(f"threshold: {args.threshold:.3f}")
    print()
    print(f"tokens: {int(tokens)}")
    print(f"exit_rate: {totals['accepted'] / tokens:.2%}")
    print(f"fidelity: {totals['safe'] / accepted:.2%}")
    print(f"false_exit_rate: {totals['wrong'] / tokens:.2%}")
    print(f"top1_agreement: {sum(row['top1_agreement'] for row in rows) / len(rows):.2%}")
    print(f"avg_layers_skipped: {sum(row['avg_layers_skipped'] for row in rows) / len(rows):.2f}")
    print(f"avg_mid_conf: {sum(row['avg_mid_conf'] for row in rows) / len(rows):.2%}")
    print(f"avg_mid_entropy: {sum(row['avg_mid_entropy'] for row in rows) / len(rows):.3f}")
    print()
    print("calibration buckets")
    for bucket, count, acc in bucketize(calibration_rows):
        print(f"  {bucket:>12} n={count:>5} acc={acc:.2%}")


if __name__ == "__main__":
    main()
