#!/usr/bin/env python3
"""Local tests for Engine A simulation invariants."""

from __future__ import annotations

import unittest

from measurements.free_signal_budget import correlation_probe, run_mode, validate
from measurements.physics_monte_carlo import (
    HardwareSpec,
    MODEL_SPECS,
    estimate_memory_gb,
    run_monte_carlo,
)
from measurements.simulate_host_branches import run_grid
from measurements.synthetic_engine_a import run_invariants


class EngineASimulationTests(unittest.TestCase):
    def test_synthetic_invariants_pass(self) -> None:
        metrics = run_invariants(seeds=20, tokens=256, exit_layer=16, total_layers=32)
        self.assertGreaterEqual(metrics["calibration"]["calibration_monotonic"], 0.75)
        self.assertGreater(metrics["entropy_compute_inversion"]["inversion_strength"], 0.05)
        self.assertGreater(metrics["layer_collapse"]["collapse_gap"], 0.30)
        self.assertLess(
            metrics["false_plateau_guard"]["guarded_false_exit"],
            metrics["false_plateau_guard"]["confidence_false_exit"],
        )
        self.assertGreater(
            metrics["fusion_gain"]["fused_exit_rate"],
            metrics["fusion_gain"]["guarded_exit_rate"],
        )

    def test_branch_grid_has_no_ambiguous_bucket(self) -> None:
        counts = run_grid(limit_print=0)
        self.assertEqual(counts.get("ambiguous", 0), 0)

    def test_engine_b_veto_is_complementary(self) -> None:
        results = [
            run_mode(mode, seeds=20, tokens=256, exit_layer=16, total_layers=32)
            for mode in [
                "confidence",
                "free_sharpness",
                "free_plus_b_veto",
                "stability_guard",
                "a_b_stability",
            ]
        ]
        checks = validate(results, correlation_probe(seeds=20, tokens=256))
        self.assertTrue(all(checks.values()), checks)

    def test_physics_monte_carlo_ranks_qwen_host_target(self) -> None:
        qwen = [model for model in MODEL_SPECS if model.name == "Qwen/Qwen3-8B"]
        peak_gb = estimate_memory_gb(
            model=qwen[0],
            dtype="float32",
            context_tokens=2048,
            eval_tail_tokens=128,
            exit_layers=1,
        )
        self.assertLess(peak_gb, 350.0)
        rows = run_monte_carlo(
            models=qwen,
            dtypes=["float32"],
            hardware=HardwareSpec(
                ram_per_socket_gb=350.0,
                sockets=2,
                mem_bandwidth_gbs=110.0,
                compute_tflops=2.0,
                numa_penalty=1.7,
            ),
            context_tokens=2048,
            eval_tail_tokens=128,
            prompts=2,
            exit_layers=1,
            target_fidelity=0.94,
            min_exit_rate=0.05,
            engine_b_veto=True,
            samples=50,
            seed=11,
        )
        self.assertGreater(rows[0][2]["fit_rate"], 0.99)
        self.assertGreater(rows[0][2]["success_rate"], 0.50)


if __name__ == "__main__":
    unittest.main()
