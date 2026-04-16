#!/usr/bin/env python3
"""Local tests for Engine A simulation invariants."""

from __future__ import annotations

import unittest

from measurements.free_signal_budget import correlation_probe, run_mode, validate
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


if __name__ == "__main__":
    unittest.main()
