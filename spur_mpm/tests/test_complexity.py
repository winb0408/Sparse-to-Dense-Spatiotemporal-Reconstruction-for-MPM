"""The cost numbers must be exact where they are analytic and honest where they
are measured.

A complexity table is easy to produce and easy to get quietly wrong: a hook left
registered doubles a count, a MAC formula that forgets the kernel area is off by
9, a latency reported without its device is a number that cannot be used. These
tests pin the two things a reader cannot check by eye -- the closed-form MAC
count of a known layer stack, and the hook bookkeeping -- and leave the latency
checks to invariants (ordering of the percentiles) rather than to wall-clock
thresholds, which would be flaky on a shared machine.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from spur_mpm.metrics.complexity import (
    complexity_report,
    count_macs,
    measure_latency,
    throughput_for_volume,
)

SHAPE = (1, 1, 8, 8)


def tiny_model() -> nn.Sequential:
    """``1 -> 4`` then ``4 -> 4``, both 3x3: MACs are countable by hand."""
    return nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Conv2d(4, 4, 3))


def hand_macs() -> int:
    """(1, 1, 8, 8) -> conv1 -> (1, 4, 6, 6) -> conv2 -> (1, 4, 4, 4).

    conv1: 4*6*6 outputs, 1*3*3 taps each  ->  144 * 9
    conv2: 4*4*4 outputs, 4*3*3 taps each  ->   64 * 36
    """
    return 4 * 6 * 6 * 1 * 3 * 3 + 4 * 4 * 4 * 4 * 3 * 3


def test_macs_match_the_hand_count():
    model = tiny_model()
    out = count_macs(model, SHAPE)
    assert out["macs"] == hand_macs() == 3600
    assert out["input_shape"] == SHAPE


def test_flops_are_two_macs():
    out = count_macs(tiny_model(), SHAPE)
    assert out["flops"] == 2 * out["macs"]


def test_params_match_the_module_sum():
    model = tiny_model()
    out = count_macs(model, SHAPE)
    assert out["params"] == sum(p.numel() for p in model.parameters()) == 188


def test_by_module_splits_the_total():
    out = count_macs(tiny_model(), SHAPE)
    assert out["by_module"]["Conv2d"] == hand_macs()
    assert sum(out["by_module"].values()) == out["macs"]


def test_counting_twice_does_not_accumulate_hooks():
    """A leftover hook would report 2x on the second call, not the same number."""
    model = tiny_model()
    first = count_macs(model, SHAPE)
    second = count_macs(model, SHAPE)
    assert first["macs"] == second["macs"]
    assert second["params"] == first["params"]


def test_count_macs_leaves_the_model_in_its_mode():
    model = tiny_model()
    model.train()
    count_macs(model, SHAPE)
    assert model.training is True


class _SelfAttention(nn.Module):
    """``nn.MultiheadAttention`` is not a standalone model: it needs q, k, v."""

    def __init__(self, embed: int = 8, heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x)[0]


def test_batchnorm_and_attention_are_counted_not_ignored():
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
    out = count_macs(model, SHAPE)
    conv = 4 * 6 * 6 * 1 * 3 * 3
    assert out["by_module"]["BatchNorm"] == 2 * (1 * 4 * 6 * 6)  # 2 MAC/elem
    assert out["macs"] == conv + 2 * (1 * 4 * 6 * 6)

    attn = _SelfAttention()
    seq = (1, 5, 8)
    # 4 * L * E^2 (three input projections + one output) + 2 * L^2 * E
    expected = 4 * 5 * 8 * 8 + 2 * 5 * 5 * 8
    got = count_macs(attn, seq)
    assert got["macs"] == pytest.approx(expected, rel=1e-9)
    assert got["by_module"]["MultiheadAttention"] == pytest.approx(expected,
                                                                  rel=1e-9)


def test_measure_latency_reports_the_documented_keys():
    model = tiny_model()
    out = measure_latency(model, torch.zeros(SHAPE), device="cpu",
                          warmup=1, iters=5)
    for key in ("median_ms", "mean_ms", "p10_ms", "p90_ms", "fps", "n_iters",
                "peak_memory_mb", "device"):
        assert key in out
    assert out["n_iters"] == 5
    assert out["device"] == "cpu"
    assert out["peak_memory_mb"] is None            # meaningless off-GPU
    assert out["p10_ms"] <= out["median_ms"] <= out["p90_ms"]
    assert out["median_ms"] >= 0.0


def test_measure_latency_does_not_change_the_mode():
    model = tiny_model()
    model.train()
    measure_latency(model, torch.zeros(SHAPE), device="cpu", warmup=1, iters=2)
    assert model.training is True


def test_throughput_scales_with_slices_and_tiles():
    model = tiny_model()
    vol = throughput_for_volume(model, (4, 8, 8), in_channels=1, patch=8,
                                device="cpu", warmup=1, iters=3)
    assert vol["patches_per_slice"] == 1            # 8/8 * 8/8
    assert vol["n_slices"] == 4
    assert vol["volume_s"] == pytest.approx(vol["slice_ms"] * 4 / 1000.0)
    # A 16x16 plane at patch 8 is four tiles, so a slice costs four patches.
    tiled = throughput_for_volume(model, (2, 16, 16), in_channels=1, patch=8,
                                  device="cpu", warmup=1, iters=3)
    assert tiled["patches_per_slice"] == 4
    assert tiled["slice_ms"] == pytest.approx(tiled["patch_latency_ms"] * 4)


def test_complexity_report_is_finite_on_cpu():
    model = tiny_model()
    rep = complexity_report(model, 1, patch=8, n_slices=3, device="cpu",
                            warmup=1, iters=3)
    assert rep["device"] == "cpu" and rep["is_cpu"] is True
    assert rep["params"] == 188
    assert rep["macs_per_slice"] == hand_macs()
    assert rep["macs_per_volume"] == hand_macs() * 3
    assert rep["flops_per_volume"] == 2 * rep["macs_per_volume"]
    for key in ("params_m", "latency_ms_per_slice", "sec_per_slice",
                "sec_per_volume"):
        assert math.isfinite(rep[key])
    assert rep["sec_per_volume"] == pytest.approx(
        rep["latency_ms_per_slice"] / 1000.0 * 3, rel=1e-3)


def test_cuda_request_falls_back_to_cpu_without_a_gpu():
    """"Run on cpu" must work on a machine with no GPU at all, not raise."""
    out = complexity_report(tiny_model(), 1, patch=8, n_slices=2,
                            device="cuda", warmup=1, iters=2)
    assert out["device"] in ("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2",
                             "cuda:3")
    assert math.isfinite(out["sec_per_volume"])
