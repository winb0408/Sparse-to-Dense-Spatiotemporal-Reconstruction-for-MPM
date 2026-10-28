"""What a self-supervised arm is allowed to see while it is being selected.

The strict accounting in plan §2.3 is easy to satisfy on paper and easy to
break in code: a single `batch["target"]` inside the validation loop turns an
SS arm into a supervised one wearing a different label. These tests fail if
that ever happens.
"""

from __future__ import annotations

import pytest
import torch

from spur_mpm.train import validate


class _NoDense(dict):
    """A batch that screams if anyone asks for the dense reference."""

    def __getitem__(self, key):
        if key == "target":
            raise AssertionError("self-supervised validation read the dense "
                                 "reference")
        return super().__getitem__(key)


class _ZeroModel(torch.nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.channels = channels
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        return self.dummy + torch.zeros(n, self.channels, h, w,
                                        dtype=x.dtype, device=x.device)


def _batch(holdout: bool = True) -> _NoDense:
    b, h, w = 2, 8, 8
    hold = torch.zeros(b, h, w)
    if holdout:
        hold[:, ::2] = 1.0
    return _NoDense(
        inputs=torch.zeros(b, 1, h, w),
        center_index=torch.zeros(b, dtype=torch.long),
        target=torch.zeros(b, h, w),
        hold_target=torch.full((b, h, w), 0.4),
        hold_mask=hold,
    )


def test_self_supervised_validation_never_reads_the_dense_reference():
    out = validate(_ZeroModel(), [_batch()], torch.device("cpu"),
                   supervision="self", rho="charbonnier", amp=False)
    assert out["select_metric"] == "holdout_loss"
    assert out["select_maximise"] is False
    assert "dense_psnr" not in out, "an SS arm must not even compute it"
    # charbonnier leaves sqrt(eps) of residual floor, hence 0.4 - 0.001
    assert out["holdout_loss"] == pytest.approx(0.4, abs=2e-3)


def test_supervised_validation_does_read_it_and_selects_on_psnr():
    # a plain dict here: a supervised arm is *allowed* to read the target
    out = validate(_ZeroModel(), [dict(_batch())], torch.device("cpu"),
                   supervision="dense", rho="charbonnier", amp=False)
    assert out["select_metric"] == "dense_psnr"
    assert out["select_maximise"] is True
    assert "dense_psnr" in out


def test_self_supervised_arm_refuses_to_select_without_a_holdout():
    with pytest.raises(RuntimeError, match="held-out measurement loss"):
        validate(_ZeroModel(), [_batch(holdout=False)], torch.device("cpu"),
                 supervision="self", rho="charbonnier", amp=False)
