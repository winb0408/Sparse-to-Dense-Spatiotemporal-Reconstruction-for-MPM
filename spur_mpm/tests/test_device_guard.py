"""The device guard: a silent CPU fallback must become a loud failure.

2026-11-14: `CUDA_VISIBLE_DEVICES=3` was set, the launch gate had just seen the
card, and the floor evaluation before it had finished on that card -- but by the
time the training arm started, `torch.cuda.is_available()` was False. The arm
trained on the CPU at ~22 s/step and wrote `"gpus": []` into its provenance.
Nothing failed. These tests hold the line that such a run now refuses to start.
"""

from __future__ import annotations

import pytest
import torch

from spur_mpm.device import OVERRIDE_ENV, resolve_device


def _cfg(device: str) -> dict:
    return {"run": {"device": device}}


def test_a_missing_cuda_device_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    with pytest.raises(SystemExit) as err:
        resolve_device(_cfg("cuda"), purpose="training")

    message = str(err.value)
    assert "training" in message
    assert "CUDA_VISIBLE_DEVICES=3" in message
    assert OVERRIDE_ENV in message, "the override must be discoverable"


def test_the_error_names_the_incident_so_the_next_person_can_find_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)

    with pytest.raises(SystemExit) as err:
        resolve_device(_cfg("cuda:1"), purpose="evaluation")

    assert "gpu2_falloff_20261113" in str(err.value)


def test_the_override_is_explicit_and_lands_on_the_cpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv(OVERRIDE_ENV, "1")

    device = resolve_device(_cfg("cuda"), purpose="training")

    assert device.type == "cpu"
    assert OVERRIDE_ENV in capsys.readouterr().out, (
        "an overridden CPU run has to say so in its log")


def test_a_deliberate_cpu_config_is_not_an_accident(monkeypatch):
    # `run.device: cpu` is a decision in the config, not a fallback, and must
    # keep working even when the override variable is absent.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)

    assert resolve_device(_cfg("cpu"), purpose="training").type == "cpu"


def test_a_usable_cuda_device_is_returned_as_requested(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv(OVERRIDE_ENV, "1")  # must not force the CPU when CUDA is fine

    assert resolve_device(_cfg("cuda")) == torch.device("cuda")
    assert resolve_device(_cfg("cuda:2")) == torch.device("cuda:2")


def test_a_missing_run_section_defaults_to_cuda_and_is_guarded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(OVERRIDE_ENV, raising=False)

    with pytest.raises(SystemExit):
        resolve_device({}, purpose="training")
