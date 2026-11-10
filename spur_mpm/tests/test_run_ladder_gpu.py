"""The GPU guard: "free" must mean nobody's job is on the card.

`_free_gpus` decides which cards a long training run may take when the caller
did not pass ``--devices``. On a shared server the dangerous mistake is not
"refused to run" but "quietly took a card someone else is using", so every
path that cannot answer the question has to come back empty and let the caller
refuse. Free *memory* is not the test: a card can have 24 GB unused while a
training job on it grows into that space later.
"""
import subprocess

import pytest

from spur_mpm.scripts.run_ladder import _free_gpus


class _FakeRun:
    """Stands in for ``subprocess.run``: answers by query, not by call order."""

    def __init__(self, gpus, compute_apps, fail_on=()):
        self.gpus = gpus                  # [(index, mem_used, uuid)]
        self.compute_apps = compute_apps  # [uuid, ...]
        self.fail_on = set(fail_on)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        flags = " ".join(cmd)
        self.calls.append(flags)
        for name in self.fail_on:
            if name in flags:
                raise subprocess.CalledProcessError(1, cmd)
        if "query-compute-apps" in flags:
            out = "\n".join(self.compute_apps)
        elif "query-gpu=index,memory.used" in flags:
            out = "\n".join(f"{i},{m}" for i, m, _ in self.gpus)
        else:
            out = "\n".join(f"{i},{u}" for i, _, u in self.gpus)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


_TWO_FREE = [(0, 9000, "GPU-a"), (1, 8000, "GPU-b"), (2, 3000, "GPU-c")]


def test_a_card_with_a_compute_process_is_never_offered():
    run = _FakeRun(_TWO_FREE, compute_apps=["GPU-b"])
    assert _free_gpus(run) == ["2", "0"]        # 2 is emptiest, then 0


def test_free_memory_alone_does_not_make_a_card_free():
    """600 MiB used and 24 GB spare is still not ours if a job is on it."""
    run = _FakeRun([(0, 600, "GPU-a")], compute_apps=["GPU-a"])
    assert _free_gpus(run) == []


def test_candidates_are_ordered_by_memory_so_the_emptiest_goes_first():
    run = _FakeRun(_TWO_FREE, compute_apps=[])
    assert _free_gpus(run) == ["2", "1", "0"]


def test_no_nvidia_smi_means_nothing_is_free_rather_than_gpu_zero():
    def missing(cmd, **kwargs):
        raise OSError("nvidia-smi: not found")
    assert _free_gpus(missing) == []


def test_a_failing_smi_query_is_not_read_as_an_empty_machine():
    run = _FakeRun(_TWO_FREE, [], fail_on=["query-gpu=index,memory.used"])
    assert _free_gpus(run) == []


def test_a_running_job_we_cannot_place_makes_every_card_suspect():
    """The old code kept the unmappable uuid and excluded nothing."""
    run = _FakeRun(_TWO_FREE, compute_apps=["GPU-from-another-node"])
    assert _free_gpus(run) == []


def test_cannot_enumerate_processes_is_a_refusal_not_an_optimistic_answer():
    run = _FakeRun(_TWO_FREE, [], fail_on=["query-compute-apps"])
    assert _free_gpus(run) == []


def test_the_guard_asks_for_the_fields_it_parses():
    run = _FakeRun(_TWO_FREE, compute_apps=["GPU-b"])
    _free_gpus(run)
    joined = " | ".join(run.calls)
    assert "query-gpu=index,memory.used" in joined
    assert "query-compute-apps=gpu_uuid" in joined
    assert "query-gpu=index,uuid" in joined


def test_running_out_of_candidates_is_reported_as_an_empty_list():
    run = _FakeRun(_TWO_FREE, compute_apps=["GPU-a", "GPU-b", "GPU-c"])
    assert _free_gpus(run) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
