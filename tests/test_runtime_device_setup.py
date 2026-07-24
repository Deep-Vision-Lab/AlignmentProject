import os

import pytest

from runtime_device_setup import isolate_local_rank_cuda_device


@pytest.mark.parametrize(
    ("local_rank", "expected"),
    [(0, "0"), (1, "2")],
)
def test_non_contiguous_slurm_gpu_mapping(monkeypatch, local_rank, expected):
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")

    selection = isolate_local_rank_cuda_device()

    assert selection.original_visible_devices == "0,2"
    assert selection.selected_device == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_local_rank_without_explicit_visibility(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    selection = isolate_local_rank_cuda_device()

    assert selection.selected_device == "1"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_invalid_rank_fails_before_cuda_import(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")

    with pytest.raises(RuntimeError, match="LOCAL_RANK=2"):
        isolate_local_rank_cuda_device()
