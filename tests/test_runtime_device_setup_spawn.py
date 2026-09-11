from types import SimpleNamespace

import pytest

import runtime_device_setup as runtime


def _fake_process(name):
    return SimpleNamespace(name=name)


def test_spawned_dataloader_worker_inherits_single_rank_gpu(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ORIGINAL_CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("RANK_WRAPPER_ISOLATED", raising=False)
    monkeypatch.delenv("RANK_SELECTED_CUDA_DEVICE", raising=False)
    monkeypatch.setattr(
        runtime.multiprocessing,
        "current_process",
        lambda: _fake_process("SpawnProcess-3"),
    )

    selected = runtime.isolate_local_rank_cuda_device()

    assert selected.local_rank == 1
    assert selected.world_size == 2
    assert selected.original_visible_devices == "0,1"
    assert selected.selected_device == "1"
    assert runtime.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_main_rank_still_rejects_unverified_single_gpu(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("ORIGINAL_CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("RANK_WRAPPER_ISOLATED", raising=False)
    monkeypatch.delenv("RANK_SELECTED_CUDA_DEVICE", raising=False)
    monkeypatch.setattr(
        runtime.multiprocessing,
        "current_process",
        lambda: _fake_process("MainProcess"),
    )

    with pytest.raises(RuntimeError, match="without verified per-rank isolation"):
        runtime.isolate_local_rank_cuda_device()
