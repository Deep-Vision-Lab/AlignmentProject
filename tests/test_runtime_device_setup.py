import os
import subprocess
from pathlib import Path

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


@pytest.mark.parametrize(
    ("visible_name", "visible_value", "local_rank", "expected"),
    [
        ("CUDA_VISIBLE_DEVICES", "0,2", "0", "0"),
        ("CUDA_VISIBLE_DEVICES", "0,2", "1", "2"),
        ("SLURM_GPU_INDEX", "1-2", "0", "1"),
        ("SLURM_GPU_INDEX", "1-2", "1", "2"),
    ],
)
def test_shell_rank_wrapper_selects_one_device(
    visible_name, visible_value, local_rank, expected
):
    project_root = Path(__file__).resolve().parents[1]
    wrapper = project_root / "scripts" / "train" / "run_rank_isolated.sh"
    env = os.environ.copy()
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "SLURM_GPU_INDEX",
    ):
        env.pop(name, None)
    env.update(
        {
            visible_name: visible_value,
            "LOCAL_RANK": local_rank,
            "LOCAL_WORLD_SIZE": "2",
            "WORLD_SIZE": "2",
            "RANK_WRAPPER_DRY_RUN": "1",
        }
    )

    completed = subprocess.run(
        ["bash", str(wrapper)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"selected={expected}" in completed.stderr
    assert f"process_visible={expected}" in completed.stderr
