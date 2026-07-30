import numpy as np
from PIL import Image

from Evaluation.sw_core import (
    alignment_region_metrics,
    dense_alignment_region,
    synthetic_mask_region_metrics,
)


def test_dense_region_fills_internal_warp_windows():
    path = [(2, 10), (3, 11), (5, 12), (6, 13)]
    # Four diagonal correspondences plus one horizontal/vertical warp transition.
    traceback = np.zeros((6, 2), dtype=np.float32)

    region = dense_alignment_region(path, traceback)
    metrics = alignment_region_metrics(path, traceback, (8, 20))

    assert region.line1_start == 2
    assert region.line1_end == 6
    assert region.line1_span_windows == 5
    assert region.line2_start == 10
    assert region.line2_end == 13
    assert region.line2_span_windows == 4
    assert region.warp_steps == 1

    assert metrics["line1_path_windows"] == 4
    assert metrics["line1_span_windows"] == 5
    assert metrics["line1_path_fraction"] == 4 / 8
    assert metrics["line1_matched_fraction"] == 5 / 8


def test_empty_path_has_no_dense_region():
    traceback = np.zeros((1, 2), dtype=np.float32)
    region = dense_alignment_region([], traceback)
    metrics = alignment_region_metrics([], traceback, (10, 12))

    assert region.empty
    assert region.line1_span_windows == 0
    assert region.line2_span_windows == 0
    assert metrics["line1_path_start"] == -1
    assert metrics["line2_path_end"] == -1
    assert metrics["line1_matched_fraction"] == 0.0
    assert metrics["line2_matched_fraction"] == 0.0


def test_synthetic_mask_metrics_compare_dense_span(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    image1 = images / "img1_1.png"
    image2 = images / "img2_1.png"
    Image.fromarray(np.zeros((4, 10, 3), dtype=np.uint8)).save(image1)
    Image.fromarray(np.zeros((4, 10, 3), dtype=np.uint8)).save(image2)

    # The dense path covers windows [2, 6), which maps exactly to pixels [2, 6)
    # for ten windows over a ten-pixel image.
    mask = np.zeros((4, 10), dtype=np.uint8)
    mask[:, 2:6] = 255
    Image.fromarray(mask).save(masks / "mask1_1.png")
    Image.fromarray(mask).save(masks / "mask2_1.png")

    path = [(2, 2), (3, 3), (5, 5)]
    traceback = np.zeros((5, 2), dtype=np.float32)
    metrics = synthetic_mask_region_metrics(
        path,
        traceback,
        (10, 10),
        image1,
        image2,
        image_width1=10,
        image_width2=10,
        use_flip=False,
    )

    assert metrics["line1_pred_start_px"] == 2.0
    assert metrics["line1_pred_end_px"] == 6.0
    assert metrics["line1_region_iou"] == 1.0
    assert metrics["line2_region_iou"] == 1.0
    assert metrics["mean_region_iou"] == 1.0
