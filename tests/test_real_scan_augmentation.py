from PIL import Image
import numpy as np

from real_scan_augmentation import ScanOnlyAugmentor


def test_scan_only_augmentation_preserves_geometry_and_changes_pixels():
    image = Image.fromarray(
        np.tile(np.arange(96, dtype=np.uint8)[None, :, None], (48, 1, 3)),
        mode="RGB",
    )
    augmentor = ScanOnlyAugmentor(
        enabled=True,
        probability=1.0,
        brightness_delta=0.0,
        contrast_delta=0.0,
        blur_probability=1.0,
        blur_radius_min=0.7,
        blur_radius_max=0.7,
        gaussian_noise_probability=1.0,
        gaussian_noise_std_min=4.0,
        gaussian_noise_std_max=4.0,
        speckle_probability=0.0,
    )
    np.random.seed(7)
    output, metadata = augmentor.augment_with_metadata(image)

    assert output.size == image.size
    assert metadata["geometry_changed"] is False
    assert metadata["rotation_degrees"] == 0.0
    assert metadata["translation_x"] == 0
    assert metadata["translation_y"] == 0
    assert metadata["scale"] == 1.0
    assert not np.array_equal(np.asarray(output), np.asarray(image))


def test_disabled_scan_augmentation_is_pixel_exact():
    image = Image.new("RGB", (64, 32), (211, 197, 173))
    augmentor = ScanOnlyAugmentor(enabled=False)
    output, metadata = augmentor.augment_with_metadata(image)
    assert output.size == image.size
    assert np.array_equal(np.asarray(output), np.asarray(image))
    assert metadata["geometry_changed"] is False
