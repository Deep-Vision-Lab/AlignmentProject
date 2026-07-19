import unittest

import torch

from embeddingModel import window_ink_ratio_from_patches


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def imagenet_normalize(patches):
    return (patches - IMAGENET_MEAN) / IMAGENET_STD


class InkRatioPolarityTest(unittest.TestCase):
    def _make_patch(self, background, foreground):
        patch = torch.full((1, 1, 3, 128, 32), float(background))
        # A centered vertical stroke occupying 25% of the window width.
        patch[:, :, :, :, 12:20] = float(foreground)
        return imagenet_normalize(patch)

    def test_black_and_white_polarities_match(self):
        black_text_on_white = self._make_patch(background=1.0, foreground=0.0)
        white_text_on_black = self._make_patch(background=0.0, foreground=1.0)

        dark_ratio = window_ink_ratio_from_patches(black_text_on_white)
        light_ratio = window_ink_ratio_from_patches(white_text_on_black)

        self.assertTrue(torch.allclose(dark_ratio, light_ratio, atol=1e-6))
        self.assertAlmostEqual(float(dark_ratio.item()), 0.25, places=5)

    def test_blank_windows_have_zero_ink(self):
        white_blank = imagenet_normalize(torch.ones(1, 1, 3, 128, 32))
        black_blank = imagenet_normalize(torch.zeros(1, 1, 3, 128, 32))

        self.assertEqual(float(window_ink_ratio_from_patches(white_blank).item()), 0.0)
        self.assertEqual(float(window_ink_ratio_from_patches(black_blank).item()), 0.0)


if __name__ == "__main__":
    unittest.main()
