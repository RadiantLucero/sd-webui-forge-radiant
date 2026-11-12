import sys
import types
import unittest
from unittest.mock import patch

import torch

from backend.misc.image_resize import resize_reference


class ReferenceResizeTests(unittest.TestCase):
    def resize(self, height, width, max_side, resolution=0):
        shared = types.ModuleType("modules.shared")
        shared.opts = types.SimpleNamespace(reference_max_side=max_side)
        samples = torch.zeros((1, 3, height, width))
        with patch.dict(sys.modules, {"modules.shared": shared}):
            return resize_reference(samples, resolution=resolution)

    def test_max_side_limits_the_long_edge(self):
        resized = self.resize(2000, 1000, 1024)
        self.assertEqual(resized.shape[-2:], (1024, 512))

    def test_max_side_does_not_upscale_small_references(self):
        resized = self.resize(512, 512, 1024)
        self.assertEqual(resized.shape[-2:], (512, 512))

    def test_resolution_target_can_upscale_within_the_max_side(self):
        resized = self.resize(512, 512, 1024, resolution=1024)
        self.assertEqual(resized.shape[-2:], (1024, 1024))


if __name__ == "__main__":
    unittest.main()
