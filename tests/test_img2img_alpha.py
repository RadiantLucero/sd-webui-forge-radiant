import unittest
from types import SimpleNamespace

from PIL import Image

from modules.img2img_alpha import (
    background_color,
    composite,
    reconstruct,
    validate_native_vae,
)
from modules.masking import fill


class Img2ImgAlphaTests(unittest.TestCase):
    def test_native_vae_error_reports_detected_channels_and_model(self):
        encoder = SimpleNamespace(conv1=SimpleNamespace(in_channels=3))
        vae = SimpleNamespace(first_stage_model=SimpleNamespace(encoder=encoder), output_channels=4)
        with self.assertRaises(RuntimeError) as error:
            validate_native_vae(vae)
        message = str(error.exception)
        self.assertIn("VAE model: types.SimpleNamespace", message)
        self.assertIn("encoder.conv1: types.SimpleNamespace", message)
        self.assertIn("encoder input channels: 3", message)
        self.assertIn("VAE output channels: 4", message)

    def test_native_vae_error_reports_missing_encoder(self):
        vae = SimpleNamespace(first_stage_model=SimpleNamespace(), output_channels=4)
        with self.assertRaisesRegex(RuntimeError, "encoder.conv1: unavailable; encoder input channels: unavailable"):
            validate_native_vae(vae)

    def test_native_vae_checks_output_channels(self):
        encoder = SimpleNamespace(conv1=SimpleNamespace(in_channels=4))
        vae = SimpleNamespace(first_stage_model=SimpleNamespace(encoder=encoder), output_channels=3)
        with self.assertRaisesRegex(RuntimeError, "encoder input channels: 4; VAE output channels: 3"):
            validate_native_vae(vae)
        vae.output_channels = 4
        validate_native_vae(vae)

    def test_unchanged_original_alpha_and_green_object_are_preserved(self):
        source = Image.new("RGBA", (2, 1))
        source.putdata([(200, 50, 40, 128), (0, 255, 0, 255)])
        rendered = Image.new("RGB", (2, 1))
        rendered.putdata([(100, 152, 20), (0, 255, 0)])
        result = reconstruct(rendered, source, (0, 255, 0))
        self.assertEqual(result.tobytes(), source.tobytes())

    def test_reconstructs_color_over_arbitrary_background(self):
        source = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        rendered = Image.new("RGB", (1, 1), (128, 64, 64))
        result = reconstruct(rendered, source, (0, 128, 128))
        r, g, b, alpha = result.getpixel((0, 0))
        self.assertEqual(r, 255)
        self.assertLessEqual(max(g, b), 1)
        self.assertEqual(alpha, 128)

    def test_reconstruction_uses_original_color_to_avoid_saturation(self):
        source = Image.new("RGBA", (1, 1), (200, 50, 40, 128))
        target = Image.new("RGBA", (1, 1), (220, 50, 40, 128))
        for background in ((0, 255, 0), (0, 255, 255), (128, 128, 128)):
            with self.subTest(background=background):
                rendered = Image.alpha_composite(Image.new("RGBA", (1, 1), background), target).convert("RGB")
                result = reconstruct(rendered, source, background)
                for actual, expected in zip(result.getpixel((0, 0)), target.getpixel((0, 0))):
                    self.assertLessEqual(abs(actual - expected), 3)

    def test_small_edits_to_semitransparent_pixels_are_not_discarded(self):
        source = Image.new("RGBA", (1, 1), (200, 50, 40, 128))
        target = Image.new("RGBA", (1, 1), (208, 50, 40, 128))
        background = (0, 255, 0)
        rendered = Image.alpha_composite(Image.new("RGBA", (1, 1), background), target).convert("RGB")
        result = reconstruct(rendered, source, background)
        self.assertGreater(result.getpixel((0, 0))[0], source.getpixel((0, 0))[0])

    def test_reconstruction_can_detect_changed_alpha(self):
        source = Image.new("RGBA", (1, 1), (200, 50, 40, 128))
        target = Image.new("RGBA", (1, 1), (200, 50, 40, 64))
        background = (0, 255, 0)
        rendered = Image.alpha_composite(Image.new("RGBA", (1, 1), background), target).convert("RGB")
        result = reconstruct(rendered, source, background)
        for actual, expected in zip(result.getpixel((0, 0)), target.getpixel((0, 0))):
            self.assertLessEqual(abs(actual - expected), 3)

    def test_reconstruction_does_not_replace_strongly_edited_colors(self):
        source = Image.new("RGBA", (1, 1), (200, 50, 40, 128))
        rendered = Image.new("RGB", (1, 1), (10, 227, 95))
        result = reconstruct(rendered, source, (0, 255, 0))
        recomposited = Image.alpha_composite(Image.new("RGBA", (1, 1), (0, 255, 0)), result).convert("RGB")
        for actual, expected in zip(recomposited.getpixel((0, 0)), rendered.getpixel((0, 0))):
            self.assertLessEqual(abs(actual - expected), 1)

    def test_edited_opaque_source_is_not_made_translucent(self):
        source = Image.new("RGBA", (1, 1), (100, 100, 100, 255))
        rendered = Image.new("RGB", (1, 1), (200, 150, 130))
        self.assertEqual(reconstruct(rendered, source, (128, 128, 128)).getpixel((0, 0)), (200, 150, 130, 255))

    def test_inpaint_mask_preserves_original_and_blends_alpha(self):
        original = Image.new("RGBA", (3, 1), (255, 0, 0, 128))
        generated = Image.new("RGBA", (3, 1), (0, 0, 255, 255))
        mask = Image.new("L", (3, 1))
        mask.putdata([0, 128, 255])
        result = composite(generated, original, mask)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0, 128))
        self.assertEqual(result.getpixel((2, 0)), (0, 0, 255, 255))
        self.assertEqual(result.getpixel((1, 0)), (85, 0, 170, 192))

    def test_automatic_background_ignores_fully_transparent_pixels(self):
        source = Image.new("RGBA", (2, 1))
        source.putdata([(0, 0, 0, 255), (255, 255, 255, 0)])
        self.assertEqual(background_color(source, "#808080", True), (255, 255, 255))
        self.assertEqual(background_color(source, "#123456", False), (18, 52, 86))

    def test_mismatched_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching image sizes"):
            reconstruct(Image.new("RGB", (1, 1)), Image.new("RGBA", (2, 1)), (0, 0, 0))

    def test_inpainting_fill_keeps_unmasked_alpha(self):
        source = Image.new("RGBA", (2, 2), (200, 30, 40, 80))
        result = fill(source, Image.new("L", source.size, 0), preserve_alpha=True)
        self.assertEqual(result.tobytes(), source.tobytes())


if __name__ == "__main__":
    unittest.main()
