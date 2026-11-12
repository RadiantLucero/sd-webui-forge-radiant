import unittest
import warnings
from types import SimpleNamespace

import numpy as np
from PIL import Image

from modules.img2img_alpha import (
    AlphaSettings,
    ChromaKey,
    ReconstructionCache,
    ReconstructionCacheItem,
    background_color,
    composite,
    reconstruct,
    validate_native_vae,
)
from modules.masking import fill

GREEN = (0, 255, 0)
KEY = (20, 230, 15)
SUBJECT = (190, 140, 110)


def scene(width=32, height=16):
    """Subject on the left half, transparent right half, and the matching generated RGB result."""
    source = np.zeros((height, width, 4), np.uint8)
    source[:, : width // 2] = (*SUBJECT, 255)
    rendered = np.empty((height, width, 3), np.uint8)
    rendered[:] = KEY
    rendered[:, : width // 2] = SUBJECT
    return source, rendered


def run(source, rendered, **settings):
    result, calibration = reconstruct(Image.fromarray(rendered, "RGB"), Image.fromarray(source, "RGBA"), GREEN, AlphaSettings(mode="Reconstruct", **settings))
    return np.asarray(result).astype(int), calibration


def shadow_scene():
    """Subject on the left, a dark translucent shadow with opaque spots in the middle, and background on the right."""
    source = np.zeros((24, 64, 4), np.uint8)
    source[:, :16] = (*SUBJECT, 255)
    source[:, 16:56] = (12, 12, 16, 199)
    rendered = np.empty((24, 64, 3), np.uint8)
    rendered[:] = KEY
    rendered[:, :16] = SUBJECT
    rendered[:, 16:56] = np.round(0.78 * np.array((12, 12, 16)) + 0.22 * np.array(KEY))
    # A subject edge with slight key dominance borders the shadow on one side only.
    rendered[:, 15] = (100, 112, 100)
    spots = ((4, 24), (12, 36), (20, 46))
    for spot in spots:
        rendered[spot] = (10, 22, 10)
    rendered[10:12, 28:30] = (20, 20, 20)
    return source, rendered, spots


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

    def test_new_content_on_former_background_stays_opaque_and_unchanged(self):
        source, rendered = scene()
        rendered[2:6, 20:24] = (120, 120, 120)
        rendered[8:12, 20:24] = (180, 120, 100)
        result, _ = run(source, rendered)
        self.assertTrue((result[2:6, 20:24] == (120, 120, 120, 255)).all())
        self.assertTrue((result[8:12, 20:24] == (180, 120, 100, 255)).all())
        self.assertTrue((result[:, :16] == (*SUBJECT, 255)).all())
        self.assertTrue((result[:, 28:, 3] == 0).all())

    def test_key_mixed_edge_is_translucent_and_despilled_without_magenta(self):
        source, rendered = scene()
        rendered[:, 16] = np.round((np.array(SUBJECT) + np.array(KEY)) / 2)
        result, _ = run(source, rendered)
        edge = result[:, 16]
        self.assertTrue(((edge[:, 3] > 0) & (edge[:, 3] < 255)).all())
        self.assertTrue((ChromaKey(GREEN).strength(edge[:, :3]) < 30).all())
        self.assertTrue((edge[:, 1] >= np.minimum(edge[:, 0], edge[:, 2])).all())

    def test_tinted_result_stays_opaque(self):
        source, rendered = scene()
        rendered[:, :16] = (150, 170, 140)
        result, calibration = run(source, rendered)
        self.assertEqual(calibration.tint, 70)
        self.assertGreater(calibration.opaque_threshold, 20)
        self.assertTrue((result[:, :16] == (150, 170, 140, 255)).all())

    def test_manual_thresholds_and_shift_override_calibration(self):
        source, rendered = scene()
        rendered[:, 16] = (65, 195, 50)
        result, calibration = run(source, rendered, opaque_threshold=100, transparent_threshold=150, color_shift=200)
        self.assertEqual((calibration.opaque_threshold, calibration.transparent_threshold, calibration.color_shift), (100, 150, 200))
        self.assertEqual(calibration.infotext()["img2img alpha opaque"], "100")
        # Dominance 130 of 200 leaves alpha 0.35; the rendered key share is removed from the color.
        self.assertTrue((result[:, 16] == (149, 130, 115, 89)).all())

    def test_translucent_areas_are_unmixed_with_auto_thresholds(self):
        source = np.zeros((24, 64, 4), np.uint8)
        source[:, :16] = (*SUBJECT, 255)
        rendered = np.empty((24, 64, 3), np.uint8)
        rendered[:] = KEY
        rendered[:, :16] = SUBJECT
        # Uniform translucent bands outnumber both the opaque subject and the background.
        bands = (((235, 235, 235), 0.75), ((235, 235, 235), 0.5), ((50, 50, 60), 0.25))
        for index, (color, alpha) in enumerate(bands):
            rows = slice(index * 8, index * 8 + 8)
            source[rows, 16:56] = (*color, round(alpha * 255))
            rendered[rows, 16:56] = np.round(alpha * np.array(color) + (1 - alpha) * np.array(KEY))
        result, calibration = run(source, rendered)
        self.assertLess(calibration.opaque_threshold, 30)
        self.assertEqual(calibration.background_color, KEY)
        for index, (color, alpha) in enumerate(bands):
            with self.subTest(color=color, alpha=alpha):
                pixels = result[index * 8 : index * 8 + 8, 16:56].reshape(-1, 4)
                self.assertTrue((np.abs(pixels[:, 3] - alpha * 255) <= 10).all())
                coverage = pixels[:, 3:] / 255
                over_white = pixels[:, :3] * coverage + 255 * (1 - coverage)
                self.assertTrue((np.abs(over_white - (alpha * np.array(color) + (1 - alpha) * 255)) <= 8).all())

    def test_island_check_unmixes_opaque_spots_inside_translucent_areas(self):
        source, rendered, spots = shadow_scene()
        plain, calibration = run(source, rendered)
        self.assertGreater(calibration.opaque_threshold, 12)
        self.assertTrue(all((plain[spot] == (10, 22, 10, 255)).all() for spot in spots))

        result, calibration = run(source, rendered, island_check=True, island_smoke=False)
        self.assertEqual(calibration.island_pixels, len(spots))
        self.assertEqual(calibration.infotext()["img2img alpha island"], "0 auto")
        self.assertEqual(calibration.infotext()["img2img alpha island smoke"], "off")
        for spot in spots:
            with self.subTest(spot=spot):
                self.assertTrue(200 < result[spot][3] < 255)
                self.assertLessEqual(ChromaKey(GREEN).strength(result[spot][:3]), 1)
        self.assertTrue((result[10:12, 28:30] == (20, 20, 20, 255)).all())
        self.assertTrue((result[:, :16] == plain[:, :16]).all())

    def test_island_smoke_takes_alpha_from_surrounding_smoke(self):
        source, rendered, spots = shadow_scene()
        rendered[20, 30] = (250, 250, 250)
        rendered[16, 40] = SUBJECT
        rendered[:, 60] = (30, 30, 30)
        plain, _ = run(source, rendered, island_check=True, island_smoke=False)
        result, calibration = run(source, rendered, island_check=True)

        # Spots and the neutral dark patch take the shadow's alpha and lose the dimmed key without turning magenta.
        smoke_alpha = plain[6, 32, 3]
        taken = [*spots, (10, 28), (10, 29), (11, 28), (11, 29)]
        for pixel in taken:
            with self.subTest(pixel=pixel):
                self.assertLessEqual(abs(result[pixel][3] - smoke_alpha), 2)
                self.assertLessEqual(np.ptp(result[pixel][:3]), 1)
        self.assertEqual((calibration.island_pixels, calibration.island_smoke_pixels), (0, len(taken) + 1))
        self.assertEqual(calibration.infotext()["img2img alpha island smoke"], "on")
        self.assertIn("8 from smoke", calibration.statistics())
        # Brightness limits transparency; colors leaning away from the key, the subject edge, and strands on empty background stay opaque.
        self.assertGreaterEqual(result[20, 30, 3], 245)
        self.assertTrue((result[16, 40] == (*SUBJECT, 255)).all())
        self.assertTrue((result[:, :16] == plain[:, :16]).all())
        self.assertTrue((result[:, 60] == (30, 30, 30, 255)).all())

    def test_original_alpha_contributes_only_statistics(self):
        source, rendered = scene()
        rendered[:, 16] = (100, 190, 60)
        shuffled = source.reshape(-1, 4).copy()
        np.random.default_rng(1).shuffle(shuffled)
        first, _ = run(source, rendered)
        second, _ = run(shuffled.reshape(source.shape), rendered)
        self.assertTrue((first == second).all())

    def test_invalid_settings_are_rejected(self):
        for settings in (
            AlphaSettings(mode="Reconstruct", key_color="#808080"),
            AlphaSettings(mode="Reconstruct", key_color="#ffffff"),
            AlphaSettings(mode="Reconstruct", opaque_threshold=120, transparent_threshold=120),
            AlphaSettings(mode="Reconstruct", opaque_threshold=255),
            AlphaSettings(mode="Reconstruct", opaque_threshold=254.5),
            AlphaSettings(mode="Reconstruct", island_check=True, island_radius=0),
            AlphaSettings(mode="Reconstruct", island_check=True, island_share=0),
            AlphaSettings(mode="Invalid"),
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()
        AlphaSettings(mode="Native", key_color="#808080").validate()

    def test_inactive_reconstruction_settings_do_not_block_generation(self):
        for mode in ("Off", "Native"):
            with self.subTest(mode=mode):
                AlphaSettings(mode=mode, key_color="#808080", opaque_threshold=200, transparent_threshold=100).validate()

    def test_result_more_saturated_than_key_calibrates_without_nan(self):
        source, _ = scene()
        source[:, :16] = (100, 100, 100, 255)
        reference = Image.fromarray(source, "RGBA")
        for color in ((0, 128, 0), (128, 0, 128)):
            with self.subTest(key=color), warnings.catch_warnings(action="error"):
                rendered = Image.new("RGB", reference.size, tuple(255 if channel else 0 for channel in color))
                rendered.paste((100, 100, 100), (0, 0, 16, 16))
                result, calibration = reconstruct(rendered, reference, color)
                pixels = np.asarray(result)
                self.assertTrue((pixels[:, :16] == (100, 100, 100, 255)).all())
                self.assertTrue((pixels[:, 16:, 3] == 0).all())
                self.assertTrue(np.isfinite((calibration.opaque_threshold, calibration.transparent_threshold, calibration.color_shift)).all())
                self.assertLessEqual(calibration.transparent_threshold, 255)

    def test_manual_opaque_limit_keeps_auto_transparent_in_slider_range(self):
        source, rendered = scene()
        rendered[:, 16:] = GREEN
        result, calibration = run(source, rendered, opaque_threshold=254)
        self.assertGreater(calibration.transparent_threshold, calibration.opaque_threshold)
        self.assertLessEqual(calibration.transparent_threshold, 255)
        self.assertTrue((result[:, 16:, 3] == 0).all())
        with self.assertRaisesRegex(ValueError, "opaque threshold"):
            run(source, rendered, opaque_threshold=255)

    def test_inpaint_mask_preserves_original_and_blends_alpha(self):
        original = Image.new("RGBA", (3, 1), (255, 0, 0, 128))
        generated = Image.new("RGBA", (3, 1), (0, 0, 255, 255))
        mask = Image.new("L", (3, 1))
        mask.putdata([0, 128, 255])
        result = composite(generated, original, mask)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0, 128))
        self.assertEqual(result.getpixel((2, 0)), (0, 0, 255, 255))
        self.assertEqual(result.getpixel((1, 0)), (85, 0, 170, 192))

    def test_automatic_key_color_avoids_visible_colors(self):
        source = Image.new("RGBA", (2, 1))
        source.putdata([(30, 220, 40, 255), (255, 0, 255, 0)])
        self.assertEqual(background_color(source, "#00ff00", True), (255, 0, 255))
        self.assertEqual(background_color(source, "#0000ff", False), (0, 0, 255))
        with self.assertRaisesRegex(ValueError, "saturated key color"):
            background_color(source, "#808080", False)

    def test_mismatched_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching image sizes"):
            reconstruct(Image.new("RGB", (1, 1)), Image.new("RGBA", (2, 1)), (0, 0, 0))

    def test_reconstruction_cache_preserves_raw_output_for_repeated_settings(self):
        source, rendered = scene()
        rendered[:, 16] = (65, 195, 50)
        raw = Image.fromarray(rendered, "RGB")
        reference = Image.fromarray(source, "RGBA")
        cache = ReconstructionCache()
        token = cache.store({0: ReconstructionCacheItem(raw, reference, GREEN)})

        raw.paste((0, 0, 0), (0, 0, 1, 1))
        cached = cache.get(token, 0)
        first, _ = reconstruct(cached.image, cached.reference, cached.background)
        second, _ = reconstruct(
            cached.image,
            cached.reference,
            cached.background,
            AlphaSettings(mode="Reconstruct", opaque_threshold=100, transparent_threshold=150, color_shift=200),
        )

        self.assertNotEqual(first.tobytes(), second.tobytes())
        self.assertEqual(cached.image.getpixel((0, 0)), tuple(rendered[0, 0]))

    def test_reconstruction_cache_evicts_old_generations(self):
        cache = ReconstructionCache(max_entries=1)
        image = Image.new("RGB", (1, 1), (10, 20, 30))
        reference = Image.new("RGBA", (1, 1), (10, 20, 30, 255))
        item = ReconstructionCacheItem(image, reference, GREEN)
        old_token = cache.store({0: item})
        new_token = cache.store({0: item})

        with self.assertRaisesRegex(KeyError, "expired"):
            cache.get(old_token, 0)
        self.assertEqual(cache.get(new_token, 0).image.getpixel((0, 0)), (10, 20, 30))

    def test_inpainting_fill_keeps_unmasked_alpha(self):
        source = Image.new("RGBA", (2, 2), (200, 30, 40, 80))
        result = fill(source, Image.new("L", source.size, 0), preserve_alpha=True)
        self.assertEqual(result.tobytes(), source.tobytes())


if __name__ == "__main__":
    unittest.main()
