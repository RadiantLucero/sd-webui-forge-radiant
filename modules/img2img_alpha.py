"""Alpha handling for img2img inputs and outputs."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageColor


def validate_native_vae(vae):
    model = vae.first_stage_model
    encoder = getattr(getattr(model, "encoder", None), "conv1", None)
    input_channels = getattr(encoder, "in_channels", None)
    output_channels = getattr(vae, "output_channels", None)
    if input_channels != 4 or output_channels != 4:
        vae_type = f"{type(vae).__module__}.{type(vae).__qualname__}"
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        encoder_type = f"{type(encoder).__module__}.{type(encoder).__qualname__}" if encoder is not None else "unavailable"
        raise RuntimeError(
            "Native img2img alpha requires a VAE with four-channel image encoding and decoding. "
            f"VAE wrapper: {vae_type}; VAE model: {model_type}; encoder.conv1: {encoder_type}; "
            f"encoder input channels: {input_channels if input_channels is not None else 'unavailable'}; "
            f"VAE output channels: {output_channels if output_channels is not None else 'unavailable'}."
        )


MODES = ("Off", "Native", "Reconstruct")
DEFAULT_KEY_COLOR = "#00ff00"
KEY_CANDIDATES = ((0, 255, 0), (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0))
MIN_KEY_CONTRAST = 128
BACKGROUND_SEARCH_START = 96
BACKGROUND_WINDOW = 32
MIN_BACKGROUND_FRACTION = 0.001
MIN_PARTIAL_BAND = 32
VISIBLE_ALPHA = 242
TRANSPARENT_ALPHA = 12
_SMOOTHING = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
_SMOOTHING /= _SMOOTHING.sum()


@dataclass(frozen=True)
class AlphaSettings:
    """User settings; ``None`` thresholds are calibrated for every output image."""

    mode: str = "Off"
    key_color: str = DEFAULT_KEY_COLOR
    auto_key_color: bool = False
    opaque_threshold: float | None = None
    transparent_threshold: float | None = None
    color_shift: float | None = None

    def validate(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown img2img transparency mode: {self.mode}")
        if self.mode != "Reconstruct":
            return
        for name in ("opaque_threshold", "transparent_threshold", "color_shift"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 255:
                raise ValueError(f"img2img transparency {name.replace('_', ' ')} must be between 0 and 255.")
        if self.opaque_threshold is not None and self.opaque_threshold > 254:
            raise ValueError("The img2img transparency opaque threshold must be at most 254.")
        if self.opaque_threshold is not None and self.transparent_threshold is not None and self.opaque_threshold >= self.transparent_threshold:
            raise ValueError("The img2img transparency opaque threshold must be lower than the transparent threshold.")
        ChromaKey(ImageColor.getrgb(self.key_color))


@dataclass(frozen=True)
class Calibration:
    """Thresholds in key-strength units: 0 is neutral and 255 is the pure key color."""

    opaque_threshold: float
    transparent_threshold: float
    color_shift: float
    auto_opaque: bool
    auto_transparent: bool
    auto_color_shift: bool
    background_level: float | None
    background_spread: float | None
    foreground_level: float
    original_level: float | None
    tint: float
    original_transparent_fraction: float
    background_fraction: float

    def infotext(self):
        def value(number, automatic):
            return f"{number:.0f} auto" if automatic else f"{number:.0f}"

        return {
            "img2img alpha opaque": value(self.opaque_threshold, self.auto_opaque),
            "img2img alpha transparent": value(self.transparent_threshold, self.auto_transparent),
            "img2img alpha color shift": value(self.color_shift, self.auto_color_shift),
            "img2img alpha statistics": self.statistics(),
        }

    def statistics(self):
        background = "none" if self.background_level is None else f"{self.background_level:.0f} +/- {self.background_spread:.0f}"
        original = "none" if self.original_level is None else f"{self.original_level:.0f}"
        return (
            f"background {background}; foreground {self.foreground_level:.0f}; original {original}; tint {self.tint:.0f}; "
            f"transparent area {self.original_transparent_fraction:.0%} -> {self.background_fraction:.0%}"
        )


class ChromaKey:
    """Measures how strongly a color leans toward a saturated key color."""

    def __init__(self, color):
        self.color = tuple(int(value) for value in color[:3])
        values = np.asarray(self.color, dtype=np.float32)
        self.high = values >= 128
        if self.high.all() or not self.high.any():
            raise ValueError(f"Alpha reconstruction requires a saturated key color such as #00ff00; {format_color(self.color)} cannot be separated from neutral colors.")
        self.contrast = float(values[self.high].min() - values[~self.high].max())
        if self.contrast < MIN_KEY_CONTRAST:
            raise ValueError(f"Alpha reconstruction requires a saturated key color such as #00ff00; {format_color(self.color)} has a key contrast of {self.contrast:.0f}, at least {MIN_KEY_CONTRAST} is required.")

    def strength(self, rgb):
        rgb = np.asarray(rgb, dtype=np.float32)
        return (rgb[..., self.high].min(axis=-1) - rgb[..., ~self.high].max(axis=-1)) * (255.0 / self.contrast)


def format_color(color):
    return "#%02x%02x%02x" % tuple(color[:3])


def background_color(image, configured_color, automatic):
    if not automatic:
        return ChromaKey(ImageColor.getrgb(configured_color)).color

    sample = image.convert("RGBA")
    sample.thumbnail((96, 96), Image.Resampling.BOX)
    pixels = np.asarray(sample, dtype=np.float32)
    visible = pixels[:, :, :3][pixels[:, :, 3] > 16]
    if not len(visible):
        return ChromaKey(ImageColor.getrgb(configured_color)).color

    # Reconstruction separates by key strength, so pick the key the visible content leans toward least.
    scores = [np.percentile(ChromaKey(candidate).strength(visible), 99) for candidate in KEY_CANDIDATES]
    return KEY_CANDIDATES[int(np.argmin(scores))]


def _spread(values, center):
    return 1.4826 * float(np.median(np.abs(values - center))) if values.size else 0.0


def calibrate(strength, original_strength, original_transparent_fraction, settings):
    """Derive thresholds from the generated key-strength histogram and global original statistics."""
    settings.validate()
    # Use the same upper bound for the histogram and its samples when the result exceeds the key's saturation.
    values = np.minimum(np.asarray(strength, dtype=np.float32).ravel(), 255.0)
    original_strength = np.asarray(original_strength, dtype=np.float32).ravel()

    background_level = background_spread = None
    foreground = values
    if np.mean(values >= BACKGROUND_SEARCH_START) >= MIN_BACKGROUND_FRACTION:
        histogram, _ = np.histogram(np.clip(values, 0, 255), bins=256, range=(0, 256))
        smoothed = np.convolve(histogram, _SMOOTHING, mode="same")
        peak = BACKGROUND_SEARCH_START + int(np.argmax(smoothed[BACKGROUND_SEARCH_START:]))
        cluster = values[np.abs(values - peak) <= BACKGROUND_WINDOW]
        background_level = float(np.median(cluster))
        background_spread = max(2.0, _spread(cluster, background_level))
        foreground = values[values < background_level / 2]

    original_level = float(np.median(original_strength)) if original_strength.size else None
    if foreground.size:
        foreground_level = float(np.median(foreground))
        foreground_spread = _spread(foreground, foreground_level)
    else:
        foreground_level = original_level if original_level is not None else 0.0
        foreground_spread = 0.0

    if original_level is None:
        tint = 0.0
        upper = foreground_level
        noise = foreground_spread
    else:
        tint = max(0.0, foreground_level - original_level)
        upper = float(np.percentile(original_strength, 99)) + tint
        noise = float(np.sqrt(max(foreground_spread**2 - _spread(original_strength, original_level) ** 2, 0.0)))

    auto_transparent = 255.0 if background_level is None else float(np.clip(round(background_level - 3 * background_spread), 64, 255))
    transparent = auto_transparent if settings.transparent_threshold is None else float(settings.transparent_threshold)

    auto_opaque = max(min(max(round(upper + max(12.0, 2 * noise)), 16.0), transparent - MIN_PARTIAL_BAND), 0.0)
    opaque = auto_opaque if settings.opaque_threshold is None else float(settings.opaque_threshold)
    if settings.transparent_threshold is None and opaque >= transparent:
        transparent = opaque + 1
    if opaque >= transparent:
        raise ValueError(f"The img2img transparency opaque threshold ({opaque:.0f}) must be lower than the transparent threshold ({transparent:.0f}).")

    reference_level = background_level if background_level is not None else transparent
    auto_shift = float(np.clip(round(reference_level - max(foreground_level, 0.0)), 0, 255))
    shift = auto_shift if settings.color_shift is None else float(settings.color_shift)

    return Calibration(
        opaque_threshold=opaque,
        transparent_threshold=transparent,
        color_shift=shift,
        auto_opaque=settings.opaque_threshold is None,
        auto_transparent=settings.transparent_threshold is None,
        auto_color_shift=settings.color_shift is None,
        background_level=background_level,
        background_spread=background_spread,
        foreground_level=foreground_level,
        original_level=original_level,
        tint=tint,
        original_transparent_fraction=float(original_transparent_fraction),
        background_fraction=float(np.mean(values >= transparent)) if values.size else 0.0,
    )


def reconstruct(image, reference, background, settings=None):
    """Estimate straight alpha from the generated RGB image.

    The reference contributes only global color statistics; every pixel decision uses the generated image.
    """
    if image.size != reference.size:
        raise ValueError(f"Alpha reconstruction requires matching image sizes: {image.size} != {reference.size}")

    settings = settings or AlphaSettings(mode="Reconstruct")
    key = ChromaKey(background)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    original = np.asarray(reference.convert("RGBA"), dtype=np.float32)
    strength = key.strength(rgb)
    calibration = calibrate(
        strength,
        key.strength(original[:, :, :3][original[:, :, 3] >= VISIBLE_ALPHA]),
        np.mean(original[:, :, 3] <= TRANSPARENT_ALPHA),
        settings,
    )

    opaque, transparent = calibration.opaque_threshold, calibration.transparent_threshold
    alpha = np.clip((transparent - strength) / (transparent - opaque), 0.0, 1.0)
    # Remove key spill in proportion to transparency, but never push a color past neutral.
    reduced = np.maximum(strength - (1.0 - alpha) * calibration.color_shift, np.minimum(strength, 0.0))
    rgb[:, :, key.high] -= ((strength - reduced) * (key.contrast / 255.0))[:, :, None]

    result = np.dstack((np.clip(rgb, 0, 255), alpha * 255.0))
    return Image.fromarray(np.round(result).astype(np.uint8), "RGBA"), calibration


def composite(image, original, mask):
    """Blend two straight-alpha images according to an inpainting mask."""
    if image.size != original.size or mask.size != original.size:
        raise ValueError("Alpha inpainting requires matching image and mask sizes")
    generated = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    source = np.asarray(original.convert("RGBA"), dtype=np.float32) / 255.0
    weight = np.asarray(mask.convert("L"), dtype=np.float32)[:, :, None] / 255.0
    alpha = generated[:, :, 3:4] * weight + source[:, :, 3:4] * (1 - weight)
    premultiplied = generated[:, :, :3] * generated[:, :, 3:4] * weight + source[:, :, :3] * source[:, :, 3:4] * (1 - weight)
    rgb = np.divide(premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 0)
    result = np.concatenate((rgb, alpha), axis=2)
    return Image.fromarray(np.round(np.clip(result, 0, 1) * 255).astype(np.uint8), "RGBA")
