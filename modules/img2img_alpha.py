"""Alpha handling for img2img inputs and outputs."""

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


def background_color(image, configured_color, automatic):
    if not automatic:
        return ImageColor.getrgb(configured_color)

    sample = image.convert("RGBA")
    sample.thumbnail((96, 96), Image.Resampling.BOX)
    pixels = np.asarray(sample, dtype=np.float32)
    visible = pixels[:, :, :3][pixels[:, :, 3] > 16]
    if not len(visible):
        return ImageColor.getrgb(configured_color)

    candidates = np.array([
        (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0),
        (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ], dtype=np.float32)
    scores = []
    for candidate in candidates:
        distances = np.sum((visible - candidate) ** 2, axis=1)
        scores.append(np.percentile(distances, 5))
    return tuple(int(value) for value in candidates[int(np.argmax(scores))])


def reconstruct(image, reference, background):
    """Estimate a straight-alpha image from an RGB composite and its source."""
    if image.size != reference.size:
        raise ValueError(f"Alpha reconstruction requires matching image sizes: {image.size} != {reference.size}")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    original = np.asarray(reference.convert("RGBA"), dtype=np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32) / 255.0
    expected = original[:, :, :3] * original[:, :, 3:4] + bg * (1.0 - original[:, :, 3:4])
    unchanged = np.max(np.abs(rgb - expected), axis=2) <= 1.0 / 255.0

    lower = np.divide(bg - rgb, bg, out=np.zeros_like(rgb), where=bg > 0)
    upper = np.divide(rgb - bg, 1.0 - bg, out=np.zeros_like(rgb), where=bg < 1)
    alpha = np.clip(np.maximum(lower, upper).max(axis=2), 0.0, 1.0)

    # Estimate alpha from channels where the reference differs enough from the background.
    difference = original[:, :, :3] - bg
    usable = np.abs(difference) >= 32.0 / 255.0
    estimates = np.divide(rgb - bg, difference, out=np.zeros_like(rgb), where=usable)
    usable &= (estimates >= -0.05) & (estimates <= 1.05)
    estimates = np.sort(np.where(usable, np.clip(estimates, 0, 1), 2.0), axis=2)
    count = usable.sum(axis=2)
    reference_alpha = np.take_along_axis(estimates, (count // 2)[:, :, None], axis=2)[:, :, 0]
    reference_composite = bg + difference * reference_alpha[:, :, None]
    reference_match = np.max(np.abs(rgb - reference_composite), axis=2) <= 32.0 / 255.0
    original_alpha = original[:, :, 3]
    use_reference = (original_alpha > 0) & (original_alpha < 1) & (count >= 2) & reference_match
    alpha = np.where(use_reference, np.maximum(alpha, reference_alpha), alpha)

    opaque = (original[:, :, 3] == 1) & (np.max(np.abs(rgb - bg), axis=2) > 25.0 / 255.0)
    alpha[opaque] = 1.0
    foreground = np.divide(
        rgb - (1.0 - alpha[:, :, None]) * bg,
        alpha[:, :, None],
        out=np.zeros_like(rgb),
        where=alpha[:, :, None] > 1.0 / 255.0,
    )
    foreground[unchanged] = original[:, :, :3][unchanged]
    alpha[unchanged] = original[:, :, 3][unchanged]
    result = np.concatenate((np.clip(foreground, 0, 1), alpha[:, :, None]), axis=2)
    return Image.fromarray(np.round(result * 255).astype(np.uint8), "RGBA")


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
