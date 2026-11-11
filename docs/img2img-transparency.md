# img2img transparency

The img2img settings include three transparency modes:

- **Off** (default) composites transparent input over the configured background,
  as before.
- **Native** passes RGBA through the model. The selected VAE must encode and
  decode four image channels, and both VAE methods must be set to Full. This
  checks technical compatibility, not whether the model was trained for alpha.
  Krea2 reference conditioning must be disabled because it discards alpha.
- **Reconstruct** sends an RGB composite to the model and estimates alpha from
  the output, background color, and original image. The optional automatic
  background setting chooses a contrasting color for each input image.

Reconstruction is approximate: a single RGB image does not uniquely determine
its original alpha. Newly generated colors may resemble the background. Inpainting
uses the overlay mask to preserve the original outside the edited area; disabling
the inpainting overlay also disables that preservation.
The reconstruction is a generalized color-compositing heuristic inspired by the
original green/cyan restoration script, not a direct port of its edge classifiers.

Transparency modes require PNG or WebP for saved samples and do not support
latent upscale, video, PiD, face restoration, or img2img color correction.
Alpha-aware resizing uses Pillow rather than the configured RGB upscaler.
Saved samples retain their alpha values without applying the global alpha
threshold; grids and other images retain their existing save behavior.
Extensions may not support RGBA and can cause an explicit error or alter results.
