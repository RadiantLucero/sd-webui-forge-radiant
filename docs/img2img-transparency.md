# img2img transparency

The **Transparency** accordion in the img2img scripts area controls alpha
handling. When the accordion is disabled, transparent input is composited over
the `img2img_background_color` setting, as before.

- **Native** passes RGBA through the model. The selected VAE must encode and
  decode four image channels, and both VAE methods must be set to Full. This
  checks technical compatibility, not whether the model was trained for alpha.
  Krea2 reference conditioning must be disabled because it discards alpha.
- **Reconstruct** composites the input over a key color, sends the RGB image to
  the model, and keys the key color out of the generated result.

## Reconstruction

Every pixel decision uses only the generated image. The original image supplies
global color statistics for calibration and, with **Auto key color**, chooses the
key; it is never copied into the reconstructed pixels. Inpainting still uses its
overlay mask to keep the original outside the edited area.

The key color must be saturated: at least one channel must be 128 or higher,
at least one must be lower, and the gap between them must be at least 128.
Gray, white, and black cannot be separated from neutral content. **Auto key
color** chooses the green, magenta, blue, red, cyan, or yellow key that the
visible input leans toward least.

Thresholds use **key strength**: the lowest key channel minus the highest
non-key channel, scaled so that the pure key color is 255. For green this is
`G - max(R, B)`. Neutral colors are 0 and colors leaning away from the key are
negative.

- **Opaque up to**: pixels at or below this strength stay fully opaque and
  unchanged.
- **Transparent from**: pixels at or above this strength become fully
  transparent.
- Pixels between both thresholds receive linear partial alpha.
- **Color shift**: key strength removed from partially transparent pixels in
  proportion to their transparency. Despill never pushes a color past neutral,
  so it cannot create the complementary tint, such as magenta for green.

*Opaque up to* ranges from 0 to 254 and must stay below *Transparent from*
(1 to 255). Invalid manual pairs produce an explicit error. Reconstruction
settings are ignored when transparency is disabled or **Native** is selected.
Results more saturated than the configured key use the top histogram bin for
calibration, keeping Auto values within the slider range.

Each value has an **Auto** checkbox. Auto values are calculated per image:

- The generated image's key-strength histogram locates the background peak.
  *Transparent from* is the peak minus three robust standard deviations.
- The visible original pixels give the expected subject strength. The shift of
  the generated subject relative to the original is reported as `tint`.
  *Opaque up to* is the original's 99th percentile plus the tint and a noise
  margin, and normally stays at least 32 below *Transparent from* (a manually
  selected transparent threshold below 32 leaves a narrower band).
- *Color shift* is the distance between the background peak and the
  generated subject.

Moving a slider switches its value to manual. After generation, the accordion
lists the values used for every image, and Auto sliders show the values from
the first image. **Use last calculated values** copies those values into the
sliders as manual values. The values and statistics are also written to the
infotext and console log.

Reconstruction remains approximate: one RGB image does not uniquely determine
alpha. New content close to the key color can become partially transparent, and
there is no neighborhood analysis.

## API

The script is named `transparency`. Its `alwayson_scripts` arguments are:
`enabled, mode, key_color, auto_key_color, opaque_auto, opaque,
transparent_auto, transparent, color_shift_auto, color_shift`.

## Limits

Transparency modes require PNG or WebP for saved samples and do not support
latent upscale, video, PiD, face restoration, or img2img color correction.
Alpha-aware resizing uses Pillow rather than the configured RGB upscaler.
Saved samples retain their alpha values without applying the global alpha
threshold; grids and other images retain their existing save behavior.
Extensions may not support RGBA and can cause an explicit error or alter results.
