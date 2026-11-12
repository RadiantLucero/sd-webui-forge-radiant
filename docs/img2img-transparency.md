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

Thresholds use **key dominance**: the lowest key channel minus the highest
non-key channel, scaled so that the pure key color is 255. For green this is
`G - max(R, B)`. Neutral colors are 0 and colors leaning away from the key are
negative. Content should not be strongly dominated by the key color, so
dominance separates the subject from the background.

- **Partial transparency above key dominance**: pixels at or below this value
  stay fully opaque and unchanged. A mix such as half red and half green is not
  dominant enough and stays opaque.
- **Transparent from key dominance**: pixels at or above this value become fully
  transparent and lose their key tint.
- Pixels between both thresholds are unmixed. The model renders a translucent
  pixel as a mix of its color and the background, so alpha falls linearly with
  dominance and reaches zero at the **Color shift** value. The background color
  measured in the result is then removed from the pixel in proportion to its
  transparency, which restores the brightness of the remaining color.
- **Color shift**: the key dominance of a fully transparent pixel. At the
  measured background dominance, unmixed colors are neutral with respect to the
  key. Lower values make edges more transparent and tint them away from the key,
  such as magenta for green. Higher values make edges more opaque and keep some
  key tint.

*Partial transparency above* ranges from 0 to 254 and must stay below
*Transparent from* (1 to 255). Invalid manual pairs produce an explicit error.
Reconstruction settings are ignored when transparency is disabled or **Native**
is selected. Results more saturated than the configured key use the top
histogram bin for the thresholds, keeping them within the slider range.

Each value has an **Auto** checkbox. Auto values are calculated per image:

- The generated image's key-dominance histogram locates the background peak.
  The search starts in the most key-like share of pixels that the original's
  transparent area accounts for, so a large uniform translucent area is not
  mistaken for the background. The background cluster's median color is the
  measured background, for example a darker green than the configured key.
  *Transparent from* is the peak minus three robust standard deviations.
- The original's opaque area determines how many of the least key-like
  generated pixels are compared with it. Translucent pixels are left out, so
  they cannot raise the threshold. The shift of the generated subject relative
  to the original is reported as `tint`. *Partial transparency above* is the
  original's 99th percentile plus the tint and a noise margin, and normally
  stays at least 32 below *Transparent from* (a manually selected transparent
  threshold below 32 leaves a narrower band).
- *Color shift* is the dominance of the measured background.

**Island check** (off by default) adds a second pass for opaque pixels that are
mostly surrounded by partially or fully transparent pixels, such as dark spots
inside a shadow or smoke whose dominance stays just below the partial
threshold. For every opaque pixel, the pass counts the other pixels within
**Island radius** in every direction (a square, clipped at the image border).
If at least **Island surrounded share** of them are partially or fully
transparent, the pixel uses **Island: partial transparency above key
dominance** instead of the normal partial threshold and is unmixed like other
partially transparent pixels. The decision uses the first pass only, so islands
do not spread into each other. Auto uses the original's 99th percentile plus
the tint without the noise margin, limited to the normal partial threshold, so
colors that the subject itself contains stay opaque. Alpha still follows the
pixel's own key dominance: the pass removes the key tint from such spots and
makes them slightly transparent, but a nearly black pixel cannot become
strongly transparent. The infotext and console statistics report the number of
island pixels.

**Alpha from surrounding smoke** (on by default, used with the island check)
assumes the usual case of an opaque subject surrounded by translucent smoke.
Islands whose neighbors are mainly partially transparent (again at least
**Island surrounded share**) take the average alpha of those neighbors instead
of their own, so dark spots where the model dimmed the key behind the smoke
become as transparent as the smoke around them. A pixel never becomes more
transparent than its brightness allows, so sparks and highlights stay strong.
Only the key share that the pixel still shows is removed from its color, so it
stays neutral instead of turning magenta. Pixels leaning clearly away from the
key, such as skin, red, or brown, are never taken from smoke, and islands on
empty background keep the island rule above. Thin dark neutral details inside
smoke, such as black hair, may become translucent. Only opaque pixels are
taken; dark smoke that still shows some key keeps its own, possibly too high,
alpha. Larger dark spots need a larger radius and a lower share, for example 8
and 60%, which may also erode corners of dark neutral subjects by a pixel or
two. The infotext and console statistics report how many island pixels were
taken from smoke.

Moving a slider switches its value to manual. After generation, the accordion
lists the values used for every image, and Auto sliders show the values from
the first image. **Use last calculated values** copies those values into the
sliders as manual values. The values and statistics are also written to the
infotext and console log.

Select a generated image in the img2img output gallery, adjust the settings, and
use **Reconstruct with current settings** to rebuild its alpha without another
model generation. **Save raw image to output** saves the generated RGB image
before reconstruction to the normal img2img output directory. Both buttons act
on the selected generated image, not on a grid or mask. If no image is selected
and there is exactly one generated result, both buttons use it automatically,
even if the gallery also contains a grid or masks. With multiple generated
results, click the desired image in the img2img gallery first. The raw images and
their calibration references are kept in memory for up to two generation batches;
restarting the application clears that cache.

Reconstruction remains approximate: one RGB image does not uniquely determine
alpha. Unmixing assumes that translucent content is not tinted toward the key,
which is exact for white, gray, black, smoke, glass, and veils. Strongly colored
soft edges, such as red or skin, become slightly too opaque and less saturated,
and edges whose dominance stays below the partial threshold remain opaque. New
content close to the key color can become partially transparent. Apart from
the optional island check, there is no neighborhood analysis.

## API

The script is named `transparency`. Its `alwayson_scripts` arguments are:
`enabled, mode, key_color, auto_key_color, opaque_auto, opaque,
transparent_auto, transparent, color_shift_auto, color_shift`, optionally
followed by `island_check, island_auto, island, island_radius,
island_share_percent, island_smoke`.

## Limits

Transparency modes require PNG or WebP for saved samples and do not support
latent upscale, video, PiD, face restoration, or img2img color correction.
Alpha-aware resizing uses Pillow rather than the configured RGB upscaler.
Saved samples retain their alpha values without applying the global alpha
threshold; grids and other images retain their existing save behavior.
Extensions may not support RGBA and can cause an explicit error or alter results.
