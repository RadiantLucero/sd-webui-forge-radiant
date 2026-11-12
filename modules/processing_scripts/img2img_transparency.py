import json
import re
from types import SimpleNamespace

import gradio as gr
from PIL import Image

from modules import images, img2img_alpha, scripts, shared
from modules.infotext_utils import PasteField, parse_generation_parameters, quote
from modules.ui_common import plaintext_to_html
from modules.ui_components import FormColorPicker, InputAccordion

RECONSTRUCTION_CACHE_KEY = "img2img alpha reconstruction cache"
RECONSTRUCTION_CACHE_INDICES_KEY = "img2img alpha reconstruction cache indices"
RECONSTRUCTION_CACHE = img2img_alpha.ReconstructionCache()

THRESHOLD_KEYS = (
    ("opaque", "img2img alpha opaque", "Partial above"),
    ("transparent", "img2img alpha transparent", "Transparent from"),
    ("shift", "img2img alpha color shift", "Color shift"),
    ("island", "img2img alpha island", "Island partial above"),
)
# Absent when the island check was off.
OPTIONAL_KEYS = {"island"}
ISLAND_RADIUS_KEY = "img2img alpha island radius"
ISLAND_SHARE_KEY = "img2img alpha island share"
ISLAND_SMOKE_KEY = "img2img alpha island smoke"
OPTIONAL_INFOTEXT_KEYS = ("img2img alpha island", ISLAND_RADIUS_KEY, ISLAND_SHARE_KEY, ISLAND_SMOKE_KEY)


def parse_reconstruction_cache(generation_info):
    try:
        info = json.loads(generation_info or "")
    except (TypeError, ValueError):
        return None

    if not isinstance(info, dict):
        return None
    params = info.get("extra_generation_params")
    if not isinstance(params, dict):
        return None
    token = params.get(RECONSTRUCTION_CACHE_KEY)
    indices = params.get(RECONSTRUCTION_CACHE_INDICES_KEY)
    if not isinstance(token, str) or not isinstance(indices, dict):
        return None
    return info, token, indices


def replace_infotext_values(infotext, values):
    *lines, parameters = infotext.rsplit("\n", 1)

    for key, value in values.items():
        pattern = re.compile(
            rf'(?P<prefix>(?:^|,\s*){re.escape(key)}:\s*)(?:"(?:\\.|[^"\\])*"|[^,]*)'
        )
        if value is None:
            parameters = pattern.sub("", parameters)
            continue
        def replacement(match):
            return match.group("prefix") + quote(str(value))

        if pattern.search(parameters):
            parameters = pattern.sub(replacement, parameters, count=1)
        else:
            parameters += f", {key}: {quote(str(value))}"

    return "\n".join([*lines, parameters])


def parse_threshold(text):
    """Parse infotext values such as ``"42 auto"`` into ``(42.0, True)``."""
    parts = str(text).split()
    try:
        return float(parts[0]), "auto" in parts[1:]
    except (IndexError, ValueError):
        return None


def last_calculated_values(generation_info):
    try:
        info = json.loads(generation_info or "")
        infotexts = info["infotexts"][info.get("index_of_first_image", 0) :]
    except (TypeError, ValueError, KeyError):
        return []

    results = []
    for text in infotexts:
        try:
            params = parse_generation_parameters(text)
        except Exception:
            continue
        values = {name: parse_threshold(params[key]) if key in params else None for name, key, _ in THRESHOLD_KEYS}
        if all(values[name] for name, _, _ in THRESHOLD_KEYS if name not in OPTIONAL_KEYS):
            results.append((values, params.get("img2img alpha statistics", "")))
    return results


def describe(results):
    if not results:
        return "No reconstructed transparency in the last generation."

    def cell(value):
        if value is None:
            return "off"
        number, automatic = value
        return f"{number:.0f} (auto)" if automatic else f"{number:.0f}"

    rows = ["| Image | " + " | ".join(label for _, _, label in THRESHOLD_KEYS) + " | Statistics |", "|---" * (len(THRESHOLD_KEYS) + 2) + "|"]
    for index, (values, statistics) in enumerate(results, start=1):
        rows.append(f"| {index} | " + " | ".join(cell(values[name]) for name, _, _ in THRESHOLD_KEYS) + f" | {statistics} |")
    return "\n".join(rows)


class ScriptImg2imgTransparency(scripts.Script):
    sorting_priority = -1

    def __init__(self):
        self.on_after_component(self.connect_generation_info, elem_id="generation_info_img2img")
        self.on_after_component(self.connect_gallery, elem_id="img2img_gallery")
        self.on_after_component(self.connect_infotext, elem_id="html_info_img2img")
        self.gallery = None
        self.infotext = None

    def title(self):
        return "Transparency"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else None

    def ui(self, is_img2img):
        with InputAccordion(False, label="Transparency", elem_id=self.elem_id("enable")) as enable:
            mode = gr.Radio(
                label="Mode",
                choices=("Reconstruct", "Native"),
                value="Reconstruct",
                elem_id=self.elem_id("mode"),
                info="Reconstruct keys out a background color from the RGB result; Native requires a VAE that encodes and decodes alpha",
            )

            with gr.Column(visible=True) as reconstruct_controls:
                gr.Markdown("Thresholds use key dominance: how far the key color's channels exceed the others, from 0 for a neutral color to 255 for the pure key color. For green this is G - max(R, B). Partially transparent pixels are unmixed from the background color measured in the result. Auto values are calculated for every image from the result and the color statistics of the input.")
                gr.Markdown("The raw generated RGB image is kept in a short-lived cache so you can adjust these settings and reconstruct it again without another generation.")
                with gr.Row():
                    key_color = FormColorPicker(label="Key color", value=img2img_alpha.DEFAULT_KEY_COLOR, elem_id=self.elem_id("key_color"))
                    auto_key = gr.Checkbox(label="Auto key color", value=False, elem_id=self.elem_id("auto_key"), info="choose the saturated color least present in the visible input")

                sliders = {}
                autos = {}

                def threshold_row(name, minimum, maximum, value, label, info):
                    with gr.Row(equal_height=True):
                        autos[name] = gr.Checkbox(label="Auto", value=True, elem_id=self.elem_id(f"{name}_auto"), scale=0, min_width=80)
                        sliders[name] = gr.Slider(label=label, minimum=minimum, maximum=maximum, step=1, value=value, elem_id=self.elem_id(name), info=info, scale=1)
                    sliders[name].input(fn=lambda: False, outputs=[autos[name]], show_progress="hidden")

                for spec in (
                    ("opaque", 0, 254, 40, "Partial transparency above key dominance", "pixels at or below this value stay fully opaque and unchanged"),
                    ("transparent", 1, 255, 200, "Transparent from key dominance", "pixels at or above this value become fully transparent"),
                    ("shift", 1, 255, 180, "Color shift", "key dominance of a fully transparent pixel when unmixing; lower values give more transparent edges tinted away from the key, higher values more opaque edges tinted toward it"),
                ):
                    threshold_row(*spec)

                island_check = gr.Checkbox(label="Island check", value=False, elem_id=self.elem_id("island_check"), info="second pass for opaque pixels that are mostly surrounded by partially or fully transparent pixels")
                with gr.Column(visible=False) as island_controls:
                    threshold_row("island", 0, 254, 0, "Island: partial transparency above key dominance", "lower partial threshold for islands not taken from smoke; auto uses the subject's own key dominance without the noise margin")
                    with gr.Row():
                        island_radius = gr.Slider(label="Island radius", minimum=1, maximum=img2img_alpha.MAX_ISLAND_RADIUS, step=1, value=4, elem_id=self.elem_id("island_radius"), info="pixels checked in every direction around an opaque pixel")
                        island_share = gr.Slider(label="Island surrounded share (%)", minimum=1, maximum=100, step=1, value=70, elem_id=self.elem_id("island_share"), info="minimum share of partially or fully transparent pixels around it")
                    island_smoke = gr.Checkbox(label="Alpha from surrounding smoke", value=True, elem_id=self.elem_id("island_smoke"), info="assumes an opaque subject surrounded by translucent smoke: islands mostly surrounded by partial transparency take its average alpha; brightness limits transparency, colors leaning away from the key stay opaque, but thin dark neutral details inside smoke may turn translucent")

                self.calculated_values = gr.Markdown(describe([]), elem_id=self.elem_id("calculated_values"))
                self.last_values = gr.State(None)
                use_values = gr.Button("Use last calculated values", elem_id=self.elem_id("use_values"))
                with gr.Row():
                    self.reconstruct_button = gr.Button("Reconstruct with current settings", interactive=False, elem_id=self.elem_id("reconstruct_again"))
                    self.save_original_button = gr.Button("Save raw image to output", interactive=False, elem_id=self.elem_id("save_original"))
                self.cache_status = gr.Textbox(show_label=False, interactive=False, visible=False, elem_id=self.elem_id("cache_status"))

        mode.change(fn=lambda value: gr.update(visible=value == "Reconstruct"), inputs=[mode], outputs=[reconstruct_controls], show_progress="hidden")
        island_check.change(fn=lambda value: gr.update(visible=bool(value)), inputs=[island_check], outputs=[island_controls], show_progress="hidden")

        names = [name for name, _, _ in THRESHOLD_KEYS]
        self.auto_inputs = [autos[name] for name in names]
        self.slider_outputs = [sliders[name] for name in names]
        self.reconstruction_inputs = [
            key_color, auto_key,
            autos["opaque"], sliders["opaque"],
            autos["transparent"], sliders["transparent"],
            autos["shift"], sliders["shift"],
            island_check, autos["island"], sliders["island"], island_radius, island_share, island_smoke,
        ]

        def use_last_values(values):
            values = values or {}
            updates = [gr.skip() if values.get(name) is None else gr.update(value=min(max(values[name], sliders[name].minimum), sliders[name].maximum)) for name in names]
            return updates + [gr.skip() if values.get(name) is None else False for name in names]

        use_values.click(fn=use_last_values, inputs=[self.last_values], outputs=self.slider_outputs + self.auto_inputs, show_progress="hidden")

        def paste_threshold(key, slider):
            def paste(params):
                value = parse_threshold(params[key]) if key in params else None
                if value is None:
                    return None
                return value[1] if slider is None else min(max(value[0], slider.minimum), slider.maximum)

            return paste

        def paste_key_color(params):
            colors = params.get("img2img alpha background")
            return colors.split(",")[0].strip() if colors else None

        def paste_number(key, slider, suffix=""):
            def paste(params):
                try:
                    value = float(str(params[key]).strip().removesuffix(suffix))
                except (KeyError, ValueError):
                    return None
                return min(max(value, slider.minimum), slider.maximum)

            return paste

        self.infotext_fields = [
            PasteField(enable, lambda params: "img2img alpha" in params),
            PasteField(mode, lambda params: params.get("img2img alpha")),
            PasteField(key_color, paste_key_color),
            PasteField(auto_key, lambda params: False if "img2img alpha background" in params else None),
            PasteField(island_check, lambda params: "img2img alpha island" in params if "img2img alpha" in params else None),
            PasteField(island_radius, paste_number(ISLAND_RADIUS_KEY, island_radius)),
            PasteField(island_share, paste_number(ISLAND_SHARE_KEY, island_share, "%")),
            PasteField(island_smoke, lambda params: params.get(ISLAND_SMOKE_KEY) == "on" if "img2img alpha island" in params else None),
        ]
        for name, key, _ in THRESHOLD_KEYS:
            self.infotext_fields.append(PasteField(sliders[name], paste_threshold(key, sliders[name])))
            self.infotext_fields.append(PasteField(autos[name], paste_threshold(key, None)))

        return [
            enable, mode, key_color, auto_key,
            autos["opaque"], sliders["opaque"], autos["transparent"], sliders["transparent"], autos["shift"], sliders["shift"],
            island_check, autos["island"], sliders["island"], island_radius, island_share, island_smoke,
        ]

    def connect_generation_info(self, component):
        generation_info = component.component

        def show_calculated_values(info, *automatic):
            results = last_calculated_values(info)
            cache_data = parse_reconstruction_cache(info)
            has_cache = cache_data is not None and bool(cache_data[2])
            if not results:
                return (
                    [describe(results), None]
                    + [gr.skip()] * len(THRESHOLD_KEYS)
                    + [
                        gr.update(interactive=False),
                        gr.update(interactive=False),
                        gr.update(value="", visible=False),
                    ]
                )

            first = {name: value[0] for name, value in results[0][0].items() if value is not None}
            sliders = [gr.update(value=min(max(first[name], slider.minimum), slider.maximum)) if auto and name in first else gr.skip() for (name, _, _), slider, auto in zip(THRESHOLD_KEYS, self.slider_outputs, automatic)]
            return [describe(results), first] + sliders + [
                gr.update(interactive=has_cache),
                gr.update(interactive=has_cache),
                gr.update(value="", visible=False),
            ]

        generation_info.change(
            fn=show_calculated_values,
            inputs=[generation_info] + self.auto_inputs,
            outputs=[self.calculated_values, self.last_values] + self.slider_outputs + [
                self.reconstruct_button,
                self.save_original_button,
                self.cache_status,
            ],
            show_progress="hidden",
        )
        reconstruct_inputs = [generation_info, self.gallery] + self.reconstruction_inputs
        self.reconstruct_button.click(
            fn=self.reconstruct_selected,
            _js=f"(...args) => [...args.slice(0, {len(reconstruct_inputs)}), selected_gallery_index_id('img2img_gallery')]",
            inputs=reconstruct_inputs + [self.infotext],
            outputs=[self.gallery, generation_info, self.infotext, self.cache_status],
            show_progress="hidden",
        )
        self.save_original_button.click(
            fn=self.save_original,
            _js="(info, gallery, _) => [info, gallery, selected_gallery_index_id('img2img_gallery')]",
            inputs=[generation_info, self.gallery, self.infotext],
            outputs=[self.cache_status],
            show_progress="hidden",
        )

    def connect_gallery(self, component):
        self.gallery = component.component

    def connect_infotext(self, component):
        self.infotext = component.component

    def get_cached_source(self, generation_info, gallery_index):
        cache_data = parse_reconstruction_cache(generation_info)
        if cache_data is None:
            raise gr.Error("The raw generated image is no longer available in the reconstruction cache.")
        info, token, indices = cache_data
        # A single generated result is unambiguous before the gallery preview is opened.
        if gallery_index == -1 and len(indices) == 1:
            gallery_index = int(next(iter(indices)))
        if not isinstance(gallery_index, int) or str(gallery_index) not in indices:
            raise gr.Error("Select a generated image in the img2img gallery first (not a grid or mask).")
        try:
            cache_index = int(indices[str(gallery_index)])
            cached = RECONSTRUCTION_CACHE.get(token, cache_index)
        except KeyError as error:
            raise gr.Error(str(error)) from error
        infotext_index = info.get("index_of_first_image", 0) + cache_index
        if infotext_index >= len(info.get("infotexts", [])):
            raise gr.Error("Generation information for the selected image is unavailable.")
        return info, cached, infotext_index, gallery_index

    def reconstruct_selected(
        self,
        generation_info,
        gallery,
        key_color,
        auto_key,
        opaque_auto,
        opaque,
        transparent_auto,
        transparent,
        shift_auto,
        shift,
        island_check,
        island_auto,
        island,
        island_radius,
        island_share,
        island_smoke,
        gallery_index,
    ):
        info, cached, infotext_index, gallery_index = self.get_cached_source(generation_info, gallery_index)
        settings = img2img_alpha.AlphaSettings(
            mode="Reconstruct",
            key_color=key_color or img2img_alpha.DEFAULT_KEY_COLOR,
            auto_key_color=bool(auto_key),
            opaque_threshold=None if opaque_auto else float(opaque),
            transparent_threshold=None if transparent_auto else float(transparent),
            color_shift=None if shift_auto else float(shift),
            island_check=bool(island_check),
            island_threshold=None if island_auto else float(island),
            island_radius=int(island_radius),
            island_share=float(island_share) / 100,
            island_smoke=bool(island_smoke),
        )
        settings.validate()
        background = img2img_alpha.background_color(cached.reference, settings.key_color, settings.auto_key_color)
        image, calibration = img2img_alpha.reconstruct(cached.image, cached.reference, background, settings)

        if cached.original is not None:
            if cached.mask is None:
                raise RuntimeError("The cached inpainting reconstruction is missing its mask.")
            if cached.paste_to is not None:
                x, y, width, height = cached.paste_to
                resized = images.resize_image(1, image, width, height, force_RGBA=True)
                canvas = Image.new("RGBA", cached.original.size)
                canvas.paste(resized, (x, y))
                image = canvas
            image = img2img_alpha.composite(image, cached.original, cached.mask)

        updated_gallery = list(gallery)
        _, caption = updated_gallery[gallery_index]
        updated_gallery[gallery_index] = (image, caption)
        infotext = info["infotexts"][infotext_index]
        values = calibration.infotext()
        for key in OPTIONAL_INFOTEXT_KEYS:
            values.setdefault(key, None)
        infotext = replace_infotext_values(infotext, values)
        info["infotexts"][infotext_index] = infotext
        info["extra_generation_params"].update(values)
        return (
            gr.update(value=updated_gallery, selected_index=gallery_index),
            json.dumps(info),
            plaintext_to_html(infotext),
            gr.skip(),
        )

    def save_original(self, generation_info, gallery, gallery_index):
        info, cached, infotext_index, _ = self.get_cached_source(generation_info, gallery_index)
        infotext = info["infotexts"][infotext_index]
        params = parse_generation_parameters(infotext)
        raw_info = replace_infotext_values(infotext, {"img2img alpha output": "raw before reconstruction"})
        # Folder and filename patterns read generation values from p, like the gallery Save button.
        p = SimpleNamespace(**info)
        image_index = infotext_index - info.get("index_of_first_image", 0)
        batch_size = info.get("batch_size") or 1
        p.batch_index = image_index % batch_size
        p.iteration = image_index // batch_size
        p.n_iter = max(1, len(info.get("all_seeds") or ()) // batch_size)
        filename, _ = images.save_image(
            cached.image.copy(),
            path=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
            basename="",
            seed=int(params.get("Seed", -1)),
            prompt=params.get("Prompt", ""),
            extension=shared.opts.samples_format,
            info=raw_info,
            p=p,
            suffix="-original",
        )
        return gr.update(value=f"Saved raw generated image to {filename}", visible=True)

    def setup(self, p, enable, mode, key_color, auto_key, opaque_auto, opaque, transparent_auto, transparent, shift_auto, shift, island_check=False, island_auto=True, island=0, island_radius=4, island_share=70, island_smoke=True):
        p.img2img_alpha_settings = img2img_alpha.AlphaSettings(
            mode=mode if enable else "Off",
            key_color=key_color or img2img_alpha.DEFAULT_KEY_COLOR,
            auto_key_color=bool(auto_key),
            opaque_threshold=None if opaque_auto else float(opaque),
            transparent_threshold=None if transparent_auto else float(transparent),
            color_shift=None if shift_auto else float(shift),
            island_check=bool(island_check),
            island_threshold=None if island_auto else float(island),
            island_radius=int(island_radius),
            island_share=float(island_share) / 100,
            island_smoke=bool(island_smoke),
        )

    def postprocess(self, p, processed, *args):
        if getattr(p, "img2img_alpha_mode", "Off") != "Reconstruct" or not p.alpha_raw_images:
            return
        if not (
            len(p.alpha_raw_images)
            == len(p.alpha_reconstruction_contexts)
            == len(p.alpha_gallery_indices)
        ):
            raise RuntimeError("The cached img2img alpha outputs and gallery indices do not match.")

        items = {}
        reference_count = len(p.alpha_references)
        if not reference_count or len(p.alpha_backgrounds) != reference_count:
            raise RuntimeError("The img2img alpha reconstruction references are incomplete.")
        for raw_index, (image, context, gallery_index) in enumerate(
            zip(p.alpha_raw_images, p.alpha_reconstruction_contexts, p.alpha_gallery_indices, strict=True)
        ):
            reference_index = raw_index % reference_count
            original, mask, paste_to = context if context is not None else (None, None, None)
            items[raw_index] = img2img_alpha.ReconstructionCacheItem(
                image=image,
                reference=p.alpha_references[reference_index],
                background=p.alpha_backgrounds[reference_index],
                original=original,
                mask=mask,
                paste_to=paste_to,
            )

        token = RECONSTRUCTION_CACHE.store(items)
        processed.extra_generation_params[RECONSTRUCTION_CACHE_KEY] = token
        processed.extra_generation_params[RECONSTRUCTION_CACHE_INDICES_KEY] = {
            str(gallery_index): raw_index for raw_index, gallery_index in enumerate(p.alpha_gallery_indices)
        }
