import json

import gradio as gr

from modules import img2img_alpha, scripts
from modules.infotext_utils import PasteField, parse_generation_parameters
from modules.ui_components import FormColorPicker, InputAccordion

THRESHOLD_KEYS = (
    ("opaque", "img2img alpha opaque", "Partial above"),
    ("transparent", "img2img alpha transparent", "Transparent from"),
    ("shift", "img2img alpha color shift", "Color shift"),
)


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
        values = {name: parse_threshold(params[key]) for name, key, _ in THRESHOLD_KEYS if key in params}
        if len(values) == len(THRESHOLD_KEYS) and all(values.values()):
            results.append((values, params.get("img2img alpha statistics", "")))
    return results


def describe(results):
    if not results:
        return "No reconstructed transparency in the last generation."

    def cell(value):
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
                with gr.Row():
                    key_color = FormColorPicker(label="Key color", value=img2img_alpha.DEFAULT_KEY_COLOR, elem_id=self.elem_id("key_color"))
                    auto_key = gr.Checkbox(label="Auto key color", value=False, elem_id=self.elem_id("auto_key"), info="choose the saturated color least present in the visible input")

                sliders = {}
                autos = {}
                for name, minimum, maximum, value, label, info in (
                    ("opaque", 0, 254, 40, "Partial transparency above key dominance", "pixels at or below this value stay fully opaque and unchanged"),
                    ("transparent", 1, 255, 200, "Transparent from key dominance", "pixels at or above this value become fully transparent"),
                    ("shift", 1, 255, 180, "Color shift", "key dominance of a fully transparent pixel when unmixing; lower values give more transparent edges tinted away from the key, higher values more opaque edges tinted toward it"),
                ):
                    with gr.Row(equal_height=True):
                        autos[name] = gr.Checkbox(label="Auto", value=True, elem_id=self.elem_id(f"{name}_auto"), scale=0, min_width=80)
                        sliders[name] = gr.Slider(label=label, minimum=minimum, maximum=maximum, step=1, value=value, elem_id=self.elem_id(name), info=info, scale=1)
                    sliders[name].input(fn=lambda: False, outputs=[autos[name]], show_progress="hidden")

                self.calculated_values = gr.Markdown(describe([]), elem_id=self.elem_id("calculated_values"))
                self.last_values = gr.State(None)
                use_values = gr.Button("Use last calculated values", elem_id=self.elem_id("use_values"))

        mode.change(fn=lambda value: gr.update(visible=value == "Reconstruct"), inputs=[mode], outputs=[reconstruct_controls], show_progress="hidden")

        names = [name for name, _, _ in THRESHOLD_KEYS]
        self.auto_inputs = [autos[name] for name in names]
        self.slider_outputs = [sliders[name] for name in names]

        def use_last_values(values):
            if not values:
                return [gr.skip()] * (len(names) * 2)
            return [gr.update(value=min(max(values[name], sliders[name].minimum), sliders[name].maximum)) for name in names] + [False] * len(names)

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

        self.infotext_fields = [
            PasteField(enable, lambda params: "img2img alpha" in params),
            PasteField(mode, lambda params: params.get("img2img alpha")),
            PasteField(key_color, paste_key_color),
            PasteField(auto_key, lambda params: False if "img2img alpha background" in params else None),
        ]
        for name, key, _ in THRESHOLD_KEYS:
            self.infotext_fields.append(PasteField(sliders[name], paste_threshold(key, sliders[name])))
            self.infotext_fields.append(PasteField(autos[name], paste_threshold(key, None)))

        return [enable, mode, key_color, auto_key, autos["opaque"], sliders["opaque"], autos["transparent"], sliders["transparent"], autos["shift"], sliders["shift"]]

    def connect_generation_info(self, component):
        generation_info = component.component

        def show_calculated_values(info, *automatic):
            results = last_calculated_values(info)
            if not results:
                return [describe(results), None] + [gr.skip()] * len(THRESHOLD_KEYS)

            first = {name: value for name, (value, _) in results[0][0].items()}
            sliders = [gr.update(value=min(max(first[name], slider.minimum), slider.maximum)) if auto else gr.skip() for (name, _, _), slider, auto in zip(THRESHOLD_KEYS, self.slider_outputs, automatic)]
            return [describe(results), first] + sliders

        generation_info.change(
            fn=show_calculated_values,
            inputs=[generation_info] + self.auto_inputs,
            outputs=[self.calculated_values, self.last_values] + self.slider_outputs,
            show_progress="hidden",
        )

    def setup(self, p, enable, mode, key_color, auto_key, opaque_auto, opaque, transparent_auto, transparent, shift_auto, shift):
        p.img2img_alpha_settings = img2img_alpha.AlphaSettings(
            mode=mode if enable else "Off",
            key_color=key_color or img2img_alpha.DEFAULT_KEY_COLOR,
            auto_key_color=bool(auto_key),
            opaque_threshold=None if opaque_auto else float(opaque),
            transparent_threshold=None if transparent_auto else float(transparent),
            color_shift=None if shift_auto else float(shift),
        )
