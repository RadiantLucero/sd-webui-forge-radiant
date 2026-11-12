import ast
import json
import math
import re
import tempfile
import unittest
from abc import abstractmethod
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
import torch
from PIL import Image

from modules import img2img_alpha

ROOT = Path(__file__).resolve().parents[1]


def load_logic(path, namespace, names=None, methods=None):
    """Execute real source definitions without importing the GPU runtime."""
    tree = ast.parse((ROOT / path).read_text())
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if names is not None and getattr(node, "name", None) not in names:
            continue
        if isinstance(node, ast.ClassDef) and methods is not None:
            node.body = [method for method in node.body if getattr(method, "name", None) in methods]
            node.decorator_list = []
        body.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *body], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)  # noqa: S102 -- fixed repository source, not user input


class Prompts(list):
    is_negative_prompt = False


class Krea2ConditioningTests(unittest.TestCase):
    def setUp(self):
        self.dynamic = SimpleNamespace(ref_latents=[], last_extra_generation_params={})
        opts = SimpleNamespace(
            data={}, CLIP_stop_at_last_layers=1, sdxl_crop_left=0, sdxl_crop_top=0,
            emphasis="Original", krea2_do_reference=True, persistent_cond_cache=True,
        )
        self.scope = {
            "torch": torch, "math": math, "abstractmethod": abstractmethod,
            "model_list": SimpleNamespace(Krea2=object()),
            "memory_management": SimpleNamespace(load_model_gpu=Mock()),
            "dynamic_args": self.dynamic, "opts": opts,
        }
        load_logic("backend/diffusion_engine/base.py", self.scope, {"ForgeDiffusionEngine"})
        load_logic("backend/diffusion_engine/krea.py", self.scope, {"Krea2"})
        self.model = self.scope["Krea2"].__new__(self.scope["Krea2"])
        config = SimpleNamespace(latent_format=None, inpaint_model=lambda: False)
        self.scope["ForgeDiffusionEngine"].__init__(self.model, config, {"vae": SimpleNamespace()})
        self.model.clear_edit()
        self.model.clear_moodboard()
        self.model._clear_ref_extras()
        self.model.extra_generation_params = {}
        self.encoder = Mock(side_effect=lambda texts, images=(): {"texts": list(texts), "images": list(images)})
        self.model.text_processing_engine_qwen = self.encoder
        self.model.forge_objects = SimpleNamespace(
            clip=SimpleNamespace(patcher=None),
            vae=SimpleNamespace(
                encode=lambda image: image.movedim(-1, 1).unsqueeze(2),
                first_stage_model=SimpleNamespace(process_in=lambda sample: sample),
            ),
        )
        self.scope.update({
            "shared": SimpleNamespace(sd_model=self.model),
            "sd_models": SimpleNamespace(checkpoint_aliases={}, model_data=SimpleNamespace(forge_loading_parameters={})),
            "args": SimpleNamespace(dynamic_args=self.dynamic),
            "StableDiffusionProcessingImg2Img": type("Img2Img", (), {}),
            "manage_model_and_prompt_cache": Mock(),
            "set_config": Mock(),
            "sd_samplers": SimpleNamespace(fix_p_invalid_sampler_and_scheduler=Mock()),
            "profiling": SimpleNamespace(Profiler=nullcontext),
        })
        load_logic(
            "modules/processing.py", self.scope,
            {"StableDiffusionProcessing", "process_images"},
            {"cached_params", "get_conds_with_caching", "clear_prompt_cache"},
        )
        self.p = self.scope["StableDiffusionProcessing"]()
        self.p.__dict__.update(
            sd_model=self.model, width=16, height=16, distilled_cfg_scale=1,
            scripts=None, override_settings={}, override_settings_restore_afterwards=False,
        )
        self.p.clear_prompt_cache()
        self.image = torch.zeros((1, 16, 16, 3))

    def conditioning(self, negative=False):
        prompts = Prompts(["same prompt"])
        prompts.is_negative_prompt = negative
        return self.p.get_conds_with_caching(
            lambda model, texts, steps, hires: model.get_learned_conditioning(texts),
            prompts, 2, [self.p.cached_uc if negative else self.p.cached_c], {},
        )

    def generate(self, operation):
        self.scope["process_images_inner"] = lambda p: operation()
        return self.scope["process_images"](self.p)

    def test_edit_and_moodboard_do_not_reuse_plain_conditioning(self):
        for mode in ("edit", "moodboard"):
            with self.subTest(mode=mode):
                self.generate(self.conditioning)

                def armed():
                    getattr(self.model, f"arm_{mode}")([self.image])
                    negative = self.conditioning(negative=True)
                    if mode == "edit":
                        self.assertTrue(negative["images"])
                    result = self.conditioning()
                    calls = self.encoder.call_count
                    self.assertIs(self.conditioning(), result)
                    self.assertEqual(self.encoder.call_count, calls)
                    self.assertTrue(result["images"])
                    if mode == "edit":
                        self.assertTrue(self.dynamic.ref_latents)
                    return result

                self.generate(armed)
                self.assertFalse(self.dynamic.ref_latents)
                plain = self.generate(self.conditioning)
                self.assertFalse(plain["images"])
                calls = self.encoder.call_count
                self.assertIs(self.generate(self.conditioning), plain)
                self.assertEqual(self.encoder.call_count, calls)

    def test_new_edit_with_same_prompt_encodes_the_new_reference(self):
        results = []
        for value in (0.0, 1.0):
            def edit():
                self.model.arm_edit([self.image + value])
                results.append(self.conditioning())
            self.generate(edit)
        self.assertEqual(results[0]["images"][0].mean().item(), 0.0)
        self.assertEqual(results[1]["images"][0].mean().item(), 1.0)

    def test_failure_clears_pending_and_published_edit_state(self):
        for encoded in (False, True):
            with self.subTest(encoded=encoded):
                def fail():
                    self.model.arm_edit([self.image], ref_mode="kv")
                    self.model.arm_moodboard([self.image])
                    if encoded:
                        self.conditioning()
                    self.model.ini_latent = self.image
                    raise RuntimeError("sampling failed")

                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    self.generate(fail)
                self.assertFalse(self.model.edit_images_vl)
                self.assertFalse(self.model.edit_ref_latents)
                self.assertFalse(self.model.images_vl)
                self.assertFalse(self.dynamic.ref_latents)
                self.assertIsNone(self.model.ini_latent)
                for name in ("krea2_ref_boosts", "krea2_ref_fit", "krea2_ref_mode", "krea2_ref_kv"):
                    self.assertIsNone(getattr(self.dynamic, name))
                self.assertFalse(self.generate(self.conditioning)["images"])

    def test_interrupted_edit_does_not_break_cached_imagestitch_references(self):
        self.model.ref_latents = [self.image]
        self.model.encode_vision = Mock(return_value=(self.image, torch.ones((1, 3, 1, 16, 16))))
        self.generate(self.conditioning)

        def fail_before_encoding():
            self.model.arm_edit([self.image])
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.generate(fail_before_encoding)

        def next_generation():
            self.conditioning()
            self.assertTrue(self.dynamic.ref_latents)

        self.generate(next_generation)


class TransparencyCallbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.saved_path = Path(self.directory) / "raw.png"

        def save_image(image, **kwargs):
            image.save(self.saved_path)
            return str(self.saved_path), None

        scope = {
            "gr": gr, "Image": Image, "json": json, "re": re, "img2img_alpha": img2img_alpha,
            "scripts": SimpleNamespace(Script=object),
            "images": SimpleNamespace(save_image=Mock(side_effect=save_image)),
            "shared": SimpleNamespace(opts=SimpleNamespace(outdir_samples="", outdir_img2img_samples=self.directory, samples_format="png")),
            "parse_generation_parameters": Mock(return_value={"Seed": "1", "Prompt": "test"}),
            "plaintext_to_html": lambda text: text,
        }
        load_logic("modules/infotext_utils.py", scope, {"quote"})
        load_logic("modules/processing_scripts/img2img_transparency.py", scope)
        script_type = scope["ScriptImg2imgTransparency"]
        script = script_type.__new__(script_type)
        self.scope = scope
        self.script = script
        self.settings = ["#00ff00", False, False, 40, False, 200, False, 255, False, False, 0, 4, 70, True]
        with gr.Blocks(analytics_enabled=False):
            script.gallery = gr.Gallery(type="pil", format="png")
            script.gallery.GRADIO_CACHE = self.directory
            script.infotext = gr.HTML()
            script.calculated_values = gr.Markdown()
            script.last_values = gr.State()
            script.cache_status = gr.Textbox()
            script.auto_inputs = [gr.Checkbox() for _ in range(4)]
            script.slider_outputs = [gr.Slider() for _ in range(4)]
            script.reconstruct_button = gr.Button()
            script.save_original_button = gr.Button()
            script.reconstruction_inputs = [
                gr.Checkbox() if isinstance(value, bool) else gr.Textbox() if isinstance(value, str) else gr.Number()
                for value in self.settings
            ]
            with patch.object(script.reconstruct_button, "click") as reconstruct, patch.object(script.save_original_button, "click") as save:
                script.connect_generation_info(SimpleNamespace(component=gr.Textbox()))
                self.reconstruct = reconstruct.call_args.kwargs
                self.save = save.call_args.kwargs

        self.raw = [Image.new("RGB", (16, 16), color) for color in ("#00ff00", "#ff0000")]
        reference = Image.new("RGBA", (16, 16))
        token = scope["RECONSTRUCTION_CACHE"].store({
            index: img2img_alpha.ReconstructionCacheItem(image, reference, (0, 255, 0))
            for index, image in enumerate(self.raw)
        })
        self.info = {
            "index_of_first_image": 1,
            "infotexts": ["grid", "first\nSteps: 1, Seed: 1", "second\nSteps: 1, Seed: 2"],
            "extra_generation_params": {
                scope["RECONSTRUCTION_CACHE_KEY"]: token,
                scope["RECONSTRUCTION_CACHE_INDICES_KEY"]: {"1": 0, "4": 1},
            },
        }
        self.gallery = [(self.raw[0], None)] * 4 + [(self.raw[1], None)]

    def invoke(self, binding, index, settings=()):
        gallery = self.script.gallery.postprocess(self.gallery)
        # The final number is supplied by selected_gallery_index() in the browser.
        inputs = [json.dumps(self.info), gallery, *settings, index]
        values = [component.preprocess(value) for component, value in zip(binding["inputs"], inputs, strict=True)]
        for image, _ in values[1]:
            image.load()
        return binding["fn"](*values)

    def test_reconstruct_and_save_use_preprocessed_gallery_selection(self):
        for index, raw_index in ((1, 0), (4, 1)):
            with self.subTest(index=index):
                gallery_update, info, _, _ = self.invoke(self.reconstruct, index, self.settings)
                self.assertEqual(gallery_update["selected_index"], index)
                result = gallery_update["value"][index][0]
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.getpixel((0, 0))[3], 0 if raw_index == 0 else 255)
                self.assertIn("img2img alpha opaque: 40", json.loads(info)["infotexts"][raw_index + 1])
                self.invoke(self.save, index)
                with Image.open(self.saved_path) as saved:
                    self.assertEqual(saved.tobytes(), self.raw[raw_index].tobytes())

    def test_single_output_with_empty_prompt(self):
        self.info["index_of_first_image"] = 0
        self.info["infotexts"] = ["Steps: 1, Seed: 1"]
        self.info["extra_generation_params"][self.scope["RECONSTRUCTION_CACHE_INDICES_KEY"]] = {"0": 0}
        self.gallery = [(self.raw[0], None)]
        _, info, _, _ = self.invoke(self.reconstruct, 0, self.settings)
        self.assertIn("img2img alpha opaque: 40", json.loads(info)["infotexts"][0])
        self.invoke(self.save, 0)
        with Image.open(self.saved_path) as saved:
            self.assertEqual(saved.tobytes(), self.raw[0].tobytes())

    def test_grid_and_mask_are_not_treated_as_generated_images(self):
        for index in (0, 2):
            for binding, settings in ((self.reconstruct, self.settings), (self.save, ())):
                with self.subTest(index=index, callback=binding["fn"].__name__):
                    with self.assertRaisesRegex(gr.Error, "Select a generated image"):
                        self.invoke(binding, index, settings)


if __name__ == "__main__":
    unittest.main()
