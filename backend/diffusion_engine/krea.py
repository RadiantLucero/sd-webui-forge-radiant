from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.misc.image_resize import resize_reference
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.krea2_engine import Qwen3VL4BEngine
from modules.shared import opts


class Krea2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VL4BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

        # Identity Edit and Moodboard state, armed by the sd-forge-krea2-edit / -moodboard extensions,
        # consumed by the next positive encode
        self.clear_edit()
        self.clear_moodboard()

    @property
    def moodboard_available(self) -> bool:
        """Kept for the extensions' vision check; Neo only loads the Qwen3-VL encoder with its vision tower."""
        return True

    def clear_edit(self, release_refs: bool = False):
        self.edit_images_vl: list[torch.Tensor] = []
        self.edit_ref_latents: list[torch.Tensor] = []
        self.edit_ref_boosts: list[float] = []
        self.edit_ref_fit: list[bool] = []
        self.edit_ref_mode: str | None = None
        self.edit_labels = False
        self.edit_instruction: str | None = None
        self.edit_system: str | None = None
        if release_refs:
            dynamic_args.ref_latents.clear()
            self._clear_ref_extras()

    @staticmethod
    def _clear_ref_extras():
        dynamic_args.krea2_ref_boosts = None
        dynamic_args.krea2_ref_fit = None
        dynamic_args.krea2_ref_mode = None
        dynamic_args.krea2_ref_kv = None

    @staticmethod
    def _fit_area(samples: torch.Tensor, max_pixels: int, snap: int = 1) -> torch.Tensor:
        """Downscale (never upscale) a (b, c, h, w) image to fit max_pixels, keeping aspect, dims snapped."""
        h, w = samples.shape[2], samples.shape[3]
        scale = min(1.0, math.sqrt(max_pixels / (w * h)))
        nw, nh = max(round(w * scale / snap) * snap, snap), max(round(h * scale / snap) * snap, snap)
        if (nh, nw) == (h, w):
            return samples
        return torch.nn.functional.interpolate(samples, size=(nh, nw), mode="area")

    @staticmethod
    def _fit_image_px(samples: torch.Tensor, px_h: int, px_w: int, fit_mode: str = "fit") -> torch.Tensor:
        """Resample the source to the output grid in pixel space before VAE encoding (v1.2 geometry).
        fit: AR-preserving fit inside the target, /16 snapped; within 8% AR it fills the grid via a minimal
        center-crop. crop: center-crop to the target AR, then resize."""
        img = samples.float()
        ih, iw = img.shape[-2:]
        if fit_mode == "fit":
            sc = min(px_h / ih, px_w / iw)
            if ih * sc >= px_h * 0.92 and iw * sc >= px_w * 0.92:
                s = max(px_h / ih, px_w / iw)
                ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
                y0, x0 = (ih - ch) // 2, (iw - cw) // 2
                img = img[..., y0 : y0 + ch, x0 : x0 + cw]
                nh, nw = px_h, px_w
            else:
                nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
                nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
            return torch.nn.functional.interpolate(img, size=(nh, nw), mode="bicubic", antialias=True).clamp(0, 1)
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img[..., y0 : y0 + ch, x0 : x0 + cw]
        return torch.nn.functional.interpolate(img, size=(px_h, px_w), mode="bicubic", antialias=True).clamp(0, 1)

    @torch.inference_mode()
    def arm_edit(self, images: list[torch.Tensor], grounding_px: int = 768, ref_boost: float = 1.0, ref_boost_a: float = 1.0, fit_size: tuple[int, int] | None = None, fit_mode: str = "fit", ref_mode: str | None = None, picture_labels: bool = False, vl_size: int | None = None, ref_area: int | None = None, instruction: str | None = None, system_prompt: str | None = None):
        """Arm identity-edit sources, each (1, H, W, C) in 0..1, scene first and subject last.
        grounding_px caps the longest side fed to Qwen3-VL. ref_boost scales target->reference attention for
        the last ref, ref_boost_a for the earlier ones. fit_size=(h, w) fits sources to the output grid in
        pixel space before encoding.
        ai-toolkit edit LoRAs (Re-render): ref_mode "kv" or "t0" conditions the refs at t=0 on their own grids,
        picture_labels writes "Picture N:" before each vision block, vl_size scales the vision input to vl_size^2
        area (up or down, as Easy_QwenEdit2509 does), ref_area caps the ref latents' area, instruction replaces the
        prompt text, system_prompt replaces the template's system instruction."""
        self.clear_edit()
        self.edit_ref_mode = ref_mode if ref_mode in ("kv", "t0") else None
        self.edit_labels = bool(picture_labels)
        self.edit_instruction = instruction.strip() if instruction and instruction.strip() else None
        self.edit_system = system_prompt.strip() if system_prompt and system_prompt.strip() else None
        for image in images:
            samples = image.movedim(-1, 1)[:, :3]  # b, c, h, w

            h, w = samples.shape[2], samples.shape[3]
            if vl_size:
                scale_by = math.sqrt(int(vl_size) ** 2 / (w * h))
                size = (max(64, round(h * scale_by)), max(64, round(w * scale_by)))
                vl = torch.nn.functional.interpolate(samples, size=size, mode="area")
            elif grounding_px and max(h, w) > grounding_px:
                scale_by = grounding_px / max(h, w)
                vl = torch.nn.functional.interpolate(samples, size=(round(h * scale_by), round(w * scale_by)), mode="area")
            else:
                vl = samples
            self.edit_images_vl.append(vl.movedim(1, -1))

            if fit_size is not None:
                enc = self._fit_image_px(samples, int(fit_size[0]), int(fit_size[1]), fit_mode)
            elif ref_area:
                enc = self._fit_area(samples, int(ref_area), snap=16)
            else:
                enc = torch.nn.functional.interpolate(samples, size=(round(h / 16) * 16, round(w / 16) * 16), mode="area")
            sample = self.forge_objects.vae.encode(enc.movedim(1, -1))
            self.edit_ref_latents.append(self.forge_objects.vae.first_stage_model.process_in(sample).squeeze(2))
            self.edit_ref_fit.append(fit_size is not None and fit_mode == "fit")

        n = len(self.edit_ref_latents)
        self.edit_ref_boosts = [float(ref_boost_a)] * max(0, n - 1) + [float(ref_boost)] if n else []
        boost_note = f", ref_boost {ref_boost:g}" + (f"/{ref_boost_a:g}" if n > 1 else "") if any(b != 1.0 for b in self.edit_ref_boosts) else ""
        fit_note = f", fit {fit_mode} -> {fit_size[1]}x{fit_size[0]}" if fit_size is not None else ""
        mode_note = f", refs at t=0 ({self.edit_ref_mode})" if self.edit_ref_mode else ""
        print(f"[Krea2 Edit] armed {n} source image(s), grounding_px {grounding_px}{boost_note}{fit_note}{mode_note}")

    # Sentences next to the moodboard block: K2 reads the whole context as a description of the target image
    KREA2_STYLE_DIRECTIVE = (
        "The image uses only the art style, color palette, lighting, texture, rendering technique and "
        "overall mood of the reference images. The subjects, objects and composition of the image come "
        "from the following text description alone. "
    )
    KREA2_SUBJECT_DIRECTIVE = (
        "The image depicts the same subjects, objects and composition as the reference images, "
        "rendered in the art style described by the following text. "
    )
    VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

    def clear_moodboard(self):
        self.images_vl: list[torch.Tensor] = []
        self.moodboard_strength = 1.0
        self.moodboard_position = "before"
        self.moodboard_extract = "style"
        self.moodboard_style_directive = False
        self.moodboard_hide_refs = False

    @staticmethod
    def _expand_style_crops(images: list[torch.Tensor], n: int = 2) -> list[torch.Tensor]:
        """Split each reference into an n x n grid of order-shuffled crops: composition cannot survive,
        palette / texture / lighting are in every crop."""
        orders = {
            2: (2, 0, 3, 1),
            4: (10, 3, 12, 5, 0, 15, 6, 9, 2, 13, 4, 11, 8, 1, 14, 7),
        }
        order = orders.get(n) or tuple(range(n * n))
        crops = []
        for image in images:
            h, w = image.shape[1] // n, image.shape[2] // n
            grid = [image[:, r * h : (r + 1) * h, c * w : (c + 1) * w] for r in range(n) for c in range(n)]
            crops.extend(grid[i] for i in order)
        return crops

    @torch.inference_mode()
    def arm_moodboard(self, images: list[torch.Tensor], strength: float = 1.0, position: str = "before", crops: int = 0, style_directive: bool = False, hide_refs: bool = False, extract: str = "style"):
        """Arm references (each (1, H, W, C) in 0..1) as style / vibe conditioning for the next positive encode.
        strength 1.0 = raw reference detail, lower = purer extract. extract "style" or "subject".
        position: image tokens "before" or "after" the prompt. crops: 0 whole, 2 quadrants, 4 fine tiles.
        hide_refs drops the reference rows from the conditioning (needs position "before")."""
        self.clear_moodboard()
        self.moodboard_strength = float(strength)
        self.moodboard_position = position
        self.moodboard_extract = "subject" if str(extract).startswith("subject") else "style"
        self.moodboard_style_directive = bool(style_directive)
        self.moodboard_hide_refs = bool(hide_refs)

        # vision pixel budget: 384x384 area (fast) or a 1 MP cap that never upscales
        high_detail = str(getattr(opts, "krea2_moodboard_vision_budget", "384")).startswith("1024")
        total = int(1024 * 1024) if high_detail else int(384 * 384)

        n_refs = len(images)
        if crops and crops > 1:
            images = self._expand_style_crops(images, n=int(crops))
            total = total // (3 if crops == 2 else 12)

        token_counts = []
        for image in images:
            samples = image.movedim(-1, 1)  # b, c, h, w
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            if high_detail:
                scale_by = min(1.0, scale_by)
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
            self.images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            token_counts.append(max(1, height // 32) * max(1, width // 32))  # patch 16, spatial merge 2

        crops_note = f" ({n_refs} reference(s) as {crops}x{crops} crops)" if crops and crops > 1 else ""
        directive_note = ", directive" if self.moodboard_style_directive else ""
        hide_note = ", refs hidden from DiT" if self.moodboard_hide_refs else ""
        print(f"[Krea2 Moodboard] armed {len(self.images_vl)} image(s){crops_note} packed into 1 vision span, ~{sum(token_counts)} vision tokens {token_counts}, extract {self.moodboard_extract}, strength {self.moodboard_strength:g}, position {self.moodboard_position}{directive_note}{hide_note}")

    def _moodboard_directive(self) -> str:
        if not self.moodboard_style_directive:
            return ""
        return self.KREA2_SUBJECT_DIRECTIVE if self.moodboard_extract == "subject" else self.KREA2_STYLE_DIRECTIVE

    def _encode_with_moodboard(self, texts: list[str], images: list, span_limit: int | None):
        """One encode with the moodboard effects and the native Qwen3-VL path switched on, then reset."""
        engine = self.text_processing_engine_qwen
        transformer = engine.qwen_vl_module
        native_vision_inputs = getattr(transformer, "native_vision_inputs", True)
        transformer.native_vision_inputs = bool(getattr(opts, "krea2_moodboard_native_vision", True))
        engine.moodboard_strength = self.moodboard_strength
        engine.moodboard_extract = self.moodboard_extract
        engine.moodboard_hide_refs = self.moodboard_hide_refs
        engine.moodboard_span_limit = span_limit
        try:
            return engine(texts, images=images)
        finally:
            transformer.native_vision_inputs = native_vision_inputs
            engine.moodboard_strength = 1.0
            engine.moodboard_extract = "style"
            engine.moodboard_hide_refs = False
            engine.moodboard_span_limit = None

    @torch.inference_mode()
    def get_learned_conditioning_moodboard(self, prompt: "SdConditioning"):
        directive = self._moodboard_directive()
        if self.moodboard_position == "after":
            texts = [line + (" " + directive if directive else "") + self.VISION_BLOCK for line in prompt]
        else:
            texts = [self.VISION_BLOCK + directive + line for line in prompt]
        # all references ride in ONE image entry, so they pack into one vision span (grid-safe)
        cond = self._encode_with_moodboard(texts, [list(self.images_vl)], span_limit=None)
        dynamic_args.ref_latents.clear()
        self._clear_ref_extras()
        self.clear_moodboard()
        return cond

    @torch.inference_mode()
    def get_learned_conditioning_edit(self, prompt: "SdConditioning"):
        # The negative is grounded too (training's unconditional is an empty prompt + the same image), needed
        # for CFG > 1. It runs first and leaves the armed state; the positive consumes it and publishes the refs.
        # An armed moodboard fuses in front on the positive only: style from the moodboard, identity from the edit.
        if self.images_vl and not prompt.is_negative_prompt:
            lead = self.VISION_BLOCK + self._moodboard_directive() + self.VISION_BLOCK * len(self.edit_images_vl)
            images = [list(self.images_vl)] + list(self.edit_images_vl)
            cond = self._encode_with_moodboard([lead + line for line in prompt], images, span_limit=1)
        else:
            lines = list(prompt)
            if self.edit_instruction is not None and not prompt.is_negative_prompt:
                lines = [self.edit_instruction] * len(lines)
            if self.edit_labels:
                lead = "".join(f"Picture {i + 1}: {self.VISION_BLOCK}" for i in range(len(self.edit_images_vl)))
                lines = [lead + line for line in lines]
            engine = self.text_processing_engine_qwen
            engine.system_prompt = self.edit_system
            try:
                cond = engine(lines, images=list(self.edit_images_vl))
            finally:
                engine.system_prompt = None
        if not prompt.is_negative_prompt:
            dynamic_args.ref_latents = list(self.edit_ref_latents)
            dynamic_args.krea2_ref_boosts = list(self.edit_ref_boosts)
            dynamic_args.krea2_ref_fit = list(self.edit_ref_fit)
            dynamic_args.krea2_ref_mode = self.edit_ref_mode
            dynamic_args.krea2_ref_kv = {}
            self.ini_latent = None
            self.clear_edit()
            self.clear_moodboard()
        return cond

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if self.edit_images_vl:
            return self.get_learned_conditioning_edit(prompt)

        if self.images_vl and not prompt.is_negative_prompt:
            return self.get_learned_conditioning_moodboard(prompt)

        if not prompt.is_negative_prompt:
            self._clear_ref_extras()
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if opts.krea2_do_reference and bool(_references):
                return self.get_learned_conditioning_with_image(prompt, _references)
            else:
                dynamic_args.ref_latents.clear()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents = [], []
        for image in images:
            v, r = self.encode_vision(image)
            images_vl.append(v)
            ref_latents.append(r.squeeze(2))

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w

        _vision = resize_reference(samples, 1, 768).movedim(1, -1)[:, :, :, :3]

        s = resize_reference(samples)
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        return (_vision, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if opts.krea2_do_reference:
            self.store_reference(x[:1].movedim(1, -1).mul(0.5).add(0.5).cpu())

        if not dynamic_args.is_referencing:
            return super().encode_first_stage(x)
