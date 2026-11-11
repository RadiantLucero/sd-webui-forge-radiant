# https://github.com/Comfy-Org/ComfyUI/blob/v0.36.0/comfy/text_encoders/krea2.py

import numbers

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from ._comfy import EMBEDDINGS, INF, TOKEN_WEIGHTS, SDClipModel, SDTokenizer


class Qwen3VL4BEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], layer_idx=None, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

        # Krea2 Moodboard state, set by the diffusion engine around one encode
        self.moodboard_strength = 1.0
        self.moodboard_extract = "style"
        self.moodboard_hide_refs = False
        self.moodboard_span_limit = None

        # system instruction override for one encode (Krea2 Re-render: the Qwen edit instruction)
        self.system_prompt = None

    @property
    def qwen_vl_module(self):
        """The Qwen3VL module behind the clip wrapper (holds native_vision_inputs and last_embeds_info)."""
        module = self.text_encoder.transformer
        return module if hasattr(module, "build_image_inputs") else getattr(module, "transformer", module)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        if isinstance(texts, str):
            return self.tokenizer.tokenizer(self.llama_template.format(texts))["input_ids"]
        else:
            return [self.tokenizer.tokenizer(self.llama_template.format(t))["input_ids"] for t in texts]

    def __call__(self, texts: list[str], images: list[torch.Tensor] = []) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunk = self._tokenize_with_weights(line, images)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cond = self._apply_moodboard_effects(cond)
                tok_pairs = chunk[0]

                count_im_start = 0
                template_end = -1

                if template_end == -1:
                    for i, v in enumerate(tok_pairs):
                        elem = v[0]
                        if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                            if elem == 151644 and count_im_start < 2:
                                template_end = i
                                count_im_start += 1
                    if cond.shape[2] > (template_end + 3):
                        if tok_pairs[template_end + 1][0] == 872:
                            if tok_pairs[template_end + 2][0] == 198:
                                template_end += 3

                cond = cond[:, :, template_end:]
                cond = cond.permute(0, 2, 1, 3).reshape(-1, 12, 2560)
                cache[line] = cond

            zs.append(cond)

        return zs

    def _tokenize_with_weights(self, text: str, images: list[torch.Tensor]) -> TOKEN_WEIGHTS:
        # texts that carry their own vision blocks (Krea2 Moodboard) keep them where they are
        blocks = "" if self.vision_block in text else self.vision_block * len(images)
        template = self.llama_template
        if self.system_prompt:
            template = "<|im_start|>system\n" + self.system_prompt.replace("{", "{{").replace("}", "}}") + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        llama_text = template.format(blocks + text.strip())
        tokens = self.tokenizer.tokenize_with_weights(llama_text, disable_weights=True)

        embed_count = 0

        for r in tokens:
            for i in range(len(r)):
                if isinstance(r[i][0], (int, float)) and r[i][0] == 151655:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1

        return tokens

    def _apply_moodboard_effects(self, z: torch.Tensor) -> torch.Tensor:
        """Krea2 Moodboard post-processing of the tapped hidden states (1, layers, seq, dim), before the template strip.

        hide_refs drops the vision rows, so style reaches the DiT only through the prompt tokens.
        strength < 1 with extract "style" lerps vision rows toward a mean / mean+std / mean-std cycle of the
        joint statistics (palette, texture, mood); "subject" lerps toward whitened rows (structure stays).
        moodboard_span_limit keeps the effects on the leading spans when edit spans follow."""
        hide = self.moodboard_hide_refs
        strength = float(self.moodboard_strength)
        if not hide and strength >= 1.0:
            return z

        info = getattr(self.qwen_vl_module, "last_embeds_info", None) or []
        spans = [(e["index"], e["index"] + e["size"]) for e in info if e.get("type") == "image"]
        spans = [(a, b) for a, b in spans if 0 <= a < b <= z.shape[2]]
        if self.moodboard_span_limit is not None:
            spans = spans[: self.moodboard_span_limit]
        if not spans:
            return z

        if hide:
            keep = torch.ones(z.shape[2], dtype=torch.bool, device=z.device)
            for a, b in spans:
                keep[a:b] = False
            return z[:, :, keep]

        strength = max(0.0, strength)
        joint = torch.cat([z[:, :, a:b] for a, b in spans], dim=2)
        mu = joint.mean(dim=2, keepdim=True)
        sigma = joint.std(dim=2, keepdim=True)
        z = z.clone()
        for a, b in spans:
            span = z[:, :, a:b]
            if self.moodboard_extract == "subject":
                target = (span - mu) / sigma.clamp_min(1e-4)
            else:
                n = span.shape[2]
                coef = torch.tensor([0.0, 1.0, -1.0], device=span.device, dtype=span.dtype).repeat((n + 2) // 3)[:n].view(1, 1, n, 1)
                target = mu + coef * sigma
            z[:, :, a:b] = target + strength * (span - target)
        return z
