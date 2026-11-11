# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/ldm/krea2/model.py
# https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import EmbedND, timestep_embedding
from backend.operations import repeat_kv_for_gqa
from backend.quant_ops import ck
from backend.utils import pad_to_patch_size


def _imgids(bs: int, frame: int, h_: int, w_: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _refids(bs: int, frame: int, gh: int, gw: int, th: int, tw: int, device: torch.device) -> torch.Tensor:
    """Stride-1 ids for a (gh, gw) reference centered in a (th, tw) target; equals _imgids when the grids match."""
    ids = _imgids(bs, frame, gh, gw, device)
    ids[..., 1] += max(0, (th - gh) // 2)
    ids[..., 2] += max(0, (tw - gw) // 2)
    return ids


def _align_last(values, n: int, default) -> list:
    """Pad from the left so the given values map onto the last n refs (last ref = the subject)."""
    values = list(values or [])[-n:] if n else []
    return [default] * (n - len(values)) + values


def _ref_attn_bias(boosts, txtlen, slens, tgtlen, device, dtype) -> torch.Tensor:
    """ref_boost: log(boost) added to target->reference attention logits on [text | refs... | target]."""
    offs = [txtlen]
    for sl in slens:
        offs.append(offs[-1] + sl)
    rows0 = offs[-1]
    bias = torch.zeros(1, 1, rows0 + tgtlen, rows0 + tgtlen, device=device, dtype=dtype)
    for i, b in enumerate(boosts):
        if b != 1.0:
            bias[:, :, rows0:, offs[i] : offs[i] + slens[i]] = math.log(max(b, 1e-4))
    return bias


def _attn_kv(attn, x, freqs, kv_capture=None, kv_cache=None, transformer_options={}):
    """Attention.forward with optional post-RoPE K/V capture, or cached K/V appended as extra keys."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = ck.apply_rope(q, k, freqs)
    if kv_capture is not None:
        kv_capture.append((k, v))
    if kv_cache is not None:
        k = torch.cat((k, kv_cache[0].to(k.dtype)), dim=2)
        v = torch.cat((v, kv_cache[1].to(v.dtype)), dim=2)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = attention_function(q, k, v, attn.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return attn.wo(out * F.sigmoid(gate))


def _block_kv(block, x, vec, freqs, kv_capture=None, kv_cache=None, transformer_options={}):
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    x = x + pregate * _attn_kv(block.attn, (1 + prescale) * block.prenorm(x) + preshift, freqs, kv_capture, kv_cache, transformer_options)
    return x + postgate * block.mlp((1 + postscale) * block.postnorm(x) + postshift)


def _block_t0(block, x, vec, refvec, split, freqs, transformer_options={}):
    """Per-span modulation: tokens [:split] (text + target) at the real t, tokens [split:] (refs) at t=0."""
    m, r = block.mod(vec), block.mod(refvec)

    def mod(h, scale, shift):
        return torch.cat(((1 + m[scale]) * h[:, :split] + m[shift], (1 + r[scale]) * h[:, split:] + r[shift]), dim=1)

    def gate(h, g):
        return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

    x = x + gate(block.attn(mod(block.prenorm(x), 0, 1), freqs, None, transformer_options=transformer_options), 2)
    return x + gate(block.mlp(mod(block.postnorm(x), 3, 4)), 5)


def _ref_key(refs, bs) -> tuple:
    return (bs,) + tuple((tuple(r.shape), float(r.float().sum()), float(r.float().square().sum())) for r in refs)


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def fp32_weight(self, x):
        return cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.fp32_weight(x).to(x.dtype), eps=self.eps)

    def modulated(self, x, scale, shift):
        if torch.is_grad_enabled():
            return torch.addcmul(shift, 1 + scale, self(x))
        # Fold the learned RMSNorm weight into the AdaLN scale before rounding.
        scale = (self.fp32_weight(x) * (1 + scale.float()) - 1).to(x.dtype)
        return ck.rms_adaln(x, scale, shift, self.eps)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = q.unflatten(-1, (self.heads, self.headdim))
        k = k.unflatten(-1, (self.kvheads, self.headdim))
        v = v.unflatten(-1, (self.kvheads, self.headdim))
        if freqs is not None and not torch.is_grad_enabled():
            qnorm, knorm = self.qknorm.qnorm, self.qknorm.knorm
            q, k = ck.rms_rope_(q, k, freqs.transpose(1, 2), qnorm.fp32_weight(q), knorm.fp32_weight(k), qnorm.eps)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
        else:
            q, k = self.qknorm(q.transpose(1, 2), k.transpose(1, 2))
            if freqs is not None:
                q, k = ck.apply_rope(q, k, freqs)
        k, v = repeat_kv_for_gqa(k, v.transpose(1, 2), self.heads, 1)
        out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x.add_(self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options))
        x.add_(self.mlp(self.postnorm(x)))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = self.projector(x.unflatten(0, (b, l)).transpose(2, 3)).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(self.prenorm.modulated(x, prescale, preshift), freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(self.postnorm.modulated(x, postscale, postshift)))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()

        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        return self.linear(self.norm.modulated(x, scale, shift))


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        **kwargs,
    ):
        super().__init__()

        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    def _pack_refs(self, refs: list[torch.Tensor], bs: int, dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference tokens and ids: ref i at RoPE frame i + 1 on its own grid from 0."""
        tokens, ids = [], []
        for i, ref in enumerate(refs):
            ref = pad_to_patch_size(ref.to(device=device, dtype=dtype), (self.patch, self.patch), padding_mode="replicate")
            if ref.shape[0] != bs:
                ref = ref[:1].repeat(bs, *([1] * (ref.ndim - 1)))
            gh, gw = ref.shape[-2] // self.patch, ref.shape[-1] // self.patch
            tokens.append(rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=self.patch, pw=self.patch))
            ids.append(_imgids(bs, i + 1, gh, gw, device))
        return torch.cat(tokens, dim=1), torch.cat(ids, dim=1)

    def _forward_ai_toolkit(self, x, H_orig, W_orig, timesteps, context, refs, mode, transformer_options):
        """ai-toolkit krea2 edit conditioning (as the Ostris ComfyUI patch runs it).

        kv: the refs run alone at t=0 once per run and every block's ref K/V is appended as extra keys on
        each step (LoRAs trained with kv_cache, refs attend only to each other).
        t0: the refs ride in the sequence after the target, modulated at t=0."""
        bs = x.shape[0]
        patch = self.patch
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch
        device = x.device

        img = self.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)
        t0 = self.tmlp(timestep_embedding(torch.zeros_like(timesteps), self.tdim).unsqueeze(1).to(img.dtype))
        tvec0 = self.tproj(t0)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)
        txtlen, imglen = context.shape[1], img.shape[1]

        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
        imgpos = _imgids(bs, 0, h_, w_, device)

        if mode == "kv":
            cache = getattr(dynamic_args, "krea2_ref_kv", None)
            if cache is None:
                cache = dynamic_args.krea2_ref_kv = {}
            key = _ref_key(refs, bs)
            ref_kv = cache.get(key)
            if ref_kv is None:
                reftok, refpos = self._pack_refs(refs, bs, img.dtype, device)
                h = self.first(reftok)
                rfreqs = self.pe_embedder(refpos)
                ref_kv = []
                for block in self.blocks:
                    cap = []
                    h = _block_kv(block, h, tvec0, rfreqs, kv_capture=cap, transformer_options=transformer_options)
                    ref_kv.append(cap[0])
                cache[key] = ref_kv
            combined = torch.cat((context, img), dim=1)
            freqs = self.pe_embedder(torch.cat((txtpos, imgpos), dim=1))
            for block, kv in zip(self.blocks, ref_kv):
                combined = _block_kv(block, combined, tvec, freqs, kv_cache=kv, transformer_options=transformer_options)
        else:
            reftok, refpos = self._pack_refs(refs, bs, img.dtype, device)
            combined = torch.cat((context, img, self.first(reftok)), dim=1)
            freqs = self.pe_embedder(torch.cat((txtpos, imgpos, refpos), dim=1))
            split = txtlen + imglen
            for block in self.blocks:
                combined = _block_t0(block, combined, tvec, tvec0, split, freqs, transformer_options=transformer_options)

        final = self.last(combined[:, txtlen : txtlen + imglen], t)
        out = rearrange(final, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        return out[:, :, :H_orig, :W_orig]

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        x = x.squeeze(2)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch

        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents

        # ai-toolkit edit LoRAs (Krea2 Re-render): refs conditioned at t=0, own grids
        ref_mode = getattr(dynamic_args, "krea2_ref_mode", None)
        if ref_latents and ref_mode in ("kv", "t0"):
            return self._forward_ai_toolkit(x, H_orig, W_orig, timesteps, context, ref_latents, ref_mode, transformer_options).unsqueeze(2)

        # Krea2 Identity Edit extras: per-ref attention boost and fit flags, aligned to the LAST refs
        n_refs = len(ref_latents)
        ref_boosts = _align_last(getattr(dynamic_args, "krea2_ref_boosts", None), n_refs, 1.0)
        ref_fit = _align_last(getattr(dynamic_args, "krea2_ref_fit", None), n_refs, False)

        if (_edit := bool(ref_latents)) is True:
            refs = []
            ref_grids = []
            for i, ref in enumerate(ref_latents):
                if ref.shape[0] != x.shape[0]:  # batch_cond_uncond / batch size
                    ref = ref[:1].repeat(x.shape[0], *([1] * (ref.ndim - 1)))
                # fit-prepared refs that fit inside the target keep their own grid
                native = ref_fit[i] and ref.shape[-2] <= H and ref.shape[-1] <= W
                if ref.shape != x.shape and not native:
                    ref = adaptive_resize(ref, W, H, "area", "center")
                ref = pad_to_patch_size(ref.to(x), (patch, patch), padding_mode="replicate")
                refs.append(ref)
                ref_grids.append((ref.shape[-2] // patch, ref.shape[-1] // patch))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        if _edit:
            refs = [rearrange(r, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch) for r in refs]
            refs = [self.first(r) for r in refs]

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        if _edit:
            reflen = sum(ref.shape[1] for ref in refs)
            combined = torch.cat((context, *refs, img), dim=1)
        else:
            reflen = 0
            combined = torch.cat((context, img), dim=1)

        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

        if _edit:
            refids = [_refids(bs, i + 1, gh, gw, h_, w_, device) for i, (gh, gw) in enumerate(ref_grids)]
            imgids = _imgids(bs, 0, h_, w_, device)
            pos = torch.cat((txtpos, *refids, imgids), dim=1)
        else:
            imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
            imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
            imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
            imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
            pos = torch.cat((txtpos, imgpos), dim=1)

        freqs = self.pe_embedder(pos)

        attn_bias = None
        if _edit and any(b != 1.0 for b in ref_boosts):
            attn_bias = _ref_attn_bias(ref_boosts, txtlen, [r.shape[1] for r in refs], imglen, device, combined.dtype)

        for block in self.blocks:
            combined = block(combined, tvec, freqs, attn_bias, transformer_options=transformer_options)

        out = self.last(combined[:, txtlen + reflen : txtlen + reflen + imglen], t)
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]

        return out.unsqueeze(2)
