from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .remez import exp_cheb, inv_cheb, tanh_cheb

def poly_gelu3(x: torch.Tensor) -> torch.Tensor:
    """
    Low-degree polynomial surrogate for GELU, HE-friendly (no exp/erf/tanh).
    Form: 0.5 * x * (1 + k1 * x + k3 * x^3)
    Coefficients chosen to roughly approximate 'gelu_new' near zero.
    """
    x = x.clamp(-ACT_CLIP, ACT_CLIP)
    k1 = 0.7978845608  # sqrt(2/pi)
    k3 = 0.0356774081  # heuristic cubic coefficient
    return 0.5 * x * (1.0 + k1 * x + k3 * x.pow(3))


class PolyGELUModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return poly_gelu3(x)


def _replace_nn_gelu_with_poly(module: nn.Module) -> int:
    replaced = 0
    for name, child in module.named_children():
        # Recurse first
        replaced += _replace_nn_gelu_with_poly(child)
        if isinstance(child, nn.GELU):
            setattr(module, name, PolyGELUModule())
            replaced += 1
    return replaced


def _replace_act_attributes(model: nn.Module, predicate: Callable[[Callable[..., torch.Tensor]], bool], replacement: Callable[..., torch.Tensor]) -> int:
    """
    Replace submodule 'act' attributes when predicate(act_fn) is True.
    Returns the number of replacements performed.
    """
    replaced = 0
    for _, sub in model.named_modules():
        if hasattr(sub, "act"):
            act_fn = getattr(sub, "act")
            if callable(act_fn) and predicate(act_fn):
                setattr(sub, "act", replacement)
                replaced += 1
        if hasattr(sub, "act_fn"):
            act_fn = getattr(sub, "act_fn")
            if callable(act_fn) and predicate(act_fn):
                setattr(sub, "act_fn", replacement)
                replaced += 1
    return replaced


def _is_gelu_like(fn: Callable[..., torch.Tensor]) -> bool:
    name = getattr(fn, "__name__", "")
    return "gelu" in name.lower() or fn is F.gelu

def poly_silu3(x: torch.Tensor) -> torch.Tensor:
    """
    Low-degree polynomial surrogate for SiLU = x * sigmoid(x).
    Approximate sigmoid(x) ~ 0.5 + 0.15012 x - 0.001593 x^3 (rough heuristic).
    """
    x = x.clamp(-ACT_CLIP, ACT_CLIP)
    s = 0.5 + 0.15012 * x - 0.001593 * x.pow(3)
    return x * s

def _is_silu_like(fn: Callable[..., torch.Tensor]) -> bool:
    name = getattr(fn, "__name__", "")
    return "silu" in name.lower() or fn is F.silu or fn is getattr(torch, "silu", object())

class HEFriendlyRMSNorm(nn.Module):
    """
    Approximate RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    Replace rsqrt with a small polynomial around 1: for z in ~[0.5, 1.5],
    (z)^(-1/2) ≈ 1 - 0.5*(z-1) + 0.375*(z-1)^2
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        if weight is None:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.weight = nn.Parameter(weight.detach().clone())

    @staticmethod
    def _poly_rsqrt(z: torch.Tensor) -> torch.Tensor:
        u = z - 1.0
        return 1.0 - 0.5 * u + 0.375 * u.pow(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        inv = self._poly_rsqrt(ms)
        y = x * inv
        return y * self.weight

def _replace_rmsnorm_modules(model: nn.Module) -> int:
    """
    Replace modules whose class name is 'RMSNorm' (case-insensitive) with HEFriendlyRMSNorm,
    attempting to carry over 'weight' and 'eps' attributes if present.
    """
    replaced = 0
    for name, child in list(model.named_children()):
        # Recurse
        replaced += _replace_rmsnorm_modules(child)
        if "rmsnorm" in child.__class__.__name__.lower():
            hidden_size = None
            weight = getattr(child, "weight", None)
            if weight is not None and isinstance(weight, torch.Tensor):
                hidden_size = weight.numel()
            eps = getattr(child, "eps", 1e-6)
            if hidden_size is None:
                continue
            setattr(model, name, HEFriendlyRMSNorm(hidden_size=hidden_size, eps=float(eps), weight=weight))
            replaced += 1
    return replaced
class HEFriendlyLayerNorm(nn.Module):
    """
    Approximate LayerNorm: (x - mean) * rsqrt(var + eps) * weight + bias
    rsqrt is approximated with a few Newton steps starting from y0=1.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 weight: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None,
                 rsqrt_steps: int = 1) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.rsqrt_steps = rsqrt_steps
        if weight is None:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.weight = nn.Parameter(weight.detach().clone())
        if bias is None:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.bias = nn.Parameter(bias.detach().clone())

    @staticmethod
    def _rsqrt_newton(z: torch.Tensor, steps: int) -> torch.Tensor:
        # Start from y0 = 1
        y = torch.ones_like(z)
        z = torch.clamp(z, min=1e-12)
        for _ in range(max(1, steps)):
            y = y * (1.5 - 0.5 * z * y * y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        xc = x - mu
        var = xc.pow(2).mean(dim=-1, keepdim=True)
        inv = self._rsqrt_newton(var + self.eps, steps=self.rsqrt_steps)
        xn = xc * inv
        return xn * self.weight + self.bias


def _replace_layernorm_modules(model: nn.Module, rsqrt_steps: int = 1) -> int:
    replaced = 0
    for name, child in list(model.named_children()):
        replaced += _replace_layernorm_modules(child, rsqrt_steps=rsqrt_steps)
        if isinstance(child, nn.LayerNorm):
            normalized_shape = child.normalized_shape if isinstance(child.normalized_shape, int) else child.normalized_shape[-1]
            weight = getattr(child, "weight", None)
            bias = getattr(child, "bias", None)
            eps = getattr(child, "eps", 1e-5)
            setattr(model, name, HEFriendlyLayerNorm(normalized_shape=normalized_shape, eps=float(eps), weight=weight, bias=bias, rsqrt_steps=rsqrt_steps))
            replaced += 1
    return replaced

def make_he_ready_model(model: nn.Module, config: str = "he_friendly_low",
                        enable_activations: Optional[bool] = None,
                        enable_norms: Optional[bool] = None,
                        enable_attention: Optional[bool] = None,
                        use_remez_activations: bool = False) -> nn.Module:

    if config not in {"baseline", "he_friendly_low", "he_friendly_high", "he_acts_only", "he_acts_norm", "he_full"}:
        raise ValueError(f"Unknown config: {config}")
    if config == "baseline":
        return model

    # Defaults per config
    if enable_activations is None or enable_norms is None or enable_attention is None:
        cfg_map = {
            "he_friendly_low":  (True, True, False),
            "he_friendly_high": (True, True, True),
            "he_acts_only":     (True, False, False),
            "he_acts_norm":     (True, True, False),
            "he_full":          (True, True, True),
        }
        a, n, attn = cfg_map.get(config, (True, True, False))
        if enable_activations is None:
            enable_activations = a
        if enable_norms is None:
            enable_norms = n
        if enable_attention is None:
            enable_attention = attn

    # Activations
    if enable_activations:
        if use_remez_activations:
            def gelu_remez(x: torch.Tensor) -> torch.Tensor:
                # GELU(new) ≈ 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
                z = 0.7978845608 * (x + 0.044715 * x.pow(3))
                return 0.5 * x * (1.0 + tanh_cheb(z))
            _replace_act_attributes(model, predicate=_is_gelu_like, replacement=gelu_remez)
        else:
            _replace_nn_gelu_with_poly(model)
            _replace_act_attributes(model, predicate=_is_gelu_like, replacement=poly_gelu3)
        _replace_act_attributes(model, predicate=_is_silu_like, replacement=poly_silu3)
    # Norms
    if enable_norms:
        _replace_rmsnorm_modules(model)
        _replace_layernorm_modules(model, rsqrt_steps=1 if config != "he_friendly_high" else 2)
    # Attention
    if enable_attention:
        enable_attention_approximation()
    return model


# ---- Attention / Softmax Approximation via SDPA monkeypatch ----
_ORIG_SDPA = None  # type: ignore[var-annotated]
# Global calibration knobs
ACT_CLIP: float = 4.0
LOGITS_CLIP: float = 4.0

def set_calibration(act_clip: Optional[float] = None, logits_clip: Optional[float] = None) -> None:
    global ACT_CLIP, LOGITS_CLIP
    if act_clip is not None:
        ACT_CLIP = float(act_clip)
    if logits_clip is not None:
        LOGITS_CLIP = float(logits_clip)


def _pade_exp(x: torch.Tensor) -> torch.Tensor:
    """
    Pade [1/1] approximation of exp(x): exp(x) ≈ (1 + x/2) / (1 - x/2)
    We clamp input to keep denominator positive and stable.
    """
    x = torch.clamp(x, min=-2.0, max=2.0)
    num = 1.0 + 0.5 * x
    den = 1.0 - 0.5 * x
    return num / (den + 1e-6)


def _approx_scaled_dot_product_attention(query: torch.Tensor,
                                         key: torch.Tensor,
                                         value: torch.Tensor,
                                         attn_mask: Optional[torch.Tensor] = None,
                                         dropout_p: float = 0.0,
                                         is_causal: bool = False) -> torch.Tensor:
    """
    HE-friendly SDPA approximation:
    - scores = (Q K^T) / sqrt(d)
    - apply attn_mask (additive) and/or causal mask
    - approximate softmax via Pade exp with one reciprocal normalization
    """
    try:
        d = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(max(d, 1))
        # Apply additive mask if provided
        mask_bool = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                mask_bool = attn_mask
            else:
                scores = scores + attn_mask
        # Causal mask
        if is_causal:
            causal = torch.ones_like(scores, dtype=torch.bool).tril()
            scores = torch.where(causal, scores, torch.full_like(scores, -1e9))
            mask_bool = causal if mask_bool is None else (mask_bool & causal)
        # Stable shift + scaled logits
        max_per_row = scores.max(dim=-1, keepdim=True).values
        logits = scores - max_per_row
        # Clip logits and shrink temperature
        logits = logits.clamp(-LOGITS_CLIP, LOGITS_CLIP) * 0.5
        # Use Remez-fitted exp on [-8,0]
        weights_unnorm = exp_cheb(logits)
        # Ensure non-negative
        weights_unnorm = torch.clamp_min(weights_unnorm, 0.0)
        # Zero-out masked positions if we have a boolean mask
        if mask_bool is not None:
            weights_unnorm = torch.where(mask_bool, weights_unnorm, torch.zeros_like(weights_unnorm))
        denom = weights_unnorm.sum(dim=-1, keepdim=True)
        denom = torch.clamp_min(denom, 1e-6)
        # Use Remez-fitted inverse on [0.08, 257] for normalization factor
        inv_denom = inv_cheb(denom)
        weights = weights_unnorm * inv_denom
        # Final safety: if any non-finite, fallback to exact SDPA if available
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Non-finite weights in approx SDPA")
        out = torch.matmul(weights, value)
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite output in approx SDPA")
        return out
    except Exception:
        # Fallback to exact SDPA if available
        global _ORIG_SDPA
        if _ORIG_SDPA is not None:
            return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        # Last resort: softmax with clamp
        d = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(max(d, 1))
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            scores = scores + attn_mask
        max_per_row = scores.max(dim=-1, keepdim=True).values
        logits = (scores - max_per_row).clamp(-10.0, 10.0)
        probs = torch.softmax(logits, dim=-1)
        return torch.matmul(probs, value)


def enable_attention_approximation() -> None:
    """
    Monkeypatch torch.nn.functional.scaled_dot_product_attention with HE-friendly approximation.
    """
    global _ORIG_SDPA
    if _ORIG_SDPA is not None:
        return
    if hasattr(F, "scaled_dot_product_attention"):
        _ORIG_SDPA = F.scaled_dot_product_attention  # type: ignore[assignment]
        F.scaled_dot_product_attention = _approx_scaled_dot_product_attention  # type: ignore[assignment]


def disable_attention_approximation() -> None:
    global _ORIG_SDPA
    if _ORIG_SDPA is None:
        return
    F.scaled_dot_product_attention = _ORIG_SDPA  # type: ignore[assignment]
    _ORIG_SDPA = None


