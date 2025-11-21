from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def poly_gelu3(x: torch.Tensor) -> torch.Tensor:
    """
    Low-degree polynomial surrogate for GELU, HE-friendly (no exp/erf/tanh).
    Form: 0.5 * x * (1 + k1 * x + k3 * x^3)
    Coefficients chosen to roughly approximate 'gelu_new' near zero.
    """
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
def make_he_ready_model(model: nn.Module, config: str = "he_friendly_low") -> nn.Module:
    """
    Return a modified model with HE-friendly approximations swapped in.

    Current behavior:
    - Replace nn.GELU modules with polynomial GELU.
    - Replace module-level '.act' if it's GELU-like with polynomial GELU.
    - Replace module-level '.act_fn' if it's SiLU-like with polynomial SiLU.
    - Replace RMSNorm modules with HE-friendly approximation.

    Future work (not yet implemented here):
    - Approximate LayerNorm rsqrt via polynomial/Newton.
    - Approximate softmax or use kernelized attention.
    """
    if config not in {"he_friendly_low", "baseline"}:
        raise ValueError(f"Unknown config: {config}")
    if config == "baseline":
        return model

    # Replace nn.GELU modules
    _replace_nn_gelu_with_poly(model)
    # Replace module-level gelu activations (e.g., GPT2MLP.act)
    _replace_act_attributes(model, predicate=_is_gelu_like, replacement=poly_gelu3)
    # Replace module-level silu activations (e.g., Qwen MLP.act_fn)
    _replace_act_attributes(model, predicate=_is_silu_like, replacement=poly_silu3)
    # Replace RMSNorm modules
    _replace_rmsnorm_modules(model)
    return model


