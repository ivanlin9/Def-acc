from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CalibStats:
    act_min: float = 0.0
    act_max: float = 0.0
    logits_min: float = 0.0
    logits_max: float = 0.0
    ln_var_min: float = 0.0
    ln_var_max: float = 0.0

    def update_range(self, name: str, t: torch.Tensor) -> None:
        t_min = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item())
        t_max = float(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item())
        if name == "act":
            self.act_min = min(self.act_min, t_min)
            self.act_max = max(self.act_max, t_max)
        elif name == "logits":
            self.logits_min = min(self.logits_min, t_min)
            self.logits_max = max(self.logits_max, t_max)
        elif name == "ln_var":
            self.ln_var_min = min(self.ln_var_min, t_min)
            self.ln_var_max = max(self.ln_var_max, t_max)


def calibrate_model_ranges(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run a few forward passes and collect rough ranges for:
    - pre-activation inputs (act_min/max)
    - attention logits (logits_min/max) via SDPA hook
    - LayerNorm/RMSNorm variance (ln_var_min/max)
    Returns suggested clamps: {'act_clip': A, 'logits_clip': L}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    stats = CalibStats(
        act_min=0.0, act_max=0.0, logits_min=0.0, logits_max=0.0, ln_var_min=0.0, ln_var_max=0.0
    )

    # Monkeypatch common activation functions to capture inputs
    orig_fns: Dict[str, Any] = {}
    # torch.nn.functional.gelu
    orig_fns["F.gelu"] = F.gelu
    def gelu_wrapped(x: torch.Tensor, *args, **kwargs):
        stats.update_range("act", x.detach())
        return orig_fns["F.gelu"](x, *args, **kwargs)
    F.gelu = gelu_wrapped  # type: ignore[assignment]
    # torch.nn.functional.silu
    if hasattr(F, "silu"):
        orig_fns["F.silu"] = F.silu
        def silu_wrapped(x: torch.Tensor, *args, **kwargs):
            stats.update_range("act", x.detach())
            return orig_fns["F.silu"](x, *args, **kwargs)
        F.silu = silu_wrapped  # type: ignore[assignment]
    # HF activations: gelu_new / quick_gelu if available
    try:
        from transformers import activations as hf_act  # type: ignore
        if hasattr(hf_act, "gelu_new"):
            orig_fns["hf.gelu_new"] = hf_act.gelu_new
            def gelu_new_wrapped(x: torch.Tensor, *args, **kwargs):
                stats.update_range("act", x.detach())
                return orig_fns["hf.gelu_new"](x, *args, **kwargs)
            hf_act.gelu_new = gelu_new_wrapped  # type: ignore[assignment]
        if hasattr(hf_act, "quick_gelu"):
            orig_fns["hf.quick_gelu"] = hf_act.quick_gelu
            def quick_gelu_wrapped(x: torch.Tensor, *args, **kwargs):
                stats.update_range("act", x.detach())
                return orig_fns["hf.quick_gelu"](x, *args, **kwargs)
            hf_act.quick_gelu = quick_gelu_wrapped  # type: ignore[assignment]
    except Exception:
        pass

    # Hook for LayerNorm/RMSNorm to capture variance
    ln_hooks: List[torch.utils.hooks.RemovableHandle] = []

    def ln_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...], _output: torch.Tensor):
        if not inputs:
            return
        x = inputs[0].detach()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        stats.update_range("ln_var", var)

    for _, sub in model.named_modules():
        if isinstance(sub, nn.LayerNorm) or sub.__class__.__name__.lower() == "rmsnorm":
            ln_hooks.append(sub.register_forward_pre_hook(ln_hook))

    # Wrap SDPA to estimate logits range
    orig_sdpa = F.scaled_dot_product_attention if hasattr(F, "scaled_dot_product_attention") else None

    def sdpa_wrapped(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        d = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d ** 0.5)
        stats.update_range("logits", scores.detach())
        if orig_sdpa is not None:
            return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        # Fallback: softmax
        return torch.matmul(torch.softmax(scores, dim=-1), value)

    if orig_sdpa is not None:
        F.scaled_dot_product_attention = sdpa_wrapped  # type: ignore[assignment]

    # Forward on small batch of texts
    with torch.inference_mode():
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt").to(device)
            try:
                _ = model(**enc)
            except Exception:
                # Some chat models require specific formats; ignore failures
                continue

    # Restore everything
    # Restore monkeypatched activations
    if "F.gelu" in orig_fns:
        F.gelu = orig_fns["F.gelu"]  # type: ignore[assignment]
    if "F.silu" in orig_fns:
        F.silu = orig_fns["F.silu"]  # type: ignore[assignment]
    try:
        from transformers import activations as hf_act  # type: ignore
        if "hf.gelu_new" in orig_fns:
            hf_act.gelu_new = orig_fns["hf.gelu_new"]  # type: ignore[assignment]
        if "hf.quick_gelu" in orig_fns:
            hf_act.quick_gelu = orig_fns["hf.quick_gelu"]  # type: ignore[assignment]
    except Exception:
        pass
    for h in ln_hooks:
        h.remove()
    if orig_sdpa is not None:
        F.scaled_dot_product_attention = orig_sdpa  # type: ignore[assignment]

    # Suggest clamps using symmetric abs max with caps
    act_clip = max(abs(stats.act_min), abs(stats.act_max))
    logits_clip = max(abs(stats.logits_min), abs(stats.logits_max))
    # Reasonable caps to avoid extreme values driving instability
    act_clip = float(min(max(act_clip, 2.0), 6.0))
    logits_clip = float(min(max(logits_clip, 2.0), 8.0))
    return {"act_clip": act_clip, "logits_clip": logits_clip}


