from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OpInfo:
    count: int
    note: str


HE_PAIN_NOTES: Dict[str, str] = {
    "Softmax": "Requires exp + sum + division; expensive under HE.",
    "LayerNorm": "Requires mean, variance, and 1/sqrt; expensive under HE.",
    "RMSNorm": "Computes 1/sqrt(mean(x^2)); approximate rsqrt for HE.",
    "GELU": "Standard GELU uses erf/tanh; replace with low-degree polynomial.",
    "SiLU": "Uses division; better replaced with polynomial approximation.",
    "Tanh": "Non-polynomial; approximate with polynomial for HE.",
    "Sigmoid": "Non-polynomial; approximate with polynomial for HE.",
}

MODULE_CLASSES: Dict[str, Tuple[type, ...]] = {
    "Softmax": (nn.Softmax,),
    "LayerNorm": (nn.LayerNorm,),
    "GELU": (nn.GELU,),
    "SiLU": (nn.SiLU,) if hasattr(nn, "SiLU") else tuple(),
    "Tanh": (nn.Tanh,),
    "Sigmoid": (nn.Sigmoid,),
}


def _iter_modules(module: nn.Module) -> Iterable[nn.Module]:
    yield module
    for submodule in module.children():
        yield from _iter_modules(submodule)


def scan_model(model: nn.Module) -> Dict[str, int]:
    """
    Recursively scan a PyTorch model to count HE-unfriendly operations by module type.
    This does not detect functional calls (e.g., torch.nn.functional.gelu) that are not
    represented as nn.Module instances.
    """
    op_counts: Dict[str, int] = {name: 0 for name in HE_PAIN_NOTES.keys()}
    for submodule in _iter_modules(model):
        # Count by isinstance for known torch.nn modules
        for op_name, cls_tuple in MODULE_CLASSES.items():
            if not cls_tuple:
                continue
            if isinstance(submodule, cls_tuple):
                op_counts[op_name] += 1
        # Heuristic: detect RMSNorm by class name to support HF modules without explicit import
        cls_name = submodule.__class__.__name__
        if cls_name.lower() == "rmsnorm":
            op_counts["RMSNorm"] += 1
    return op_counts


def list_op_locations(model: nn.Module) -> Dict[str, List[str]]:
    """
    Return qualified module names where each HE-unfriendly module class appears.
    """
    locations: Dict[str, List[str]] = {name: [] for name in HE_PAIN_NOTES.keys()}
    for name, m in model.named_modules():
        for op_name, cls_tuple in MODULE_CLASSES.items():
            if not cls_tuple:
                continue
            if isinstance(m, cls_tuple):
                locations[op_name].append(name)
        if m.__class__.__name__.lower() == "rmsnorm":
            locations["RMSNorm"].append(name)
    return locations


def list_attention_and_mlp_blocks(model: nn.Module) -> Dict[str, List[str]]:
    """
    Heuristically list attention and MLP blocks by qualified names.
    Useful for identifying where softmax-like attention and GELU-like MLPs live.
    """
    attn: List[str] = []
    mlp: List[str] = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__.lower()
        # Attention modules: class name contains 'attention' or 'attn'
        if "attention" in cls or cls.endswith("attn") or ".attn" in name:
            attn.append(name)
        # MLP blocks: class name contains 'mlp' or attribute path ends with '.mlp'
        if "mlp" in cls or name.endswith(".mlp") or ".mlp" in name:
            mlp.append(name)
    return {"Attention": attn, "MLP": mlp}


def _estimate_multiplicative_depth(op_counts: Dict[str, int]) -> int:
    """
    Provide a coarse heuristic score for multiplicative depth pressure.
    This is NOT an HE-accurate depth computation; it is a relative indicator.
    """
    weights = {
        "Softmax": 3,  # exp + sum + reciprocal
        "LayerNorm": 3,  # mean + var + rsqrt
        "GELU": 2,  # erf/tanh or polynomial replacement
        "SiLU": 2,  # division or polynomial
        "Tanh": 2,  # polynomial replacement
        "Sigmoid": 2,  # polynomial replacement
    }
    score = 0
    for name, count in op_counts.items():
        score += weights.get(name, 1) * count
    return score


def build_he_pain_report(model: nn.Module) -> Dict[str, OpInfo]:
    """
    Build a structured report mapping operation name -> OpInfo(count, note).
    """
    counts = scan_model(model)
    return {name: OpInfo(count=counts[name], note=HE_PAIN_NOTES[name]) for name in counts}


def pretty_print_report(model: nn.Module) -> str:
    """
    Return a human-readable table string summarizing HE pain points and a heuristic depth score.
    """
    report = build_he_pain_report(model)
    headers = ["Op", "Count", "Notes (HE pain)"]
    lines = ["\t".join(headers)]
    for name, info in report.items():
        lines.append(f"{name}\t{info.count}\t{info.note}")
    depth_score = _estimate_multiplicative_depth({k: v.count for k, v in report.items()})
    lines.append("")
    lines.append(f"Heuristic multiplicative-depth score: {depth_score}")
    return "\n".join(lines)


# --- Optional functional-call tracing (for models using F.* or tensor methods) ---
_ORIGINAL_FUNCS: Dict[str, Any] = {}
_FUNC_COUNTS: Dict[str, int] = {}


def _inc(key: str) -> None:
    _FUNC_COUNTS[key] = _FUNC_COUNTS.get(key, 0) + 1


def enable_functional_counters() -> None:
    """
    Monkeypatch selected torch/torch.nn.functional ops to count calls during a forward pass.
    """
    global _ORIGINAL_FUNCS, _FUNC_COUNTS
    if _ORIGINAL_FUNCS:
        return
    _FUNC_COUNTS = {}
    # GELU
    _ORIGINAL_FUNCS["F.gelu"] = F.gelu

    def gelu_wrapped(*args, **kwargs):
        _inc("GELU")
        return _ORIGINAL_FUNCS["F.gelu"](*args, **kwargs)

    F.gelu = gelu_wrapped  # type: ignore[assignment]

    # HuggingFace activations: gelu_new / quick_gelu
    try:
        from transformers import activations as hf_act  # type: ignore
        if hasattr(hf_act, "gelu_new"):
            _ORIGINAL_FUNCS["hf.gelu_new"] = hf_act.gelu_new

            def gelu_new_wrapped(*args, **kwargs):
                _inc("GELU")
                return _ORIGINAL_FUNCS["hf.gelu_new"](*args, **kwargs)

            hf_act.gelu_new = gelu_new_wrapped  # type: ignore[assignment]
        if hasattr(hf_act, "quick_gelu"):
            _ORIGINAL_FUNCS["hf.quick_gelu"] = hf_act.quick_gelu

            def quick_gelu_wrapped(*args, **kwargs):
                _inc("GELU")
                return _ORIGINAL_FUNCS["hf.quick_gelu"](*args, **kwargs)

            hf_act.quick_gelu = quick_gelu_wrapped  # type: ignore[assignment]
        # Also patch ACT2FN mapping to catch pre-bound references
        if hasattr(hf_act, "ACT2FN"):
            act2fn = hf_act.ACT2FN  # type: ignore[attr-defined]
            if "gelu_new" in act2fn:
                _ORIGINAL_FUNCS["hf.ACT2FN.gelu_new"] = act2fn["gelu_new"]

                def gelu_new_wrapped_map(*args, **kwargs):
                    _inc("GELU")
                    return _ORIGINAL_FUNCS["hf.ACT2FN.gelu_new"](*args, **kwargs)

                act2fn["gelu_new"] = gelu_new_wrapped_map  # type: ignore[index]
            if "quick_gelu" in act2fn:
                _ORIGINAL_FUNCS["hf.ACT2FN.quick_gelu"] = act2fn["quick_gelu"]

                def quick_gelu_wrapped_map(*args, **kwargs):
                    _inc("GELU")
                    return _ORIGINAL_FUNCS["hf.ACT2FN.quick_gelu"](*args, **kwargs)

                act2fn["quick_gelu"] = quick_gelu_wrapped_map  # type: ignore[index]
    except Exception:
        pass

    # Softmax
    _ORIGINAL_FUNCS["F.softmax"] = F.softmax
    _ORIGINAL_FUNCS["torch.softmax"] = torch.softmax
    _ORIGINAL_FUNCS["Tensor.softmax"] = torch.Tensor.softmax
    # Scaled Dot-Product Attention (fused softmax path)
    if hasattr(F, "scaled_dot_product_attention"):
        _ORIGINAL_FUNCS["F.sdpa"] = F.scaled_dot_product_attention

    def softmax_wrapped_F(*args, **kwargs):
        _inc("Softmax")
        return _ORIGINAL_FUNCS["F.softmax"](*args, **kwargs)

    def softmax_wrapped_torch(*args, **kwargs):
        _inc("Softmax")
        return _ORIGINAL_FUNCS["torch.softmax"](*args, **kwargs)

    def softmax_wrapped_tensor(self, *args, **kwargs):
        _inc("Softmax")
        return _ORIGINAL_FUNCS["Tensor.softmax"](self, *args, **kwargs)

    F.softmax = softmax_wrapped_F  # type: ignore[assignment]
    torch.softmax = softmax_wrapped_torch  # type: ignore[assignment]
    torch.Tensor.softmax = softmax_wrapped_tensor  # type: ignore[assignment]
    if "F.sdpa" in _ORIGINAL_FUNCS:
        def sdpa_wrapped(*args, **kwargs):
            # SDPA includes masking + softmax internally
            _inc("Softmax")
            return _ORIGINAL_FUNCS["F.sdpa"](*args, **kwargs)
        F.scaled_dot_product_attention = sdpa_wrapped  # type: ignore[assignment]

    # Tanh
    _ORIGINAL_FUNCS["F.tanh"] = F.tanh if hasattr(F, "tanh") else None
    _ORIGINAL_FUNCS["torch.tanh"] = torch.tanh

    def tanh_wrapped_torch(*args, **kwargs):
        _inc("Tanh")
        return _ORIGINAL_FUNCS["torch.tanh"](*args, **kwargs)

    torch.tanh = tanh_wrapped_torch  # type: ignore[assignment]
    if _ORIGINAL_FUNCS["F.tanh"] is not None:
        def tanh_wrapped_F(*args, **kwargs):  # type: ignore[no-redef]
            _inc("Tanh")
            return _ORIGINAL_FUNCS["F.tanh"](*args, **kwargs)
        F.tanh = tanh_wrapped_F  # type: ignore[assignment]

    # Sigmoid
    _ORIGINAL_FUNCS["F.sigmoid"] = F.sigmoid
    _ORIGINAL_FUNCS["torch.sigmoid"] = torch.sigmoid

    def sigmoid_wrapped_F(*args, **kwargs):
        _inc("Sigmoid")
        return _ORIGINAL_FUNCS["F.sigmoid"](*args, **kwargs)

    def sigmoid_wrapped_torch(*args, **kwargs):
        _inc("Sigmoid")
        return _ORIGINAL_FUNCS["torch.sigmoid"](*args, **kwargs)

    F.sigmoid = sigmoid_wrapped_F  # type: ignore[assignment]
    torch.sigmoid = sigmoid_wrapped_torch  # type: ignore[assignment]
    # SiLU
    if hasattr(F, "silu"):
        _ORIGINAL_FUNCS["F.silu"] = F.silu
        def silu_wrapped_F(*args, **kwargs):
            _inc("SiLU")
            return _ORIGINAL_FUNCS["F.silu"](*args, **kwargs)
        F.silu = silu_wrapped_F  # type: ignore[assignment]
    if hasattr(torch, "silu"):
        _ORIGINAL_FUNCS["torch.silu"] = torch.silu
        def silu_wrapped_torch(*args, **kwargs):
            _inc("SiLU")
            return _ORIGINAL_FUNCS["torch.silu"](*args, **kwargs)
        torch.silu = silu_wrapped_torch  # type: ignore[assignment]


def disable_functional_counters() -> None:
    global _ORIGINAL_FUNCS
    if not _ORIGINAL_FUNCS:
        return
    # Restore originals
    if "F.gelu" in _ORIGINAL_FUNCS:
        F.gelu = _ORIGINAL_FUNCS["F.gelu"]  # type: ignore[assignment]
    # Restore HF activations if patched
    try:
        from transformers import activations as hf_act  # type: ignore
        if "hf.gelu_new" in _ORIGINAL_FUNCS:
            hf_act.gelu_new = _ORIGINAL_FUNCS["hf.gelu_new"]  # type: ignore[assignment]
        if "hf.quick_gelu" in _ORIGINAL_FUNCS:
            hf_act.quick_gelu = _ORIGINAL_FUNCS["hf.quick_gelu"]  # type: ignore[assignment]
        if hasattr(hf_act, "ACT2FN"):
            act2fn = hf_act.ACT2FN  # type: ignore[attr-defined]
            if "hf.ACT2FN.gelu_new" in _ORIGINAL_FUNCS:
                act2fn["gelu_new"] = _ORIGINAL_FUNCS["hf.ACT2FN.gelu_new"]  # type: ignore[index]
            if "hf.ACT2FN.quick_gelu" in _ORIGINAL_FUNCS:
                act2fn["quick_gelu"] = _ORIGINAL_FUNCS["hf.ACT2FN.quick_gelu"]  # type: ignore[index]
    except Exception:
        pass
    if "F.softmax" in _ORIGINAL_FUNCS:
        F.softmax = _ORIGINAL_FUNCS["F.softmax"]  # type: ignore[assignment]
    if "torch.softmax" in _ORIGINAL_FUNCS:
        torch.softmax = _ORIGINAL_FUNCS["torch.softmax"]  # type: ignore[assignment]
    if "Tensor.softmax" in _ORIGINAL_FUNCS:
        torch.Tensor.softmax = _ORIGINAL_FUNCS["Tensor.softmax"]  # type: ignore[assignment]
    if "F.sdpa" in _ORIGINAL_FUNCS:
        F.scaled_dot_product_attention = _ORIGINAL_FUNCS["F.sdpa"]  # type: ignore[assignment]
    if "F.tanh" in _ORIGINAL_FUNCS and _ORIGINAL_FUNCS["F.tanh"] is not None:
        F.tanh = _ORIGINAL_FUNCS["F.tanh"]  # type: ignore[assignment]
    if "torch.tanh" in _ORIGINAL_FUNCS:
        torch.tanh = _ORIGINAL_FUNCS["torch.tanh"]  # type: ignore[assignment]
    if "F.sigmoid" in _ORIGINAL_FUNCS:
        F.sigmoid = _ORIGINAL_FUNCS["F.sigmoid"]  # type: ignore[assignment]
    if "torch.sigmoid" in _ORIGINAL_FUNCS:
        torch.sigmoid = _ORIGINAL_FUNCS["torch.sigmoid"]  # type: ignore[assignment]
    if "F.silu" in _ORIGINAL_FUNCS:
        F.silu = _ORIGINAL_FUNCS["F.silu"]  # type: ignore[assignment]
    if "torch.silu" in _ORIGINAL_FUNCS:
        torch.silu = _ORIGINAL_FUNCS["torch.silu"]  # type: ignore[assignment]
    _ORIGINAL_FUNCS = {}


def trace_functional_calls(model: nn.Module, example_inputs: Dict[str, Any]) -> Dict[str, int]:
    """
    Run a single forward pass to count common functional calls (GELU, Softmax, Tanh, Sigmoid).
    Returns a dict of counts. The model is not modified after the call.
    """
    model_was_training = model.training
    model = model.eval()
    enable_functional_counters()
    # Wrap per-module activation functions (e.g., GPT2MLP.act = gelu_new) to capture GELU calls
    wrapped_acts: List[Tuple[nn.Module, str, Any]] = []
    def make_act_wrapper(original_fn: Any):
        def _wrapped(*args, **kwargs):
            _inc("GELU")
            return original_fn(*args, **kwargs)
        return _wrapped
    for _, sub in model.named_modules():
        if hasattr(sub, "act"):
            act_fn = getattr(sub, "act")
            if callable(act_fn):
                # Heuristic: only wrap GELU-like functions
                fn_name = getattr(act_fn, "__name__", "")
                if "gelu" in fn_name:
                    wrapped_acts.append((sub, "act", act_fn))
                    setattr(sub, "act", make_act_wrapper(act_fn))
    try:
        with torch.inference_mode():
            _ = model(**example_inputs)
        # Merge with zeros to ensure all keys exist
        counts = {name: _FUNC_COUNTS.get(name, 0) for name in HE_PAIN_NOTES.keys()}
        # If model config indicates gelu, map tanh-inside-gelu to GELU if needed
        act_name = getattr(getattr(model, "config", object()), "activation_function", "")
        if isinstance(act_name, str) and "gelu" in act_name and counts.get("GELU", 0) == 0 and counts.get("Tanh", 0) > 0:
            counts["GELU"] = counts.get("GELU", 0) + counts["Tanh"]
            counts["Tanh"] = 0
        return counts
    finally:
        # Restore wrapped acts
        for module, attr, original in wrapped_acts:
            setattr(module, attr, original)
        disable_functional_counters()
        if model_was_training:
            model.train()


