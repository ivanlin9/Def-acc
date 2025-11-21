#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# Ensure 'src' is on sys.path when invoking this file directly
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from scanner.scanner import pretty_print_report, trace_functional_calls, list_op_locations, list_attention_and_mlp_blocks  # type: ignore


def _load_hf_model(model_name_or_path: str, device: torch.device) -> Optional[nn.Module]:
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except Exception as exc:
        print(
            "Transformers not available. Install with `pip install transformers` "
            "to scan a HuggingFace model.",
            file=sys.stderr,
        )
        print(f"(Import error: {exc})", file=sys.stderr)
        return None
    try:
        model = AutoModel.from_pretrained(model_name_or_path)
        return model.to(device)
    except Exception as exc:
        print(f"AutoModel failed for '{model_name_or_path}': {exc}", file=sys.stderr)
        # Fallback to common task-specific classes (e.g., GPT2LMHeadModel)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            return model.to(device)
        except Exception as exc2:
            print(f"AutoModelForCausalLM failed for '{model_name_or_path}': {exc2}", file=sys.stderr)
            return None


def _build_example_inputs_hf(model_name_or_path: str, device: torch.device) -> Optional[Dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception:
        return None
    encoded = tok("hello world", return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a model for HE pain points.")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="",
        help="HuggingFace model name or local path (e.g., 'gpt2', 'sshleifer/tiny-gpt2').",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Run a single forward pass to count functional calls (F.gelu, softmax, tanh, etc.).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print module locations for LayerNorm, Attention, and MLP blocks.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for trace. 'auto' picks CUDA if available.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    model: Optional[nn.Module] = None
    example_inputs: Optional[Dict[str, Any]] = None
    if args.hf_model:
        model = _load_hf_model(args.hf_model, device=device)
        if model is not None and args.trace:
            example_inputs = _build_example_inputs_hf(args.hf_model, device=device)
    if model is None:
        # Fallback: a tiny Torch-only toy model so the script is always runnable.
        class TinyToy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(8)
                self.act = nn.GELU()
                self.sm = nn.Softmax(dim=-1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.norm(x)
                x = self.act(x)
                return self.sm(x)

        model = TinyToy().to(device)
        if args.trace:
            example_inputs = {"x": torch.randn(2, 8, device=device)}

    print(pretty_print_report(model))

    # Report HF activation function(s) if available
    cfg = getattr(model, "config", None)
    if cfg is not None:
        act_name = getattr(cfg, "activation_function", None)
        hidden_act = getattr(cfg, "hidden_act", None)
        if act_name:
            print(f"\nHF activation_function: {act_name}")
        if hidden_act:
            print(f"HF hidden_act: {hidden_act}")

    if args.verbose:
        locs = list_op_locations(model)
        am = list_attention_and_mlp_blocks(model)
        def _print_list(title: str, items):
            print(f"\n{title} ({len(items)}):")
            for it in items:
                print(f" - {it}")
        _print_list("LayerNorm modules", locs.get("LayerNorm", []))
        _print_list("Attention blocks (heuristic)", am.get("Attention", []))
        _print_list("MLP blocks (heuristic)", am.get("MLP", []))

    if args.trace and example_inputs is not None:
        counts = trace_functional_calls(model, example_inputs)
        print("\nFunctional-call trace (single forward pass):")
        for name, cnt in counts.items():
            if name in ("Softmax", "GELU", "Tanh", "Sigmoid", "SiLU"):
                print(f"{name}\t{cnt}")


if __name__ == "__main__":
    main()


