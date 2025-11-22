#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Ensure 'src' on sys.path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from wrapper import make_he_ready_model, enable_attention_approximation


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        model = AutoModel.from_pretrained(model_name)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    return model.to(device)


def build_inputs(model_name: str, device: torch.device) -> dict:
    tok = AutoTokenizer.from_pretrained(model_name)
    text = "hello world"
    encoded = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in encoded.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply HE-ready wrapper to a HF model and run a tiny forward pass.")
    parser.add_argument("--hf_model", type=str, required=True)
    parser.add_argument("--config", type=str, default="he_friendly_low", choices=["baseline", "he_friendly_low", "he_friendly_high"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--approx_attention", action="store_true", help="Enable HE-style SDPA approximation")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    model = load_model(args.hf_model, device=device)
    model = make_he_ready_model(model, config=args.config)
    if args.approx_attention:
        enable_attention_approximation()
    inputs = build_inputs(args.hf_model, device=device)

    model.eval()
    with torch.inference_mode():
        _ = model(**inputs)
    print("Forward pass OK with HE-ready wrapper.")


if __name__ == "__main__":
    main()


