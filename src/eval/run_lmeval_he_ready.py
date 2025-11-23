#!/usr/bin/env python3
import os
import json
from datetime import datetime

from lm_eval import evaluator

# Monkeypatch HF loading to apply HE-ready norms-only wrapper
from transformers import AutoModel, AutoModelForCausalLM  # type: ignore
from wrapper import make_he_ready_model

_orig_auto_model_from_pretrained_func = AutoModel.from_pretrained.__func__  # type: ignore[attr-defined]
_orig_auto_model_for_causallm_from_pretrained_func = AutoModelForCausalLM.from_pretrained.__func__  # type: ignore[attr-defined]


def _apply_he_ready_norms_only(model):
    if getattr(model, "_he_ready_applied", False):
        return model
    # Heuristic: Qwen-family uses RMSNorm heavily; our coarse RMSNorm poly hurts.
    # For RMSNorm/Qwen models, apply activations-only. Otherwise, apply higher-fidelity LayerNorm (2-step rsqrt).
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", "") if cfg is not None else ""
    is_qwen = isinstance(model_type, str) and ("qwen" in model_type.lower())
    has_rms = is_qwen or any("rmsnorm" in m.__class__.__name__.lower() for m in model.modules())
    if has_rms:
        model = make_he_ready_model(
            model,
            config="he_acts_only",
            enable_activations=True,
            enable_norms=False,
            enable_attention=False,
        )
        setattr(model, "_he_ready_policy", "acts_only")
    else:
        model = make_he_ready_model(
            model,
            config="he_friendly_high",  # yields rsqrt_steps=2
            enable_activations=False,
            enable_norms=True,
            enable_attention=False,  # keep exact attention for stability
        )
        setattr(model, "_he_ready_policy", "norms_rsqrt2")
    setattr(model, "_he_ready_applied", True)
    return model


def _patched_from_pretrained(cls, *args, **kwargs):
    model = _orig_auto_model_from_pretrained_func(cls, *args, **kwargs)  # type: ignore[misc]
    return _apply_he_ready_norms_only(model)


def _patched_causallm_from_pretrained(cls, *args, **kwargs):
    # Drop lm-eval specific kwargs that shouldn't go to HF from_pretrained
    kwargs = dict(kwargs)
    kwargs.pop("apply_chat_template", None)
    kwargs.pop("fewshot_as_multiturn", None)
    model = _orig_auto_model_for_causallm_from_pretrained_func(cls, *args, **kwargs)  # type: ignore[misc]
    return _apply_he_ready_norms_only(model)


AutoModel.from_pretrained = classmethod(_patched_from_pretrained)  # type: ignore[assignment]
AutoModelForCausalLM.from_pretrained = classmethod(_patched_causallm_from_pretrained)  # type: ignore[assignment]


def main():
    models_to_test = [
        {"name": "GPT-2", "model_args": "pretrained=gpt2"},
        {
            "name": "Qwen2.5-Coder-3B-Instruct",
            "model_args": "pretrained=Qwen/Qwen2.5-Coder-3B-Instruct,trust_remote_code=True",
        },
    ]

    tasks_list = ["hellaswag", "lambada_openai"]

    print(f"Starting evaluation for: {[m['name'] for m in models_to_test]}")

    all_results = []
    timestamp = datetime.now().isoformat()

    for model_info in models_to_test:
        print("\n" + "=" * 40)
        print(f"Evaluating: {model_info['name']}")
        print("=" * 40)
        try:
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=model_info["model_args"],
                tasks=tasks_list,
            )
            print(f"\nResults for {model_info['name']}:")
            model_result = {"model": model_info["name"], "timestamp": timestamp, "metrics": {}}

            for task in tasks_list:
                task_res = results["results"].get(task, {})
                acc = task_res.get("acc,none", task_res.get("acc", "N/A"))
                print(f"  - {task}: {acc}")
                model_result["metrics"][task] = acc

            all_results.append(model_result)
        except Exception as e:
            print(f"Error evaluating {model_info['name']}: {e}")

    output_file = "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()


