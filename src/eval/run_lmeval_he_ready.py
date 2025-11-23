#!/usr/bin/env python3
import os
import sys
import json
import argparse
import pickle
from datetime import datetime
from pathlib import Path

# Add src to path
SRC_ROOT = Path(__file__).parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lm_eval import evaluator

# Monkeypatch HF loading to apply HE-ready norms-only wrapper
from transformers import AutoModel, AutoModelForCausalLM  # type: ignore
from wrapper import make_he_ready_model
from wrapper.encrypt import EncryptedModel

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


def load_and_decrypt_encrypted_model(repo_id: str, secret_key_path: str, config: str = "he_friendly_high"):
    """
    Load an encrypted model from HuggingFace Hub and decrypt it.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "IvanLin/gpt2-encrypted")
        secret_key_path: Path to secret key file
        config: HE config used when encrypting
    
    Returns:
        Decrypted model ready for evaluation
    """
    print(f"\n{'='*60}")
    print(f"Loading Encrypted Model: {repo_id}")
    print(f"{'='*60}")
    
    # Step 1: Load model architecture from Hub
    print("\n1. Loading model architecture from HuggingFace Hub...")
    model = AutoModel.from_pretrained(repo_id)
    print(f"   ✓ Model architecture loaded")
    
    # Step 2: Make HE-friendly (same config as when encrypted)
    print(f"\n2. Applying HE-friendly transformations (config: {config})...")
    he_model = make_he_ready_model(model, config=config)
    print(f"   ✓ HE-friendly transformations applied")
    
    # Step 3: Download encrypted model files
    print("\n3. Downloading encrypted weights from Hub...")
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(repo_id=repo_id, cache_dir=None)
    encrypted_model_dir = os.path.join(cache_dir, "encrypted_model")
    
    if not os.path.exists(encrypted_model_dir):
        raise FileNotFoundError(
            f"Encrypted model directory not found in {repo_id}. "
            f"Expected 'encrypted_model/' folder in the repository."
        )
    print(f"   ✓ Encrypted weights downloaded to {encrypted_model_dir}")
    
    # Step 4: Load encrypted model
    print("\n4. Loading encrypted model...")
    encrypted_model = EncryptedModel.load(encrypted_model_dir, he_model)
    print(f"   ✓ Encrypted model loaded")
    
    # Step 5: Load secret key
    print(f"\n5. Loading secret key from {secret_key_path}...")
    if not os.path.exists(secret_key_path):
        raise FileNotFoundError(f"Secret key not found at {secret_key_path}")
    
    with open(secret_key_path, 'rb') as f:
        secret_key_serialized = pickle.load(f)
    print(f"   ✓ Secret key loaded")
    
    # Step 6: Decrypt weights
    print("\n6. Decrypting model weights (this may take a while)...")
    decrypted_weights = encrypted_model.decrypt_weights(secret_key_serialized=secret_key_serialized)
    print(f"   ✓ Weights decrypted")
    
    # Step 7: Load decrypted weights into model
    print("\n7. Loading decrypted weights into model...")
    he_model.load_state_dict(decrypted_weights)
    print(f"   ✓ Model ready for evaluation")
    
    return he_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate HE-ready or encrypted models")
    parser.add_argument(
        "--encrypted_repo",
        type=str,
        default=None,
        help="HuggingFace repo ID of encrypted model (e.g., 'IvanLin/gpt2-encrypted')"
    )
    parser.add_argument(
        "--secret_key",
        type=str,
        default="secret_key.pkl",
        help="Path to secret key file (required if --encrypted_repo is set)"
    )
    parser.add_argument(
        "--he_config",
        type=str,
        default="he_friendly_high",
        choices=["he_friendly_low", "he_friendly_high", "he_full"],
        help="HE config used when encrypting (must match encryption config)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["hellaswag", "lambada_openai"],
        help="Tasks to evaluate on"
    )
    args = parser.parse_args()
    
    # If encrypted repo specified, load and decrypt it
    if args.encrypted_repo:
        print(f"\n{'='*60}")
        print("EVALUATING ENCRYPTED MODEL")
        print(f"{'='*60}")
        print(f"Encrypted Repo: {args.encrypted_repo}")
        print(f"Secret Key: {args.secret_key}")
        print(f"HE Config: {args.he_config}")
        
        # Load and decrypt
        model = load_and_decrypt_encrypted_model(
            repo_id=args.encrypted_repo,
            secret_key_path=args.secret_key,
            config=args.he_config
        )
        
        # Evaluate using the decrypted model
        # Save it temporarily so lm-eval can load it
        print(f"\n{'='*60}")
        print("Running Evaluation")
        print(f"{'='*60}")
        
        import tempfile
        import torch
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, "model")
        
        try:
            # Save model state dict and config
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
            if hasattr(model, 'config') and model.config is not None:
                model.config.save_pretrained(model_path)
            
            # Load tokenizer from original repo
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.encrypted_repo)
            tokenizer.save_pretrained(model_path)
            
            results = evaluator.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_path}",
                tasks=args.tasks,
            )
            
            print(f"\nResults for encrypted model ({args.encrypted_repo}):")
            model_result = {
                "model": args.encrypted_repo,
                "type": "encrypted_decrypted",
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
            
            for task in args.tasks:
                task_res = results["results"].get(task, {})
                acc = task_res.get("acc,none", task_res.get("acc", "N/A"))
                print(f"  - {task}: {acc}")
                model_result["metrics"][task] = acc
            
            output_file = "benchmark_results_encrypted.json"
            with open(output_file, "w") as f:
                json.dump([model_result], f, indent=2)
            print(f"\nResults saved to {output_file}")
            
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("\nDone.")
        return
    
    # Default: Evaluate standard models with HE-ready transformations
    models_to_test = [
        {"name": "GPT-2", "model_args": "pretrained=gpt2"},
        {
            "name": "Qwen2.5-Coder-3B-Instruct",
            "model_args": "pretrained=Qwen/Qwen2.5-Coder-3B-Instruct,trust_remote_code=True",
        },
    ]

    print(f"Starting evaluation for: {[m['name'] for m in models_to_test]}")
    print("(Using HE-ready transformations, but NOT encrypted)")

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
                tasks=args.tasks,
            )
            print(f"\nResults for {model_info['name']}:")
            model_result = {"model": model_info["name"], "timestamp": timestamp, "metrics": {}}

            for task in args.tasks:
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


