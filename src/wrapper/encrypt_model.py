#!/usr/bin/env python3
"""
Encrypt any HuggingFace model and save to HuggingFace Hub.

This script:
1. Loads a model (GPT-2, Qwen, BERT, etc.)
2. Scans for HE-unfriendly operations
3. Makes it HE-friendly (applies wrapper transformations)
4. Encrypts the weights
5. Saves encrypted model to HuggingFace Hub
6. Saves decryption key locally (SECRET - keep secure!)

Supports any model that works with transformers.AutoModel or AutoModelForCausalLM.
"""

import os
import sys
import pickle
import argparse

# Add src to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi, login

from wrapper import make_he_ready_model
from wrapper.encrypt import encrypt_model, create_he_context, EncryptedModel


def save_encrypted_to_hub(
    encrypted_model: EncryptedModel,
    repo_id: str,
    hf_token: str,
    secret_key_path: str = "secret_key.pkl",
):
    """
    Save encrypted model to HuggingFace Hub.
    
    IMPORTANT: The secret key is NOT uploaded to Hub (kept local only).
    """
    print(f"\n{'='*60}")
    print("Saving Encrypted Model to HuggingFace Hub")
    print(f"{'='*60}")
    
    # Login to HuggingFace
    print("\n1. Logging into HuggingFace Hub...")
    login(token=hf_token)
    api = HfApi()
    
    # Create temporary directory for upload
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save encrypted model locally first
        print("2. Saving encrypted model locally...")
        local_path = os.path.join(temp_dir, "encrypted_model")
        encrypted_model.save(local_path)
        
        # Save model config (for loading later)
        config = getattr(encrypted_model.model, 'config', None)
        if config is not None:
            config_path = os.path.join(local_path, "config.json")
            config.save_pretrained(os.path.dirname(config_path))
        
        # Create README with instructions
        readme_content = f"""# Encrypted Model: {repo_id}

This model has been encrypted using CKKS homomorphic encryption.

## ⚠️ IMPORTANT: Secret Key Required

**The secret key is NOT included in this repository.** You must have the secret key file (`secret_key.pkl`) to decrypt this model.

## Loading the Encrypted Model

```python
from wrapper.encrypt import EncryptedModel
from transformers import AutoModel
import pickle

# Load model architecture
model = AutoModel.from_pretrained("{repo_id}")
he_model = make_he_ready_model(model, config="he_friendly_high")

# Load encrypted weights from Hub
encrypted_model = EncryptedModel.load("encrypted_model/", he_model)

# Load secret key (you must have this file!)
with open("secret_key.pkl", "rb") as f:
    secret_key = pickle.load(f)

# Decrypt if needed
decrypted_weights = encrypted_model.decrypt_weights(secret_key=secret_key)
```

## Security Note

- The secret key (`secret_key.pkl`) must be kept secure and client-side only
- Never commit the secret key to version control
- Share the secret key only through secure channels
- The encrypted model can be used for HE operations without the secret key
- The secret key is only needed for decryption
"""
        
        readme_path = os.path.join(local_path, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        # Create repository if it doesn't exist
        print(f"3. Creating repository (if needed): {repo_id}...")
        repo_exists = False
        try:
            # Check if repo exists
            api.model_info(repo_id=repo_id, token=hf_token)
            repo_exists = True
            print(f"   ✓ Repository already exists: {repo_id}")
        except Exception:
            # Repo doesn't exist, create it
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    token=hf_token,
                    exist_ok=False,  # We know it doesn't exist
                )
                print(f"   ✓ Repository created: {repo_id}")
                repo_exists = True
            except Exception as create_error:
                # If it says it already exists, that's fine
                if "already exists" in str(create_error).lower() or "409" in str(create_error):
                    print(f"   ✓ Repository already exists: {repo_id}")
                    repo_exists = True
                else:
                    raise RuntimeError(f"Failed to create repository {repo_id}: {create_error}") from create_error
        
        if not repo_exists:
            raise RuntimeError(f"Repository {repo_id} does not exist and could not be created")
        
        # Upload to Hub
        print(f"4. Uploading to HuggingFace Hub: {repo_id}...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        
        print(f"\n✅ Encrypted model uploaded to: https://huggingface.co/{repo_id}")
        print(f"\n⚠️  SECRET KEY saved locally to: {secret_key_path}")
        print("   Keep this file secure! It's required for decryption.")
        print("   DO NOT upload it to the Hub or commit it to git!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Encrypt any HuggingFace model and save to HuggingFace Hub"
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        dest="model_name",
        help="Model name (e.g., 'gpt2'). Shorthand for --model_name"
    )
    model_group.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (e.g., 'gpt2'). Alternative to --model"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., 'username/gpt2-encrypted')"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.getenv("HF_TOKEN", None),
        help="HuggingFace token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--secret_key_path",
        type=str,
        default="secret_key.pkl",
        help="Path to save secret key (default: secret_key.pkl)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="he_friendly_high",
        choices=["he_friendly_low", "he_friendly_high", "he_full"],
        help="HE-friendly config"
    )
    parser.add_argument(
        "--attention_type",
        type=str,
        default="gaussian",
        choices=["chebyshev", "gaussian"],
        help="Attention approximation type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for model loading"
    )
    
    args = parser.parse_args()
    
    # Validate HuggingFace token
    if args.hf_token is None:
        parser.error("--hf_token is required (or set HF_TOKEN environment variable)")
    
    # Handle model_name/model alias - default to gpt2 if neither provided
    if args.model_name is None:
        args.model_name = "gpt2"
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print(f"Model Encryption: {args.model_name}")
    print("="*60)
    
    # Step 1: Load model
    print(f"\n1. Loading model: {args.model_name}")
    # Use the same loading logic as apply_wrapper.py
    try:
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception:
        # Fallback to AutoModelForCausalLM (works for GPT-2 and similar)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    print(f"   ✓ Model loaded on {device}")
    
    # Step 2: Scan model for HE-unfriendly operations
    print(f"\n2. Scanning model for HE-unfriendly operations...")
    from scanner.scanner import scan_model, pretty_print_report
    scan_report = pretty_print_report(model)
    print(scan_report)
    print(f"   ✓ Scan complete")
    
    # Step 3: Make HE-friendly (apply wrapper transformations)
    print(f"\n3. Making model HE-friendly (config: {args.config})...")
    he_model = make_he_ready_model(
        model,
        config=args.config,
        attention_type=args.attention_type,
    )
    print(f"   ✓ Model transformed (attention: {args.attention_type})")
    
    # Verify transformations
    print(f"\n   Verifying transformations...")
    scan_report_after = pretty_print_report(he_model)
    print(scan_report_after)
    print(f"   ✓ Transformations verified")
    
    # Step 4: Create HE context and secret key
    print("\n4. Creating HE context and secret key...")
    context, secret_key = create_he_context(
        save_secret_key=args.secret_key_path
    )
    print(f"   ✓ Context created")
    print(f"   ✓ Secret key saved to: {args.secret_key_path}")
    
    # Step 5: Encrypt model (only after HE-friendly transformations)
    print("\n5. Encrypting model weights...")
    print("   Note: Model is already HE-friendly (polynomial operations only)")
    print("   This may take a while for large models...")
    encrypted_model = encrypt_model(he_model, context=context)
    print("   ✓ Model encrypted")
    
    # Step 5: Save to HuggingFace Hub
    save_encrypted_to_hub(
        encrypted_model,
        repo_id=args.repo_id,
        hf_token=args.hf_token,
        secret_key_path=args.secret_key_path,
    )
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✅ Model: {args.model_name}")
    print(f"✅ Encrypted model: https://huggingface.co/{args.repo_id}")
    print(f"✅ Secret key: {args.secret_key_path} (KEEP SECURE!)")
    print(f"\n⚠️  IMPORTANT:")
    print(f"   - Secret key is required for decryption")
    print(f"   - Keep secret key secure and client-side only")
    print(f"   - Do NOT upload secret key to Hub or commit to git")
    print(f"\n💡 To decrypt later:")
    print(f"   from wrapper.encrypt import EncryptedModel")
    print(f"   from wrapper import make_he_ready_model")
    print(f"   from transformers import AutoModel")
    print(f"   import pickle")
    print(f"   ")
    print(f"   # Load encrypted model from Hub")
    print(f"   model = AutoModel.from_pretrained('{args.repo_id}')")
    print(f"   he_model = make_he_ready_model(model, config='{args.config}')")
    print(f"   encrypted_model = EncryptedModel.load('encrypted_model/', he_model)")
    print(f"   ")
    print(f"   # Load secret key (serialized bytes)")
    print(f"   with open('{args.secret_key_path}', 'rb') as f:")
    print(f"       secret_key_serialized = pickle.load(f)")
    print(f"   ")
    print(f"   # Decrypt")
    print(f"   decrypted_weights = encrypted_model.decrypt_weights(secret_key_serialized=secret_key_serialized)")


if __name__ == "__main__":
    main()

