#!/usr/bin/env python3
"""
Example script showing how to encrypt models and run inference.

This demonstrates:
1. Making a model HE-friendly
2. Encrypting the model weights
3. Saving the encrypted model
4. Loading and using encrypted models
5. Fine-tuning considerations
"""

import os
import sys
import pickle

# Add src to path
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import torch
from transformers import AutoModel, AutoTokenizer

from wrapper import make_he_ready_model
from wrapper.encrypt import encrypt_model, EncryptedModel, create_he_context


def example_encrypt_and_save():
    """Example: Encrypt a model and save it."""
    print("=" * 60)
    print("Example 1: Encrypt and Save Model")
    print("=" * 60)
    
    # 1. Load and prepare model
    print("\n1. Loading model...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 2. Make HE-friendly
    print("2. Making model HE-friendly...")
    he_model = make_he_ready_model(
        model,
        config="he_friendly_high",
        attention_type="gaussian"  # Faster under HE
    )
    
    # 3. Create HE context (public key)
    print("3. Creating HE context...")
    context, secret_key = create_he_context(
        save_secret_key="secret_key.pkl"  # Save secret key securely
    )
    
    # 4. Encrypt model
    print("4. Encrypting model...")
    encrypted_model = encrypt_model(he_model, context=context)
    
    # 5. Save encrypted model
    print("5. Saving encrypted model...")
    encrypted_model.save("encrypted_bert_model/")
    
    print("\n✓ Model encrypted and saved!")
    print("  - Encrypted model: encrypted_bert_model/")
    print("  - Secret key: secret_key.pkl (keep secure!)")
    print("\n⚠️  Note: Secret key is required for decryption.")
    print("   Keep it secure and client-side only!")


def example_load_and_use():
    """Example: Load encrypted model (inference would require HE operations)."""
    print("\n" + "=" * 60)
    print("Example 2: Load Encrypted Model")
    print("=" * 60)
    
    # 1. Load model architecture
    print("\n1. Loading model architecture...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    he_model = make_he_ready_model(model, config="he_friendly_high")
    
    # 2. Load encrypted model
    print("2. Loading encrypted weights...")
    encrypted_model = EncryptedModel.load("encrypted_bert_model/", he_model)
    
    print("\n✓ Encrypted model loaded!")
    print("  - Model has encrypted weights")
    print("  - Can perform HE operations (inference, fine-tuning)")
    print("  - Requires secret key for decryption")


def example_decrypt():
    """Example: Decrypt model weights (requires secret key)."""
    print("\n" + "=" * 60)
    print("Example 3: Decrypt Model (Requires Secret Key)")
    print("=" * 60)
    
    # 1. Load encrypted model
    model = AutoModel.from_pretrained("bert-base-uncased")
    he_model = make_he_ready_model(model, config="he_friendly_high")
    encrypted_model = EncryptedModel.load("encrypted_bert_model/", he_model)
    
    # 2. Load secret key
    print("\n1. Loading secret key...")
    with open("secret_key.pkl", "rb") as f:
        secret_key = pickle.load(f)
    
    # 3. Decrypt weights
    print("2. Decrypting weights...")
    decrypted_weights = encrypted_model.decrypt_weights(secret_key=secret_key)
    
    # 4. Load into model
    print("3. Loading decrypted weights into model...")
    he_model.load_state_dict(decrypted_weights)
    
    print("\n✓ Model decrypted and ready for plaintext inference!")
    print("  - Weights are now in plaintext")
    print("  - Can run normal inference")


def explain_fine_tuning():
    """Explain how fine-tuning works on encrypted models."""
    print("\n" + "=" * 60)
    print("Fine-Tuning Encrypted Models")
    print("=" * 60)
    
    print("""
YES, you CAN fine-tune encrypted models! Here's how:

1. **LoRA Fine-Tuning** (Recommended):
   - Add small LoRA adapters to the model
   - Only LoRA weights are trainable (much smaller)
   - Main model weights stay encrypted
   - Update LoRA weights using encrypted gradients
   - Much faster than full model fine-tuning

2. **How It Works**:
   - Forward pass: Run on encrypted data → encrypted output
   - Loss: Compute on encrypted outputs (polynomial operations)
   - Backward: Compute gradients through HE operations
   - Update: Add encrypted gradients to encrypted weights
   - All operations happen on ciphertexts!

3. **Example Workflow**:
   ```python
   # Add LoRA to encrypted model
   from peft import LoraConfig, get_peft_model
   
   lora_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
   encrypted_model.model = get_peft_model(encrypted_model.model, lora_config)
   
   # Fine-tune (gradients computed on encrypted data)
   for batch in dataloader:
       encrypted_input = encrypt(batch, context)
       encrypted_output = encrypted_model(encrypted_input)
       loss = compute_loss(encrypted_output, encrypted_target)
       loss.backward()  # Gradients are encrypted!
       optimizer.step()  # Updates encrypted LoRA weights
   ```

4. **Key Insight**:
   - HE allows arithmetic on encrypted data
   - Gradients are just arithmetic operations
   - So you can compute and apply gradients without decryption!
   - The secret key is only needed for final decryption (client-side)
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Encryption examples")
    parser.add_argument(
        "--example",
        type=str,
        choices=["encrypt", "load", "decrypt", "explain"],
        default="encrypt",
        help="Which example to run"
    )
    args = parser.parse_args()
    
    if args.example == "encrypt":
        example_encrypt_and_save()
    elif args.example == "load":
        example_load_and_use()
    elif args.example == "decrypt":
        example_decrypt()
    elif args.example == "explain":
        explain_fine_tuning()

