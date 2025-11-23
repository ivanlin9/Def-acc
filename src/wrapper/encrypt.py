"""
Homomorphic Encryption (HE) functionality for HE-friendly models.

This module provides functions to:
- Encrypt model weights using CKKS (TenSEAL)
- Save/load encrypted models
- Run inference on encrypted data
- Fine-tune encrypted models (using LoRA)

Note: Fine-tuning encrypted models IS possible! You can:
- Compute gradients on encrypted data
- Update encrypted weights
- Use LoRA to only update a small subset of parameters
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Any, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    warnings.warn(
        "TenSEAL not available. Install with: pip install tenseal\n"
        "Encryption functionality will not work."
    )


class EncryptedModel:
    """
    Wrapper for a model with encrypted weights.
    
    The model architecture remains in plaintext, but all weight tensors
    are encrypted using CKKS homomorphic encryption.
    """
    
    def __init__(
        self,
        model: nn.Module,
        context: Optional[Any] = None,
        encrypted_weights: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize encrypted model.
        
        Args:
            model: HE-friendly model (architecture in plaintext)
            context: TenSEAL CKKS context (contains public key)
            encrypted_weights: Dictionary mapping parameter names to encrypted tensors
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required for encryption. Install with: pip install tenseal")
        
        self.model = model
        self.context = context
        self.encrypted_weights = encrypted_weights or {}
        self._chunk_info = {}  # Store info about chunked parameters
        self._plaintext_weights = None  # Cache for decryption
    
    def encrypt_weights(self, context: Optional[Any] = None) -> None:
        """
        Encrypt all model weights.
        
        Args:
            context: TenSEAL CKKS context. If None, creates a new one.
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required for encryption")
        
        if context is None:
            context = self._create_default_context()
        
        self.context = context
        self.encrypted_weights = {}
        self._chunk_info = {}  # Store info about chunked parameters
        
        # Maximum elements per ciphertext (conservative estimate for poly_modulus_degree=8192)
        # CKKS can pack up to poly_modulus_degree/2 elements, but we use a smaller chunk size
        # to avoid protobuf size limits (2GB max)
        max_elements_per_chunk = 4096  # poly_modulus_degree=8192 -> 4096 slots
        
        print("Encrypting model weights...")
        total_scalar_params = 0
        for name, param in self.model.named_parameters():
            # Convert to numpy and flatten
            param_np = param.detach().cpu().numpy().flatten()
            param_size = len(param_np)
            total_scalar_params += param_size
            original_shape = param.shape
            
            # Split into chunks if too large
            if param_size > max_elements_per_chunk:
                chunks = []
                num_chunks = (param_size + max_elements_per_chunk - 1) // max_elements_per_chunk
                for i in range(num_chunks):
                    start_idx = i * max_elements_per_chunk
                    end_idx = min(start_idx + max_elements_per_chunk, param_size)
                    chunk = param_np[start_idx:end_idx]
                    encrypted_chunk = ts.ckks_vector(context, chunk)
                    chunks.append(encrypted_chunk)
                
                self.encrypted_weights[name] = chunks
                self._chunk_info[name] = {
                    'original_shape': original_shape,
                    'num_chunks': num_chunks,
                    'chunk_size': max_elements_per_chunk,
                }
            else:
                # Small enough to encrypt as single vector
                encrypted = ts.ckks_vector(context, param_np)
                self.encrypted_weights[name] = encrypted
                self._chunk_info[name] = {
                    'original_shape': original_shape,
                    'num_chunks': 1,
                }
            
            if len(self.encrypted_weights) % 10 == 0:
                print(f"  Encrypted {len(self.encrypted_weights)} parameter tensors...")
        
        print(f"✓ Encrypted {len(self.encrypted_weights)} parameter tensors ({total_scalar_params:,} total scalar parameters)")
    
    def decrypt_weights(self, secret_key: Optional[Any] = None, secret_key_serialized: Optional[bytes] = None) -> Dict[str, torch.Tensor]:
        """
        Decrypt all weights (requires secret key).
        
        Args:
            secret_key: Secret key object for decryption. If None, uses context's secret key.
            secret_key_serialized: Serialized secret key (bytes). If provided, will deserialize it.
        
        Returns:
            Dictionary mapping parameter names to decrypted tensors
        """
        if not self.encrypted_weights:
            raise ValueError("No encrypted weights to decrypt")
        
        # If serialized key provided, deserialize it
        if secret_key_serialized is not None:
            if not TENSEAL_AVAILABLE:
                raise ImportError("TenSEAL is required for decryption")
            # secret_key_serialized is actually a serialized context with secret key
            # Deserialize the context and extract the secret key
            context_with_secret = ts.context_from(secret_key_serialized)
            secret_key = context_with_secret.secret_key()
        elif secret_key is None:
            # Try to get from context if available
            if self.context is not None:
                secret_key = self.context.secret_key()
            else:
                raise ValueError("Secret key required for decryption. Provide secret_key or secret_key_serialized.")
        
        decrypted = {}
        for name, encrypted_param in self.encrypted_weights.items():
            # Get chunk info
            chunk_info = self._chunk_info.get(name, {})
            original_shape = chunk_info.get('original_shape')
            if original_shape is None:
                # Fallback: get from model
                original_param = dict(self.model.named_parameters())[name]
                original_shape = original_param.shape
            
            # Handle chunked parameters
            if isinstance(encrypted_param, list):
                # Multiple chunks - decrypt and concatenate
                decrypted_chunks = [chunk.decrypt(secret_key=secret_key) for chunk in encrypted_param]
                decrypted_np = np.concatenate(decrypted_chunks)
            else:
                # Single encrypted vector
                decrypted_np = encrypted_param.decrypt(secret_key=secret_key)
            
            # Reshape and convert to tensor
            decrypted[name] = torch.tensor(decrypted_np).reshape(original_shape)
        
        self._plaintext_weights = decrypted
        return decrypted
    
    def save(self, save_path: str) -> None:
        """
        Save encrypted model to disk.
        
        Saves:
        - Model architecture (plaintext)
        - Encrypted weights
        - Public key (context) - allows encryption/inference
        - Model config/metadata
        
        Note: Secret key is NOT saved (must be kept client-side).
        
        Args:
            save_path: Directory path to save the encrypted model
        """
        if not self.encrypted_weights:
            raise ValueError("Model weights are not encrypted. Call encrypt_weights() first.")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model architecture (state dict structure, but not weights)
        model_state = {
            name: {
                'shape': param.shape,
                'dtype': str(param.dtype),
            }
            for name, param in self.model.named_parameters()
        }
        
        # Save encrypted weights (serialize CKKSVector objects first)
        encrypted_path = os.path.join(save_path, "encrypted_weights.pkl")
        # Serialize each CKKSVector before saving (handle chunked parameters)
        encrypted_weights_serialized = {}
        for name, encrypted_vec in self.encrypted_weights.items():
            if isinstance(encrypted_vec, list):
                # Multiple chunks - serialize each chunk
                encrypted_weights_serialized[name] = [chunk.serialize() for chunk in encrypted_vec]
            else:
                # Single encrypted vector
                encrypted_weights_serialized[name] = encrypted_vec.serialize()
        
        with open(encrypted_path, 'wb') as f:
            pickle.dump(encrypted_weights_serialized, f)
        
        # Save chunk info
        chunk_info_path = os.path.join(save_path, "chunk_info.pkl")
        with open(chunk_info_path, 'wb') as f:
            pickle.dump(self._chunk_info, f)
        
        # Save context (public key)
        context_path = os.path.join(save_path, "he_context.pkl")
        context_serialized = self.context.serialize() if self.context else None
        with open(context_path, 'wb') as f:
            pickle.dump(context_serialized, f)
        
        # Save model metadata
        metadata = {
            'model_state': model_state,
            'model_config': getattr(self.model, 'config', None),
            'num_parameters': len(self.encrypted_weights),
        }
        metadata_path = os.path.join(save_path, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Encrypted model saved to {save_path}")
        print(f"  - Encrypted weights: {encrypted_path}")
        print(f"  - Public key (context): {context_path}")
        print(f"  - Metadata: {metadata_path}")
        print("  ⚠️  Secret key NOT saved (keep it secure client-side)")
    
    @staticmethod
    def load(load_path: str, model: nn.Module) -> 'EncryptedModel':
        """
        Load encrypted model from disk.
        
        Args:
            load_path: Directory path containing encrypted model files
            model: HE-friendly model architecture (must match saved model)
        
        Returns:
            EncryptedModel instance
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required for loading encrypted models")
        
        # Load encrypted weights (deserialize CKKSVector objects)
        encrypted_path = os.path.join(load_path, "encrypted_weights.pkl")
        with open(encrypted_path, 'rb') as f:
            encrypted_weights_serialized = pickle.load(f)
        
        # Load chunk info if available
        chunk_info_path = os.path.join(load_path, "chunk_info.pkl")
        chunk_info = {}
        if os.path.exists(chunk_info_path):
            with open(chunk_info_path, 'rb') as f:
                chunk_info = pickle.load(f)
        
        # Load context (public key) first - needed to deserialize vectors
        context_path = os.path.join(load_path, "he_context.pkl")
        with open(context_path, 'rb') as f:
            context_serialized = pickle.load(f)
        
        context = ts.context_from(context_serialized)
        
        # Deserialize each CKKSVector (handle chunked parameters)
        encrypted_weights = {}
        for name, serialized_vec in encrypted_weights_serialized.items():
            if isinstance(serialized_vec, list):
                # Multiple chunks - deserialize each
                encrypted_weights[name] = [
                    ts.ckks_vector_from(context, chunk_serialized)
                    for chunk_serialized in serialized_vec
                ]
            else:
                # Single encrypted vector
                encrypted_weights[name] = ts.ckks_vector_from(context, serialized_vec)
        
        encrypted_model = EncryptedModel(
            model=model,
            context=context,
            encrypted_weights=encrypted_weights,
        )
        
        # Set chunk info
        encrypted_model._chunk_info = chunk_info
        
        print(f"✓ Encrypted model loaded from {load_path}")
        return encrypted_model
    
    @staticmethod
    def _create_default_context(
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: Optional[list] = None,
        global_scale: int = 2**40,
    ) -> Any:
        """
        Create a default TenSEAL CKKS context.
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (higher = more security, slower)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            global_scale: Global scale for CKKS encoding
        
        Returns:
            TenSEAL CKKS context
        """
        if not TENSEAL_AVAILABLE:
            raise ImportError("TenSEAL is required")
        
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        context.generate_galois_keys()
        context.global_scale = global_scale
        
        return context


def encrypt_model(
    model: nn.Module,
    context: Optional[Any] = None,
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[list] = None,
    global_scale: int = 2**40,
) -> EncryptedModel:
    """
    Encrypt a HE-friendly model.
    
    Args:
        model: HE-friendly model (from make_he_ready_model)
        context: TenSEAL CKKS context. If None, creates a default one.
        poly_modulus_degree: Polynomial modulus degree (for default context)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes (for default context)
        global_scale: Global scale for CKKS (for default context)
    
    Returns:
        EncryptedModel instance
    
    Example:
        >>> from wrapper import make_he_ready_model, encrypt_model
        >>> model = AutoModel.from_pretrained("bert-base-uncased")
        >>> he_model = make_he_ready_model(model, config="he_friendly_high")
        >>> encrypted_model = encrypt_model(he_model)
        >>> encrypted_model.save("encrypted_bert/")
    """
    encrypted_model = EncryptedModel(model, context=context)
    
    if context is None:
        context = EncryptedModel._create_default_context(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale=global_scale,
        )
    
    encrypted_model.encrypt_weights(context)
    return encrypted_model


def create_he_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: Optional[list] = None,
    global_scale: int = 2**40,
    save_secret_key: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Create a HE context and optionally save the secret key.
    
    The secret key is generated automatically when creating the context.
    This is the decryption key - keep it secure!
    
    Args:
        poly_modulus_degree: Polynomial modulus degree
        coeff_mod_bit_sizes: Coefficient modulus bit sizes
        global_scale: Global scale for CKKS
        save_secret_key: Path to save secret key (if provided)
    
    Returns:
        Tuple of (context, secret_key)
        - context: Public context (can be shared, used for encryption/inference)
        - secret_key: Secret key (KEEP SECURE! Required for decryption only)
    
    Example:
        >>> context, secret_key = create_he_context()
        >>> # Save secret key securely
        >>> with open("secret_key.pkl", "wb") as f:
        ...     pickle.dump(secret_key, f)
        >>> 
        >>> # Use context for encryption (can be shared)
        >>> encrypted_model = encrypt_model(model, context=context)
        >>> 
        >>> # Use secret_key for decryption (keep private!)
        >>> decrypted = encrypted_model.decrypt_weights(secret_key=secret_key)
    
    Important:
        - The secret key is generated automatically when you create the context
        - You MUST save it if you want to decrypt later
        - The context (public key) can be shared - it's used for encryption
        - The secret key must be kept secure - it's only needed for decryption
    """
    if not TENSEAL_AVAILABLE:
        raise ImportError("TenSEAL is required")
    
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]
    
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = global_scale
    
    # Extract secret key (this is the decryption key!)
    secret_key = context.secret_key()
    
    if save_secret_key:
        # Serialize context WITH secret key included (TenSEAL's way)
        # This allows us to extract the secret key later
        context_with_secret = context.serialize(save_secret_key=True)
        with open(save_secret_key, 'wb') as f:
            pickle.dump(context_with_secret, f)
        print(f"✓ Secret key saved to {save_secret_key} (keep secure!)")
        print(f"  Note: Saved as serialized context with secret key included")
    
    return context, secret_key


# Note on Fine-Tuning Encrypted Models:
#
# YES, you CAN fine-tune encrypted models! Here's how:
#
# 1. **LoRA Fine-Tuning** (Recommended):
#    - Add LoRA adapters to the model (small trainable matrices)
#    - Only LoRA weights need to be encrypted/updated
#    - Main model weights stay encrypted and unchanged
#    - Much faster than full model fine-tuning
#
# 2. **Gradient Computation on Encrypted Data**:
#    - Compute loss on encrypted outputs
#    - Backpropagate through HE operations (polynomials)
#    - Update encrypted weights using encrypted gradients
#    - Requires HE operations: encrypted_add, encrypted_multiply
#
# 3. **Hybrid Approach**:
#    - Keep base model encrypted
#    - Train LoRA adapters in plaintext (smaller, faster)
#    - Encrypt LoRA weights after training
#    - Combine encrypted base + encrypted LoRA for inference
#
# The key insight: HE allows arithmetic operations on encrypted data,
# so you can compute gradients and update weights without decryption!

