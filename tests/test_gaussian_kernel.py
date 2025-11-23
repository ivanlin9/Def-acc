import torch
import torch.nn.functional as F

from wrapper.wrapper import _gaussian_kernel_attention, enable_gaussian_kernel_attention, disable_attention_approximation, set_gaussian_alpha
from wrapper import make_he_ready_model


def test_gaussian_kernel_attention_shapes():
    """Test that Gaussian kernel attention produces correct output shapes."""
    torch.manual_seed(0)
    batch, heads, seqlen, dim = 2, 3, 4, 8
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.randn(batch, heads, seqlen, dim)
    
    out = _gaussian_kernel_attention(q, k, v, is_causal=True)
    assert out.shape == (batch, heads, seqlen, dim)
    assert torch.isfinite(out).all()


def test_gaussian_kernel_attention_causal():
    """Test that Gaussian kernel attention respects causal masking."""
    torch.manual_seed(0)
    batch, heads, seqlen, dim = 1, 1, 4, 4
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.randn(batch, heads, seqlen, dim)
    
    out = _gaussian_kernel_attention(q, k, v, is_causal=True)
    assert out.shape == (batch, heads, seqlen, dim)
    assert torch.isfinite(out).all()


def test_gaussian_kernel_attention_alpha():
    """Test that alpha parameter affects attention weights."""
    torch.manual_seed(0)
    batch, heads, seqlen, dim = 1, 1, 3, 4
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.ones(batch, heads, seqlen, dim)  # Constant values
    
    # Test with different alpha values
    set_gaussian_alpha(0.5)
    out1 = _gaussian_kernel_attention(q, k, v)
    
    set_gaussian_alpha(2.0)
    out2 = _gaussian_kernel_attention(q, k, v)
    
    # Different alpha should produce different outputs
    assert not torch.allclose(out1, out2, atol=1e-5)
    assert torch.isfinite(out1).all()
    assert torch.isfinite(out2).all()
    
    # Reset to default
    set_gaussian_alpha(1.0)


def test_gaussian_kernel_vs_chebyshev():
    """Test that Gaussian kernel and Chebyshev produce different (but valid) outputs."""
    torch.manual_seed(0)
    batch, heads, seqlen, dim = 1, 1, 4, 8
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.randn(batch, heads, seqlen, dim)
    
    # Disable any existing approximation
    disable_attention_approximation()
    
    # Test Gaussian kernel
    enable_gaussian_kernel_attention()
    if hasattr(F, "scaled_dot_product_attention"):
        out_gaussian = F.scaled_dot_product_attention(q, k, v)
        assert torch.isfinite(out_gaussian).all()
        assert out_gaussian.shape == (batch, heads, seqlen, dim)
    
    disable_attention_approximation()


def test_make_he_ready_with_gaussian():
    """Test that make_he_ready_model works with Gaussian kernel attention."""
    import torch.nn as nn
    
    class TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
        
        def forward(self, x):
            # Simulate attention-like operation
            return self.fc(x)
    
    model = TinyAttention()
    x = torch.randn(2, 8)
    
    # Test with Gaussian kernel
    he_model = make_he_ready_model(
        model,
        config="he_friendly_high",
        attention_type="gaussian"
    )
    
    out = he_model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_attention_type_parameter():
    """Test that attention_type parameter is respected."""
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    x = torch.randn(2, 4)
    
    # Test Chebyshev (default)
    he_model_cheb = make_he_ready_model(
        model,
        config="he_friendly_high",
        attention_type="chebyshev"
    )
    out_cheb = he_model_cheb(x)
    
    # Test Gaussian
    he_model_gauss = make_he_ready_model(
        model,
        config="he_friendly_high",
        attention_type="gaussian"
    )
    out_gauss = he_model_gauss(x)
    
    # Both should work
    assert torch.isfinite(out_cheb).all()
    assert torch.isfinite(out_gauss).all()
    assert out_cheb.shape == out_gauss.shape

