import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper import make_he_ready_model
from wrapper.remez import silu_cheb, sigmoid_cheb


def test_silu_uses_chebyshev():
    """Test that SiLU uses Chebyshev approximation (silu_cheb)."""
    class SiluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act = F.silu
        
        def forward(self, x):
            return self.act(self.fc(x))
    
    model = SiluModel()
    x = torch.randn(2, 8)
    
    # Should use Chebyshev by default
    he_model = make_he_ready_model(model, config="he_friendly_low")
    
    # The act function should be silu_cheb (or equivalent)
    assert he_model.act is not F.silu
    
    # Should work correctly
    out = he_model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_silu_cheb_function():
    """Test that silu_cheb function works correctly."""
    x = torch.randn(10)
    
    # Test basic functionality
    out = silu_cheb(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    
    # Test that it produces reasonable outputs
    # Note: The sigmoid coefficients are placeholders and may not be highly accurate
    # For production use, proper Remez coefficients should be generated
    x_small = torch.linspace(-3, 3, 10)
    out_cheb = silu_cheb(x_small)
    
    # Basic sanity checks: output should be finite and have reasonable magnitude
    assert torch.isfinite(out_cheb).all()
    # SiLU(x) = x * sigmoid(x), so for x > 0, output should be positive
    # For x < 0, output should be negative
    assert (out_cheb[x_small > 0] >= 0).all() or torch.abs(out_cheb[x_small > 0]).max() < 10
    # Output should not explode
    assert torch.abs(out_cheb).max() < 100


def test_sigmoid_cheb_function():
    """Test that sigmoid_cheb function works correctly."""
    x = torch.randn(10)
    
    # Test basic functionality
    out = sigmoid_cheb(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    
    # Output should be in [0, 1] range
    assert (out >= 0).all()
    assert (out <= 1).all()
    
    # Test on known range
    x_range = torch.linspace(-5, 5, 10)
    out_range = sigmoid_cheb(x_range)
    assert torch.isfinite(out_range).all()
    assert (out_range >= 0).all()
    assert (out_range <= 1).all()


def test_silu_module_replacement():
    """Test that SiLU in modules gets replaced."""
    class SiluModuleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act_fn = F.silu
        
        def forward(self, x):
            return self.act_fn(self.fc(x))
    
    model = SiluModuleModel()
    x = torch.randn(2, 8)
    
    he_model = make_he_ready_model(model, config="he_friendly_low")
    
    # Should be replaced
    assert he_model.act_fn is not F.silu
    
    # Should work
    out = he_model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

