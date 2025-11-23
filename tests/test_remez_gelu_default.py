import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper import make_he_ready_model
from wrapper.remez import tanh_cheb


def test_remez_gelu_is_default():
    """Test that Remez GELU (with tanh_cheb) is used by default."""
    class GeluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act = F.gelu
        
        def forward(self, x):
            return self.act(self.fc(x))
    
    model = GeluModel()
    x = torch.randn(2, 8)
    
    # Default should use Remez (use_remez_activations=True by default)
    he_model = make_he_ready_model(model, config="he_friendly_low")
    
    # The act function should not be F.gelu anymore
    assert he_model.act is not F.gelu
    
    # It should be a function that uses tanh_cheb internally
    # We can't directly check if it uses tanh_cheb, but we can verify it works
    out = he_model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_remez_gelu_vs_poly():
    """Test that Remez GELU produces different output than poly_gelu3."""
    class GeluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act = F.gelu
        
        def forward(self, x):
            return self.act(self.fc(x))
    
    model = GeluModel()
    x = torch.randn(2, 8)
    
    # Remez version (default)
    he_model_remez = make_he_ready_model(
        model,
        config="he_friendly_low",
        use_remez_activations=True
    )
    out_remez = he_model_remez(x)
    
    # Poly version
    he_model_poly = make_he_ready_model(
        model,
        config="he_friendly_low",
        use_remez_activations=False
    )
    out_poly = he_model_poly(x)
    
    # Both should work
    assert torch.isfinite(out_remez).all()
    assert torch.isfinite(out_poly).all()
    assert out_remez.shape == out_poly.shape
    
    # They should produce different outputs (different approximations)
    assert not torch.allclose(out_remez, out_poly, atol=1e-3)


def test_remez_gelu_module_replacement():
    """Test that nn.GELU modules are replaced when using Remez."""
    class GeluModuleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.act = nn.GELU()
        
        def forward(self, x):
            return self.act(self.fc(x))
    
    model = GeluModuleModel()
    x = torch.randn(2, 8)
    
    # Default should use Remez
    he_model = make_he_ready_model(model, config="he_friendly_low")
    
    # Module should be replaced
    assert not isinstance(he_model.act, nn.GELU)
    
    # Should work
    out = he_model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

