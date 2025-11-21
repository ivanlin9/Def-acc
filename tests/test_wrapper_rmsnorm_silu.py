import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper import make_he_ready_model


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        inv = torch.rsqrt(ms)
        return x * inv * self.weight


class SiluActModule(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.fc(x))


class TinyCombo(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.block = SiluActModule(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.block(x)


def test_rmsnorm_and_silu_replacement():
    torch.manual_seed(0)
    model = TinyCombo()
    x = torch.randn(2, 8)
    y0 = model(x)
    he_model = make_he_ready_model(model, config="he_friendly_low")
    y1 = he_model(x)
    assert y0.shape == y1.shape
    # Confirm RMSNorm replaced
    assert he_model.norm.__class__.__name__ != "RMSNorm"
    # Confirm SiLU replaced
    assert he_model.block.act_fn is not F.silu


