import torch
import torch.nn as nn

from wrapper import make_he_ready_model


class LNNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.ln(x))


def test_layernorm_replacement_and_forward():
    torch.manual_seed(0)
    model = LNNet()
    x = torch.randn(2, 8)
    y0 = model(x)
    he_model = make_he_ready_model(model, config="he_acts_norm")
    y1 = he_model(x)
    assert y0.shape == y1.shape
    assert he_model.ln.__class__.__name__ != "LayerNorm"


