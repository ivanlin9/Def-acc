import torch
import torch.nn as nn

from scanner.scanner import trace_functional_calls


class TanhOnlyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.fc(x)
        x = torch.tanh(x)  # only tanh, no gelu
        return x


def test_tanh_only_tracing():
    model = TanhOnlyNet()
    inputs = {"x": torch.randn(2, 8)}
    counts = trace_functional_calls(model, inputs)
    assert counts["Tanh"] >= 1
    assert counts["GELU"] == 0


