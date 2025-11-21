import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper import make_he_ready_model


class GeluToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.act = F.gelu  # gelu-like via functional
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GeluModuleToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.act_mod = nn.GELU()
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_mod(self.fc1(x)))


def test_wrapper_replaces_functional_gelu():
    torch.manual_seed(0)
    model = GeluToyMLP()
    x = torch.randn(2, 8)
    y0 = model(x)
    he_model = make_he_ready_model(model, config="he_friendly_low")
    y1 = he_model(x)
    assert y0.shape == y1.shape
    # Ensure the act attribute no longer references F.gelu
    assert he_model.act is not F.gelu


def test_wrapper_replaces_module_gelu():
    torch.manual_seed(0)
    model = GeluModuleToyMLP()
    x = torch.randn(2, 8)
    y0 = model(x)
    he_model = make_he_ready_model(model, config="he_friendly_low")
    y1 = he_model(x)
    assert y0.shape == y1.shape
    # Check the module type changed from nn.GELU to something else
    assert not isinstance(he_model.act_mod, nn.GELU)


