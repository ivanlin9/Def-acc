import torch
import torch.nn as nn

from scanner.scanner import scan_model, build_he_pain_report, pretty_print_report, trace_functional_calls


class ToyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(16)
        self.act1 = nn.GELU()
        self.ln2 = nn.LayerNorm(16)
        self.act2 = nn.GELU()
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.act1(x)
        x = self.ln2(x)
        x = self.act2(x)
        return self.sm(x)


def test_scan_counts_basic():
    model = ToyNet()
    counts = scan_model(model)
    # Expected: 1 Softmax, 2 LayerNorm, 2 GELU
    assert counts["Softmax"] == 1
    assert counts["LayerNorm"] == 2
    assert counts["GELU"] == 2


def test_build_report_structure():
    model = ToyNet()
    report = build_he_pain_report(model)
    assert "Softmax" in report
    assert "LayerNorm" in report
    assert "GELU" in report
    assert report["Softmax"].count == 1
    assert isinstance(report["GELU"].note, str)


def test_pretty_print_nonempty():
    model = ToyNet()
    table = pretty_print_report(model)
    assert isinstance(table, str)
    assert "Op\tCount\tNotes" in table
    assert "Heuristic multiplicative-depth score" in table


class ToyFunctional(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = torch.nn.functional.gelu(x)
        x = torch.tanh(x)
        x = torch.softmax(x, dim=-1)
        return x


def test_trace_functional_calls():
    model = ToyFunctional()
    inputs = {"x": torch.randn(2, 10)}
    counts = trace_functional_calls(model, inputs)
    # At least 1 call registered for the functional ops we used
    assert counts["GELU"] >= 1
    assert counts["Tanh"] >= 1
    assert counts["Softmax"] >= 1


