"""
scanner: Tools to inspect PyTorch/HF models for HE-unfriendly operations.
"""

from .scanner import (
    scan_model,
    build_he_pain_report,
    pretty_print_report,
    trace_functional_calls,
)

__all__ = [
    "scan_model",
    "build_he_pain_report",
    "pretty_print_report",
    "trace_functional_calls",
]


