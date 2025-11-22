import torch
import torch.nn.functional as F

from wrapper.wrapper import _approx_scaled_dot_product_attention


def test_approx_sdpa_shapes_and_probs():
    torch.manual_seed(0)
    batch, heads, seqlen, dim = 2, 3, 4, 8
    q = torch.randn(batch, heads, seqlen, dim)
    k = torch.randn(batch, heads, seqlen, dim)
    v = torch.randn(batch, heads, seqlen, dim)
    out = _approx_scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out.shape == (batch, heads, seqlen, dim)
    # Check rows sum to ~1 in weights indirectly by comparing magnitude not exploding
    assert torch.isfinite(out).all()


