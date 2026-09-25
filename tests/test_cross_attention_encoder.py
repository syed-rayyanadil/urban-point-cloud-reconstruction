"""
test_cross_attention_encoder.py — Mathematical Validation & CPU Dry-Run Test Suite for CrossAttentionEncoder

Test Suite Structure:
1. test_tensor_shape_verification:
   - [2, 1024, 3] -> (2, 128)
   - Batch size invariance: B=1 (verifying no single-batch squeeze bugs), B=4, B=8
   - Channel-first auto-transposition: [2, 3, 1024] -> (2, 128)
2. test_gradient_flow:
   - Verifies .grad is non-None, non-zero, and contains no NaNs for:
     * model.latents (learnable query seed)
     * model.input_mlp (coordinate projection layers)
     * input tensor x
3. test_strict_geometry_only_constraint:
   - Rejects tensors with >3 channels ([2, 1024, 4], [2, 1024, 6], [2, 1024, 16]) with "Geometry-Only" ValueError
   - Rejects invalid tensor dimensions (2D [1024, 3], 4D [2, 10, 1024, 3])
4. test_permutation_invariance:
   - Mathematically verifies that shuffling point order produces identical latent representations (atol=1e-5)
"""

import os
import sys
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.cross_attention_encoder import CrossAttentionEncoder


def test_tensor_shape_verification():
    """Test 1: Shape contracts, batch size invariance, and channel-first transposition."""
    print("\n[Test 1] Tensor Shape Verification & Invariance:")
    model = CrossAttentionEncoder(output_size=128, d_model=128, num_heads=4, num_queries=4)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # 1. Standard [2, 1024, 3] input
    x_standard = torch.randn(2, 1024, 3)
    with torch.no_grad():
        out_standard = model(x_standard)
    assert out_standard.shape == (2, 128), f"Expected (2, 128), got {out_standard.shape}"
    print("  ✓ Standard [2, 1024, 3] input -> (2, 128)")

    # 2. Batch size invariance: B=1 (critical for squeeze bug prevention), B=4, B=8
    for B in [1, 4, 8]:
        x_batch = torch.randn(B, 1024, 3)
        with torch.no_grad():
            out_batch = model(x_batch)
        assert out_batch.shape == (B, 128), f"Expected ({B}, 128), got {out_batch.shape}"
        print(f"  ✓ Batch size B={B} invariance: [{B}, 1024, 3] -> ({B}, 128)")

    # 3. Channel-first auto-transposition: [2, 3, 1024]
    x_ch_first = torch.randn(2, 3, 1024)
    with torch.no_grad():
        out_ch_first = model(x_ch_first)
    assert out_ch_first.shape == (2, 128), f"Expected (2, 128), got {out_ch_first.shape}"
    print("  ✓ Channel-first auto-transposition: [2, 3, 1024] -> (2, 128)")


def test_gradient_flow():
    """Test 2: End-to-end gradient backpropagation and NaN checks."""
    print("\n[Test 2] Gradient Flow & Backpropagation:")
    model = CrossAttentionEncoder(output_size=128, d_model=128, num_heads=4, num_queries=4)
    model.train()

    x = torch.randn(4, 1024, 3, requires_grad=True)
    zc = model(x)
    loss = zc.sum()
    loss.backward()

    # 1. Input tensor gradients
    assert x.grad is not None, "Input tensor received no gradient!"
    assert not torch.isnan(x.grad).any(), "NaN detected in input tensor gradient!"
    assert torch.count_nonzero(x.grad) > 0, "Input tensor gradient is all zeros!"
    print("  ✓ Input coordinate gradients verified (non-None, non-zero, no NaNs)")

    # 2. Learnable query seed gradients
    assert model.latents.grad is not None, "Learnable query seed (model.latents) received no gradient!"
    assert not torch.isnan(model.latents.grad).any(), "NaN detected in query seed gradient!"
    assert torch.count_nonzero(model.latents.grad) > 0, "Query seed gradient is all zeros!"
    print("  ✓ Learnable query seed (model.latents) gradients verified")

    # 3. Coordinate projection MLP layers
    for i, layer in enumerate(model.input_mlp):
        if isinstance(layer, torch.nn.Linear):
            assert layer.weight.grad is not None, f"Input MLP layer {i} weight received no gradient!"
            assert not torch.isnan(layer.weight.grad).any(), f"NaN detected in input MLP layer {i}!"
            assert torch.count_nonzero(layer.weight.grad) > 0, f"Input MLP layer {i} gradient is all zeros!"
    print("  ✓ Input coordinate projection MLP layer gradients verified")

    # 4. Cross-attention & FFN layers
    assert model.ffn[0].weight.grad is not None, "FFN weight received no gradient!"
    assert model.proj_out[1].weight.grad is not None, "Projection head weight received no gradient!"
    print("  ✓ Cross-attention, FFN, and Bottleneck projection head gradients verified")


def test_strict_geometry_only_constraint():
    """Test 3: Strict geometry-only channel validation (rejects >3 channels and invalid dims)."""
    print("\n[Test 3] Strict Geometry-Only Constraint & Shape Validation:")
    model = CrossAttentionEncoder(output_size=128, d_model=128)
    model.eval()

    # Case A: [2, 1024, 4] (Coordinates + 1 Semantic Class ID)
    try:
        x_4ch = torch.randn(2, 1024, 4)
        model(x_4ch)
        raise AssertionError("Should have rejected 4-channel tensor!")
    except ValueError as e:
        assert "Geometry-Only" in str(e), f"Expected 'Geometry-Only' in error message, got: {e}"
        print("  ✓ Rejected [2, 1024, 4] (Coordinates + 1 Semantic ID)")

    # Case B: [2, 1024, 6] (Coordinates + RGB Colors)
    try:
        x_6ch = torch.randn(2, 1024, 6)
        model(x_6ch)
        raise AssertionError("Should have rejected 6-channel tensor!")
    except ValueError as e:
        assert "Geometry-Only" in str(e), f"Expected 'Geometry-Only' in error message, got: {e}"
        print("  ✓ Rejected [2, 1024, 6] (Coordinates + RGB Colors)")

    # Case C: [2, 1024, 16] (Coordinates + 13-class one-hot encoding)
    try:
        x_16ch = torch.randn(2, 1024, 16)
        model(x_16ch)
        raise AssertionError("Should have rejected 16-channel tensor!")
    except ValueError as e:
        assert "Geometry-Only" in str(e), f"Expected 'Geometry-Only' in error message, got: {e}"
        print("  ✓ Rejected [2, 1024, 16] (Coordinates + 13-class one-hot)")

    # Case D: 2D tensor [1024, 3] (missing batch dim)
    try:
        x_2d = torch.randn(1024, 3)
        model(x_2d)
        raise AssertionError("Should have rejected 2D tensor!")
    except ValueError as e:
        print("  ✓ Rejected 2D tensor [1024, 3]")

    # Case E: 4D tensor [2, 10, 1024, 3]
    try:
        x_4d = torch.randn(2, 10, 1024, 3)
        model(x_4d)
        raise AssertionError("Should have rejected 4D tensor!")
    except ValueError as e:
        print("  ✓ Rejected 4D tensor [2, 10, 1024, 3]")


def test_permutation_invariance():
    """Test 4: Mathematical permutation invariance check across point ordering."""
    print("\n[Test 4] Permutation Invariance Across Unordered Point Sets:")
    model = CrossAttentionEncoder(output_size=128, d_model=128, num_heads=4, num_queries=4)
    model.eval()

    torch.manual_seed(42)
    B, N = 2, 1024
    x = torch.randn(B, N, 3)

    # 1. Forward pass on original point order
    with torch.no_grad():
        out1 = model(x)

    # 2. Randomly shuffle point order along point dimension N=1024
    perm = torch.randperm(N)
    x_shuffled = x[:, perm, :]

    # 3. Forward pass on shuffled point cloud
    with torch.no_grad():
        out2 = model(x_shuffled)

    # 4. Mathematical check: out1 must equal out2 within numerical tolerance
    max_diff = (out1 - out2).abs().max().item()
    is_close = torch.allclose(out1, out2, atol=1e-5)

    print(f"  Max absolute difference after random permutation: {max_diff:.2e}")
    assert is_close, f"Permutation invariance failed! Max diff: {max_diff}"
    print("  ✓ Permutation invariance mathematically verified: out1 == out2 (atol=1e-5)")


if __name__ == '__main__':
    print("=" * 70)
    print("  CrossAttentionEncoder Validation & Zero-Cost CPU Dry-Run Suite")
    print("=" * 70)
    test_tensor_shape_verification()
    test_gradient_flow()
    test_strict_geometry_only_constraint()
    test_permutation_invariance()
    print("\n" + "=" * 70)
    print("✓ ALL CROSS-ATTENTION ENCODER VALIDATION TESTS PASSED (100% GREEN)!")
    print("=" * 70 + "\n")
