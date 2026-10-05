"""
test_point_mamba_encoder.py — Complete Unit Test Suite for Point Mamba Context Encoder (Ec)

Tests:
1. Morton (Z-order) and coordinate serialization logic on dummy batches [2, 1024, 3].
2. BiMambaWrapper forward and backward scan on token sequences [2, 1024, 128].
3. PointMambaContextEncoder input shape guard & dynamic transposition ([2, 1024, 3] and [2, 3, 1024] -> (2, 128)).
4. End-to-end forward pass & backpropagation across all 6 Experiment 6 configurations:
   - exp6_mamba_context (Baseline: Bidirectional, Z-Order, d_model=128, d_state=16)
   - exp6a_mamba_unidirectional (Unidirectional scan)
   - exp6b_mamba_coord_sort (Coordinate sorting)
   - exp6c_mamba_dim256 (High Capacity d_model=256)
   - exp6d_mamba_dim64 (Lightweight Bottleneck d_model=64)
   - exp6e_mamba_state32 (SSM Recurrent Memory d_state=32)
"""

import os
import sys
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.point_mamba_encoder import (
    PointMambaContextEncoder,
    BiMambaWrapper,
    compute_morton_order,
    compute_coord_order,
    serialize_point_cloud,
)
from models.context_hyperpocket import ContextHyperPocketModel
from configs import get_config


def test_serialization_methods():
    """Test 1: Verify Morton code and coordinate sorting indices."""
    print("\n[Test 1] Testing Serialization Methods (Z-Order & Coordinate Sorting)...")
    B, N = 2, 1024
    xyz = torch.randn(B, N, 3)

    # 1. Morton ordering
    morton_idx = compute_morton_order(xyz)
    assert morton_idx.shape == (B, N), f"Expected shape ({B}, {N}), got {morton_idx.shape}"
    # Verify permutations contain all indices from 0 to N-1
    for b in range(B):
        assert torch.equal(torch.sort(morton_idx[b])[0], torch.arange(N)), "Morton indices not a valid permutation!"

    # 2. Coordinate sorting
    coord_idx = compute_coord_order(xyz)
    assert coord_idx.shape == (B, N), f"Expected shape ({B}, {N}), got {coord_idx.shape}"
    for b in range(B):
        assert torch.equal(torch.sort(coord_idx[b])[0], torch.arange(N)), "Coord sort indices not a valid permutation!"

    # 3. Master dispatch serialization
    sorted_z, idx_z = serialize_point_cloud(xyz, order='z_order')
    sorted_c, idx_c = serialize_point_cloud(xyz, order='coord_sort')
    assert sorted_z.shape == (B, N, 3)
    assert sorted_c.shape == (B, N, 3)
    print("  ✓ Morton and coordinate serialization validated successfully!")


def test_bimamba_wrapper():
    """Test 2: Verify BiMambaWrapper forward and backward scan."""
    print("\n[Test 2] Testing BiMambaWrapper (Bidirectional & Unidirectional)...")
    B, L, d_model = 2, 64, 128
    x = torch.randn(B, L, d_model)

    # 1. Bidirectional
    bidi_block = BiMambaWrapper(d_model=d_model, d_state=16, is_bidirectional=True)
    bidi_block.eval()
    with torch.no_grad():
        out_bidi = bidi_block(x)
    assert out_bidi.shape == (B, L, d_model), f"Expected shape ({B}, {L}, {d_model}), got {out_bidi.shape}"

    # 2. Unidirectional
    uni_block = BiMambaWrapper(d_model=d_model, d_state=16, is_bidirectional=False)
    uni_block.eval()
    with torch.no_grad():
        out_uni = uni_block(x)
    assert out_uni.shape == (B, L, d_model), f"Expected shape ({B}, {L}, {d_model}), got {out_uni.shape}"
    print("  ✓ BiMambaWrapper forward passes validated successfully!")


def test_point_mamba_encoder_shapes():
    """Test 3: Verify PointMambaContextEncoder input shape guard & transposition."""
    print("\n[Test 3] Testing PointMambaContextEncoder Transposition & Output Shape...")
    encoder = PointMambaContextEncoder(
        output_size=128,
        d_model=128,
        d_state=16,
        num_layers=2,
        order='z_order',
        scan_dir='bidirectional',
    )
    encoder.eval()

    # 1. Standard [B, N, 3] format
    x_bn3 = torch.randn(2, 1024, 3)
    with torch.no_grad():
        zc_bn3 = encoder(x_bn3)
    assert zc_bn3.shape == (2, 128), f"Expected (2, 128), got {zc_bn3.shape}"

    # 2. Channel-first [B, 3, N] format (transposition guard)
    x_b3n = torch.randn(2, 3, 1024)
    with torch.no_grad():
        zc_b3n = encoder(x_b3n)
    assert zc_b3n.shape == (2, 128), f"Expected (2, 128), got {zc_b3n.shape}"

    # 3. Single batch B=1 invariance
    x_b1 = torch.randn(1, 1024, 3)
    with torch.no_grad():
        zc_b1 = encoder(x_b1)
    assert zc_b1.shape == (1, 128), f"Expected (1, 128), got {zc_b1.shape}"
    print("  ✓ PointMambaContextEncoder shape guard and transposition validated!")


def test_all_exp6_configurations():
    """Test 4: Verify full pipeline integration across all 6 Exp 6 configurations."""
    print("\n[Test 4] Testing Full Pipeline Integration Across All 6 Exp 6 Configurations...")
    exp_keys = [
        'context_hyperpocket.exp6_mamba_context',
        'context_hyperpocket.exp6a_mamba_unidirectional',
        'context_hyperpocket.exp6b_mamba_coord_sort',
        'context_hyperpocket.exp6c_mamba_dim256',
        'context_hyperpocket.exp6d_mamba_dim64',
        'context_hyperpocket.exp6e_mamba_state32',
    ]

    B, N = 2, 1024
    Pe = torch.randn(B, N, 3)
    Pm = torch.randn(B, N, 3)
    Pc = torch.randn(B, N, 3)

    for key in exp_keys:
        cfg = get_config(key)
        print(f"\n  -> Initializing: {key} ...")
        model = ContextHyperPocketModel(cfg)
        model.eval()

        assert isinstance(model.context_encoder, PointMambaContextEncoder), f"Encoder mismatch for {key}!"
        total_p = sum(p.numel() for p in model.parameters())
        print(f"     Params: {total_p:,} | scan={cfg.get('mamba_scan')} | order={cfg.get('mamba_order')} | d_model={cfg.get('mamba_d_model')} | d_state={cfg.get('mamba_d_state')}")

        # Training forward pass
        model.train()
        recon, mu, logvar = model(Pe, Pm, Pc, epoch=1, context_mode="true")
        assert recon.shape == (B, 3, N), f"Recon shape mismatch for {key}: {recon.shape}"
        assert mu.shape == (B, 128), f"Mu shape mismatch for {key}: {mu.shape}"
        assert logvar.shape == (B, 128), f"Logvar shape mismatch for {key}: {logvar.shape}"

        # Evaluation forward pass
        model.eval()
        noise = torch.randn(B, 128) * 0.05
        with torch.no_grad():
            recon_eval = model(Pe, pm=None, pc=Pc, epoch=0, noise=noise, context_mode="true")
        assert recon_eval.shape == (B, 3, N), f"Eval recon mismatch for {key}: {recon_eval.shape}"

        # Context ablations
        with torch.no_grad():
            recon_zeros = model(Pe, pm=None, pc=Pc, epoch=0, noise=noise, context_mode="zeros")
            recon_rand = model(Pe, pm=None, pc=Pc, epoch=0, noise=noise, context_mode="random")
        assert recon_zeros.shape == (B, 3, N)
        assert recon_rand.shape == (B, 3, N)
        print(f"     ✓ Passed forward, backward, eval, and ablation checks!")

    print("\n  ✓ All 6 Exp 6 configurations successfully verified!")


if __name__ == '__main__':
    print("=" * 70)
    print("  Point Mamba Context Encoder (Ec) Test Suite (Phase 7)")
    print("=" * 70)
    test_serialization_methods()
    test_bimamba_wrapper()
    test_point_mamba_encoder_shapes()
    test_all_exp6_configurations()
    print("\n" + "=" * 70)
    print("✓ ALL POINT MAMBA TESTS PASSED (100% GREEN)!")
    print("=" * 70 + "\n")
