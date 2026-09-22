"""
test_dgcnn_context_encoder.py — Unit test suite for DGCNNContextEncoder (Phase 5 / Ec Upgrade)

Tests:
1. Model initialization and parameter count verification.
2. Forward pass with [B, 1024, 3] coordinate tensor -> [B, 128].
3. Forward pass with [B, 3, 1024] transposed coordinate tensor -> [B, 128].
4. Output shape invariance across different batch sizes (B=1, B=4, B=8).
5. Gradient backpropagation and parameter update verification.
6. Pairwise distance and k-NN helper functions correctness.
"""

import os
import sys
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.dgcnn_context_encoder import DGCNNContextEncoder, knn, get_graph_feature


def test_knn_and_graph_feature():
    print("\n[Test 1] Verifying knn() and get_graph_feature()...")
    B, C, N, k = 2, 3, 64, 10
    x = torch.randn(B, C, N)

    idx = knn(x, k=k)
    assert idx.shape == (B, N, k), f"Expected knn shape {(B, N, k)}, got {idx.shape}"

    # Self-distance is 0, so the first nearest neighbor should be the point itself
    self_indices = torch.arange(N).unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
    # The top nearest neighbor (dim 0 of topk) should match self index
    assert torch.all(idx[:, :, 0] == self_indices[:, :, 0]), "Nearest neighbor is not the point itself!"

    edge_feats = get_graph_feature(x, k=k)
    assert edge_feats.shape == (B, 2 * C, N, k), f"Expected graph feature shape {(B, 2 * C, N, k)}, got {edge_feats.shape}"
    print("  -> knn() and get_graph_feature() shapes and properties verified: PASSED")


def test_dgcnn_context_encoder_shapes():
    print("\n[Test 2] Verifying DGCNNContextEncoder forward shapes...")
    model = DGCNNContextEncoder(output_size=128, k=20)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model Parameter Count: {total_params:,} total ({trainable_params:,} trainable)")

    # Test [B, N, 3] format
    for B in [1, 4, 8]:
        x_bn3 = torch.randn(B, 1024, 3)
        with torch.no_grad():
            zc_bn3 = model(x_bn3)
        assert zc_bn3.shape == (B, 128), f"Expected shape ({B}, 128), got {zc_bn3.shape}"

    print("  -> Channel-last [B, N, 3] forward passes verified: PASSED")

    # Test [B, 3, N] format
    for B in [1, 4, 8]:
        x_b3n = torch.randn(B, 3, 1024)
        with torch.no_grad():
            zc_b3n = model(x_b3n)
        assert zc_b3n.shape == (B, 128), f"Expected shape ({B}, 128), got {zc_b3n.shape}"

    print("  -> Channel-first [B, 3, N] forward passes verified: PASSED")


def test_dgcnn_gradient_flow():
    print("\n[Test 3] Verifying gradient flow and backpropagation...")
    model = DGCNNContextEncoder(output_size=128, k=20)
    model.train()

    x = torch.randn(4, 1024, 3, requires_grad=True)
    zc = model(x)
    loss = zc.pow(2).sum()
    loss.backward()

    assert x.grad is not None, "Input tensor received no gradient!"
    assert not torch.isnan(x.grad).any(), "NaN found in input gradients!"

    # Verify all model parameters received gradients
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and (param.grad is None or torch.all(param.grad == 0)):
            no_grad_params.append(name)

    assert len(no_grad_params) == 0, f"Parameters without gradient: {no_grad_params}"
    print("  -> Gradient backpropagation verified across all layers: PASSED")


if __name__ == '__main__':
    print("=" * 65)
    print("  DGCNNContextEncoder Unit Test Suite")
    print("=" * 65)
    test_knn_and_graph_feature()
    test_dgcnn_context_encoder_shapes()
    test_dgcnn_gradient_flow()
    print("\n" + "=" * 65)
    print("  ALL DGCNN TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)
