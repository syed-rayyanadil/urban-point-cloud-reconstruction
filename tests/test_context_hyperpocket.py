"""Test suite for ContextHyperPocketModel architecture.

Mirrors models/context_hyperpocket.py.
"""
import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.context_hyperpocket import ContextHyperPocketModel


def test_context_hyperpocket():
    print("\n" + "=" * 70)
    print("  ContextHyperPocketModel Architecture Self-Test Suite (Phase 3)")
    print("=" * 70)

    test_cfg = {
        'n_points'                   : 1024,
        'random_encoder_output_size' : 128,
        'real_encoder_output_size'   : 128,
        'context_encoder_output_size': 128,
        'use_bias'                   : True,
        'target_network_layers'      : [32, 64, 128, 64],
        'progressive_norm_epochs'    : 100,
        'use_context'                : True,
    }

    device = torch.device('cpu')
    model = ContextHyperPocketModel(test_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Initialized Successfully:")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")

    B, N = 2, 1024
    Pe = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pm = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pc = torch.randn(B, N, 3, device=device, requires_grad=False)

    # --- Test 1: Training forward pass with use_context=True & Backprop ---
    print("\n[Test 1] Training Forward Pass with use_context=True:")
    model.train()
    recon_train, mu, logvar = model(Pe, Pm, Pc, epoch=1, device=device)

    print(f"  recon_train shape : {recon_train.shape}  # Expected: [2, 3, 1024]")
    print(f"  mu shape          : {mu.shape}           # Expected: [2, 128]")
    print(f"  logvar shape      : {logvar.shape}       # Expected: [2, 128]")

    assert recon_train.shape == (B, 3, N), f"Unexpected recon shape: {recon_train.shape}"
    assert mu.shape == (B, 128), f"Unexpected mu shape: {mu.shape}"
    assert logvar.shape == (B, 128), f"Unexpected logvar shape: {logvar.shape}"

    # Gradient check
    loss = recon_train.sum() + mu.sum() + logvar.sum()
    loss.backward()

    # Check that gradients flowed to all 3 encoders and the hypernetwork
    assert model.real_encoder.mu_layer.weight.grad is not None, "Real encoder received no gradients!"
    assert model.random_encoder.mu_layer.weight.grad is not None, "Random encoder received no gradients!"
    assert model.context_encoder.mu_layer.weight.grad is not None, "Context encoder received no gradients!"
    assert model.hyper_network.backbone[0].weight.grad is not None, "HyperNetwork received no gradients!"
    print("  ✓ Gradients successfully flowed to Ee, Em, Ec, and HyperNetwork!")

    # --- Test 2: Zero-Context Ablation (use_context=False) ---
    print("\n[Test 2] Zero-Context Ablation Forward Pass (use_context=False):")
    model.eval()
    with torch.no_grad():
        recon_zero = model(Pe, Pm, Pc, epoch=1, device=device, use_context=False)
    print(f"  recon_zero shape  : {recon_zero.shape}  # Expected: [2, 3, 1024]")
    assert recon_zero.shape == (B, 3, N), f"Unexpected recon_zero shape: {recon_zero.shape}"
    print("  ✓ Zero-context ablation forward pass executed successfully!")

    # --- Test 3: Evaluation Generative Inference (noise passed, pm=None) ---
    print("\n[Test 3] Evaluation Generative Inference (pm=None, noise override):")
    noise = torch.randn(B, 128, device=device) * 0.05
    with torch.no_grad():
        recon_eval = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise)
    print(f"  recon_eval shape  : {recon_eval.shape}  # Expected: [2, 3, 1024]")
    assert recon_eval.shape == (B, 3, N), f"Unexpected recon_eval shape: {recon_eval.shape}"
    print("  ✓ Generative inference mode executed successfully!")

    print("\n" + "=" * 70)
    print("✓ ALL CONTEXT HYPERPOCKET MODEL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_context_hyperpocket()
