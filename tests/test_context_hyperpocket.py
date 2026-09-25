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
from models.dgcnn_context_encoder import DGCNNContextEncoder
from models.cross_attention_encoder import CrossAttentionEncoder


def test_context_hyperpocket_pointnet():
    print("\n" + "=" * 70)
    print("  ContextHyperPocketModel (PointNet Ec) Architecture Self-Test")
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
        'encoder_type'               : 'pointnet',
    }

    device = torch.device('cpu')
    model = ContextHyperPocketModel(test_cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPointNet Model Initialized Successfully:")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")

    B, N = 2, 1024
    Pe = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pm = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pc = torch.randn(B, N, 3, device=device, requires_grad=False)

    # --- Test 1: Training forward pass with context_mode='true' & Backprop ---
    print("\n[Test 1] Training Forward Pass with context_mode='true':")
    model.train()
    recon_train, mu, logvar = model(Pe, Pm, Pc, epoch=1, device=device, context_mode="true")

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

    # --- Test 2: Zero-Context Ablation (context_mode='zeros') ---
    print("\n[Test 2] Zero-Context Ablation Forward Pass (context_mode='zeros'):")
    model.eval()
    with torch.no_grad():
        recon_zeros = model(Pe, Pm, Pc, epoch=1, device=device, context_mode="zeros")
    print(f"  recon_zeros shape : {recon_zeros.shape}  # Expected: [2, 3, 1024]")
    assert recon_zeros.shape == (B, 3, N), f"Unexpected recon_zeros shape: {recon_zeros.shape}"
    print("  ✓ Zero-context ablation forward pass executed successfully!")

    # --- Test 3: Random Noise Context Ablation (context_mode='random') ---
    print("\n[Test 3] Random Noise Context Ablation Forward Pass (context_mode='random'):")
    with torch.no_grad():
        recon_random = model(Pe, Pm, Pc, epoch=1, device=device, context_mode="random")
    print(f"  recon_random shape: {recon_random.shape}  # Expected: [2, 3, 1024]")
    assert recon_random.shape == (B, 3, N), f"Unexpected recon_random shape: {recon_random.shape}"
    print("  ✓ Random noise context ablation forward pass executed successfully!")

    # --- Test 4: Backward Compatibility with use_context=False ---
    print("\n[Test 4] Backward Compatibility Test (use_context=False):")
    with torch.no_grad():
        recon_compat = model(Pe, Pm, Pc, epoch=1, device=device, use_context=False)
    print(f"  recon_compat shape: {recon_compat.shape}  # Expected: [2, 3, 1024]")
    assert recon_compat.shape == (B, 3, N), f"Unexpected recon_compat shape: {recon_compat.shape}"
    print("  ✓ Backward compatibility check passed!")

    # --- Test 5: Evaluation Generative Inference (noise passed, pm=None) ---
    print("\n[Test 5] Evaluation Generative Inference (pm=None, noise override):")
    noise = torch.randn(B, 128, device=device) * 0.05
    with torch.no_grad():
        recon_eval = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="true")
    print(f"  recon_eval shape  : {recon_eval.shape}  # Expected: [2, 3, 1024]")
    assert recon_eval.shape == (B, 3, N), f"Unexpected recon_eval shape: {recon_eval.shape}"
    print("  ✓ Generative inference mode executed successfully!")


def test_context_hyperpocket_dgcnn():
    print("\n" + "=" * 70)
    print("  ContextHyperPocketModel (DGCNN Ec) Architecture Integration Test")
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
        'encoder_type'               : 'dgcnn',
        'dgcnn_k'                    : 20,
        'dgcnn_dropout'              : 0.0,
    }

    device = torch.device('cpu')
    model = ContextHyperPocketModel(test_cfg).to(device)

    # Assert DGCNN encoder is properly instantiated
    assert isinstance(model.context_encoder, DGCNNContextEncoder), "Context encoder is not DGCNNContextEncoder!"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDGCNN Model Initialized Successfully:")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")

    B, N = 2, 1024
    Pe = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pm = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pc = torch.randn(B, N, 3, device=device, requires_grad=False)

    # --- Test 1: Training forward pass with DGCNN Ec (context_mode='true') ---
    print("\n[DGCNN Test 1] Training Forward Pass with DGCNN Ec:")
    model.train()
    recon_train, mu, logvar = model(Pe, Pm, Pc, epoch=1, device=device, context_mode="true")

    print(f"  recon_train shape : {recon_train.shape}  # Expected: [2, 3, 1024]")
    print(f"  mu shape          : {mu.shape}           # Expected: [2, 128]")
    print(f"  logvar shape      : {logvar.shape}       # Expected: [2, 128]")

    assert recon_train.shape == (B, 3, N), f"Unexpected recon shape: {recon_train.shape}"
    assert mu.shape == (B, 128), f"Unexpected mu shape: {mu.shape}"
    assert logvar.shape == (B, 128), f"Unexpected logvar shape: {logvar.shape}"

    # Gradient check for DGCNN integration
    loss = recon_train.sum() + mu.sum() + logvar.sum()
    loss.backward()

    # Check gradients in DGCNN encoder components
    assert model.context_encoder.conv1[0].weight.grad is not None, "DGCNN conv1 received no gradients!"
    assert model.context_encoder.conv2[0].weight.grad is not None, "DGCNN conv2 received no gradients!"
    assert model.context_encoder.conv_fuse[0].weight.grad is not None, "DGCNN conv_fuse received no gradients!"
    assert model.context_encoder.projection_head[0].weight.grad is not None, "DGCNN projection head received no gradients!"
    assert model.hyper_network.backbone[0].weight.grad is not None, "HyperNetwork received no gradients!"
    print("  ✓ Gradients successfully flowed through DGCNN Ec and into HyperNetwork!")

    # --- Test 2: Evaluation Generative Inference with DGCNN Ec ---
    print("\n[DGCNN Test 2] Evaluation Generative Inference:")
    model.eval()
    noise = torch.randn(B, 128, device=device) * 0.05
    with torch.no_grad():
        recon_eval = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="true")
    print(f"  recon_eval shape  : {recon_eval.shape}  # Expected: [2, 3, 1024]")
    assert recon_eval.shape == (B, 3, N), f"Unexpected recon_eval shape: {recon_eval.shape}"
    print("  ✓ DGCNN Generative inference mode executed successfully!")

    # --- Test 3: Ablations with DGCNN model ---
    with torch.no_grad():
        recon_zeros = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="zeros")
        recon_random = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="random")
    assert recon_zeros.shape == (B, 3, N), f"Unexpected recon_zeros shape: {recon_zeros.shape}"
    assert recon_random.shape == (B, 3, N), f"Unexpected recon_random shape: {recon_random.shape}"
    print("  ✓ DGCNN Zero-context and random-context ablation checks passed!")


def test_context_hyperpocket_cross_attention():
    print("\n" + "=" * 70)
    print("  ContextHyperPocketModel (CrossAttention Ec) Architecture Integration Test")
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
        'encoder_type'               : 'cross_attention',
        'ca_d_model'                 : 128,
        'ca_num_heads'               : 4,
        'ca_num_queries'             : 1,
        'ca_dim_feedforward'         : 512,
        'ca_dropout'                 : 0.0,
    }

    device = torch.device('cpu')
    model = ContextHyperPocketModel(test_cfg).to(device)

    # Assert CrossAttention encoder is properly instantiated
    assert isinstance(model.context_encoder, CrossAttentionEncoder), "Context encoder is not CrossAttentionEncoder!"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nCrossAttention Model Initialized Successfully:")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")

    B, N = 2, 1024
    Pe = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pm = torch.randn(B, N, 3, device=device, requires_grad=False)
    Pc = torch.randn(B, N, 3, device=device, requires_grad=False)

    # --- Test 1: Training forward pass with CrossAttention Ec ---
    print("\n[CrossAttention Test 1] Training Forward Pass with CrossAttention Ec:")
    model.train()
    recon_train, mu, logvar = model(Pe, Pm, Pc, epoch=1, device=device, context_mode="true")

    print(f"  recon_train shape : {recon_train.shape}  # Expected: [2, 3, 1024]")
    print(f"  mu shape          : {mu.shape}           # Expected: [2, 128]")
    print(f"  logvar shape      : {logvar.shape}       # Expected: [2, 128]")

    assert recon_train.shape == (B, 3, N), f"Unexpected recon shape: {recon_train.shape}"
    assert mu.shape == (B, 128), f"Unexpected mu shape: {mu.shape}"
    assert logvar.shape == (B, 128), f"Unexpected logvar shape: {logvar.shape}"

    # Gradient check for CrossAttention integration
    loss = recon_train.sum() + mu.sum() + logvar.sum()
    loss.backward()

    # Check gradients in CrossAttention encoder components
    assert model.context_encoder.latents.grad is not None, "Query seed latents received no gradients!"
    assert model.context_encoder.input_mlp[0].weight.grad is not None, "CA input MLP received no gradients!"
    assert model.context_encoder.ffn[0].weight.grad is not None, "CA FFN received no gradients!"
    assert model.context_encoder.proj_out[1].weight.grad is not None, "CA projection head received no gradients!"
    assert model.hyper_network.backbone[0].weight.grad is not None, "HyperNetwork received no gradients!"
    print("  ✓ Gradients successfully flowed through CrossAttention Ec and into HyperNetwork!")

    # --- Test 2: Evaluation Generative Inference with CrossAttention Ec ---
    print("\n[CrossAttention Test 2] Evaluation Generative Inference:")
    model.eval()
    noise = torch.randn(B, 128, device=device) * 0.05
    with torch.no_grad():
        recon_eval = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="true")
    print(f"  recon_eval shape  : {recon_eval.shape}  # Expected: [2, 3, 1024]")
    assert recon_eval.shape == (B, 3, N), f"Unexpected recon_eval shape: {recon_eval.shape}"
    print("  ✓ CrossAttention Generative inference mode executed successfully!")

    # --- Test 3: Ablations with CrossAttention model ---
    with torch.no_grad():
        recon_zeros = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="zeros")
        recon_random = model(Pe, pm=None, pc=Pc, epoch=0, device=device, noise=noise, context_mode="random")
    assert recon_zeros.shape == (B, 3, N), f"Unexpected recon_zeros shape: {recon_zeros.shape}"
    assert recon_random.shape == (B, 3, N), f"Unexpected recon_random shape: {recon_random.shape}"
    print("  ✓ CrossAttention Zero-context and random-context ablation checks passed!")


# Alias for backward compatibility
test_context_hyperpocket = test_context_hyperpocket_pointnet


if __name__ == "__main__":
    test_context_hyperpocket_pointnet()
    test_context_hyperpocket_dgcnn()
    test_context_hyperpocket_cross_attention()
    print("\n" + "=" * 70)
    print("✓ ALL CONTEXT HYPERPOCKET (POINTNET, DGCNN, CROSS-ATTENTION) TESTS PASSED!")
    print("=" * 70 + "\n")
