"""
cross_attention_encoder.py — Pure PyTorch Set Transformer / Latent Cross-Attention Context Encoder (Ec)

Drop-in replacement for the Context Encoder (Ec) in the 3D VAE completion pipeline.
Extracts permutation-invariant global semantics and aggregated geometric representations from
surrounding spatial context point clouds (Pc) via learnable latent queries and multi-head cross-attention.

Pure PyTorch implementation with zero external dependencies (no PyG / PyTorch3D / CUDA extensions).
Enforces a strict geometry-only constraint: only accepts 3D coordinates (X, Y, Z).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionEncoder(nn.Module):
    """Set Transformer / Latent Cross-Attention Context Encoder (Ec).

    Architecture:
        1. Coordinate Projection MLP: [B, 1024, 3] -> Linear(3->64) -> LayerNorm -> LeakyReLU(0.2)
                                                   -> Linear(64->d_model) -> LayerNorm -> LeakyReLU(0.2)
                                                   -> Tokens: [B, 1024, d_model]
        2. Learnable Latent Query Seed: S in R^[1, num_queries, d_model]
        3. Multi-Head Cross-Attention (PMA / Set Transformer):
           - Queries (Q): Expanded latent queries [B, num_queries, d_model]
           - Keys (K)   : Projected point tokens [B, 1024, d_model]
           - Values (V) : Projected point tokens [B, 1024, d_model]
           - Output     : [B, num_queries, d_model]
        4. Transformer Stability Blocks:
           - Residual cross-attention connection: LayerNorm(Q + Dropout(attn_out))
           - Feed-Forward Network: Linear(d_model->dim_feedforward) -> GELU -> Dropout -> Linear(->d_model)
           - Residual FFN connection: LayerNorm(x + Dropout(ffn_out))
        5. Safe Dimension Reduction (No raw .squeeze()):
           - Flatten(start_dim=1) -> Linear(num_queries * d_model -> output_size) -> [B, 128]
           - Guarantees strict [B, output_size] output shape across any batch size (including B=1).

    Args:
        output_size (int): Output latent context vector dimension (default: 128).
        d_model (int): Hidden feature dimension for cross-attention tokens (default: 128).
        num_heads (int): Number of multi-head attention heads (default: 4).
        dim_feedforward (int): Dimension of feed-forward network (default: 512).
        dropout (float): Dropout probability (default: 0.0).
        num_queries (int): Number of learnable latent query tokens (default: 4).
    """

    def __init__(
        self,
        output_size: int = 128,
        d_model: int = 128,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_queries: int = 4,
    ):
        super().__init__()
        self.output_size = output_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_queries = num_queries

        # 1. Coordinate Projection MLP: [B, 1024, 3] -> [B, 1024, d_model]
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # 2. Learnable Latent Query Seed Tensor: [1, num_queries, d_model]
        self.latents = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.normal_(self.latents, mean=0.0, std=1.0 / math.sqrt(d_model))

        # 3. Multi-Head Cross-Attention Block (Q = latents, K = tokens, V = tokens)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Transformer Stability Blocks (LayerNorm + Dropout + FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(dim_feedforward, d_model),
        )

        # 5. Safe Bottleneck Projection Head (Guaranteed [B, output_size] without raw .squeeze())
        self.proj_out = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(num_queries * d_model, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Context Point Cloud encoding.

        Args:
            x (torch.Tensor): Point cloud tensor of shape [B, 1024, 3] or [B, 3, 1024].

        Returns:
            torch.Tensor: Latent context vector z_c of shape [B, output_size] (default 128).

        Raises:
            ValueError: If input tensor does not have exactly 3 coordinate channels (X, Y, Z).
        """
        # Dimensionality check
        if x.dim() != 3:
            raise ValueError(
                f"CrossAttentionEncoder expects a 3D point cloud tensor of shape [B, 1024, 3] or [B, 3, 1024], "
                f"got tensor with {x.dim()} dimensions: {x.shape}"
            )

        # Auto-transpose channel-first format: [B, 3, N] -> [B, N, 3]
        if x.size(1) == 3 and x.size(2) != 3:
            x = x.transpose(1, 2).contiguous()

        # Strict geometry-only channel validation to prevent semantic/color leakage
        if x.size(-1) != 3:
            raise ValueError(
                f"Geometry-Only Constraint Violated: CrossAttentionEncoder strictly accepts 3 "
                f"coordinate channels (X, Y, Z). Received input with {x.size(-1)} channels: {x.shape}. "
                f"Semantic labels, intensity, or RGB channels must not be passed to prevent label leakage."
            )

        B, N, _ = x.shape

        # 1. Project spatial coordinates into d_model token representations
        # tokens: [B, N, d_model]
        tokens = self.input_mlp(x)

        # 2. Expand learnable latent query seed across batch
        # queries: [B, num_queries, d_model]
        queries = self.latents.expand(B, -1, -1)

        # 3. Multi-Head Cross-Attention: Q=queries, K=tokens, V=tokens
        # attn_out: [B, num_queries, d_model]
        attn_out, _ = self.cross_attn(query=queries, key=tokens, value=tokens)
        queries = self.norm1(queries + self.dropout(attn_out))

        # 4. Transformer Feed-Forward Network with residual connection
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + self.dropout(ffn_out))

        # 5. Safe Bottleneck Projection (Flatten -> Linear) -> [B, output_size]
        zc = self.proj_out(queries)

        return zc


if __name__ == '__main__':
    print("=" * 65)
    print("  CrossAttentionEncoder Self-Verification")
    print("=" * 65)

    model = CrossAttentionEncoder(
        output_size=128,
        d_model=128,
        num_heads=4,
        dim_feedforward=512,
        num_queries=4,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # 1. Test [B, 1024, 3] with B=8
    x1 = torch.randn(8, 1024, 3)
    out1 = model(x1)
    print(f"Input shape [8, 1024, 3] -> Output shape: {out1.shape}")
    assert out1.shape == (8, 128), f"Expected (8, 128), got {out1.shape}"

    # 2. Test [B, 3, 1024] with B=8 (channel-first)
    x2 = torch.randn(8, 3, 1024)
    out2 = model(x2)
    print(f"Input shape [8, 3, 1024] -> Output shape: {out2.shape}")
    assert out2.shape == (8, 128), f"Expected (8, 128), got {out2.shape}"

    # 3. Test B=1 to verify no squeeze degradation
    x_single = torch.randn(1, 1024, 3)
    out_single = model(x_single)
    print(f"Input shape [1, 1024, 3] -> Output shape: {out_single.shape}")
    assert out_single.shape == (1, 128), f"Expected (1, 128), got {out_single.shape}"

    # 4. Test Geometry-only constraint violation
    try:
        x_leak = torch.randn(4, 1024, 6)  # e.g., XYZ + RGB
        model(x_leak)
        raise AssertionError("Failed to catch geometry constraint violation with >3 channels!")
    except ValueError as e:
        print(f"Geometry constraint check: PASSED (caught: {e})")

    # 5. Gradient flow check
    model.train()
    x3 = torch.randn(4, 1024, 3, requires_grad=True)
    out3 = model(x3)
    loss = out3.sum()
    loss.backward()
    assert model.latents.grad is not None, "Gradients did not flow to learnable query tensor!"
    assert model.input_mlp[0].weight.grad is not None, "Gradients did not flow to input MLP!"
    assert x3.grad is not None, "Gradients did not flow to input tensor!"
    print("Gradient backward check: PASSED")
    print("All self-verification checks passed successfully!")
