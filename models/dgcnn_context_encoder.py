"""
dgcnn_context_encoder.py — Pure PyTorch DGCNN (Dynamic Graph CNN) Context Encoder (Ec)

Drop-in replacement for the Context Encoder (Ec) in the 3D VAE completion pipeline.
Extracts rich local geometric structures and global semantics from surrounding spatial
context point clouds (Pc) via dynamic k-NN graphs and EdgeConv operations (Wang et al., 2019).

Pure PyTorch implementation with zero external dependencies (no PyG / PyTorch3D / CUDA extensions).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute pairwise Euclidean distance in feature space and return top-k neighbor indices.

    Memory-efficient formulation:
        ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>

    Args:
        x (torch.Tensor): Feature or coordinate tensor of shape [B, C, N].
        k (int): Number of nearest neighbors.

    Returns:
        torch.Tensor: Neighbor indices of shape [B, N, k].
    """
    # inner: [B, N, N]
    inner = -2.0 * torch.bmm(x.transpose(2, 1), x)
    # xx: [B, 1, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    # pairwise_dist: [B, N, N] (negative squared Euclidean distance)
    pairwise_dist = -xx - inner - xx.transpose(2, 1)

    # Top-k largest values in negative distance correspond to smallest distances (nearest neighbors)
    # idx: [B, N, k]
    idx = pairwise_dist.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None) -> torch.Tensor:
    """Construct dynamic graph edge features for each point and its k-nearest neighbors.

    Edge feature definition:
        e_{ij} = (x_j - x_i, x_i)

    Args:
        x (torch.Tensor): Point features tensor of shape [B, C, N].
        k (int): Number of nearest neighbors (used if idx is None).
        idx (torch.Tensor, optional): Precomputed neighbor indices of shape [B, N, k].

    Returns:
        torch.Tensor: Graph edge features of shape [B, 2 * C, N, k].
    """
    B, C, N = x.size()

    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # [B, 1, 1]
    idx = idx + idx_base  # [B, N, k]
    idx = idx.view(-1)    # [B * N * k]

    # Gather neighbor features
    x_trans = x.transpose(2, 1).contiguous()               # [B, N, C]
    feature = x_trans.view(B * N, -1)[idx, :]              # [B * N * k, C]
    feature = feature.view(B, N, k, C).permute(0, 3, 1, 2).contiguous()  # [B, C, N, k] (x_j)

    # Central point features replicated along k dimension
    x_central = x.unsqueeze(-1).expand(-1, -1, -1, k)      # [B, C, N, k] (x_i)

    # Edge feature: (x_j - x_i, x_i) -> [B, 2 * C, N, k]
    edge_feature = torch.cat([feature - x_central, x_central], dim=1)
    return edge_feature


class DGCNNContextEncoder(nn.Module):
    """Dynamic Graph CNN Context Encoder (Ec) with Multi-Scale EdgeConv Aggregation.

    Architecture:
        1. EdgeConv 1: [B, 3, N] -> k-NN -> [B, 6, N, k] -> Conv2d(6->64->64) -> MaxPool -> [B, 64, N]
        2. EdgeConv 2: [B, 64, N] -> k-NN -> [B, 128, N, k] -> Conv2d(128->64->64) -> MaxPool -> [B, 64, N]
        3. EdgeConv 3: [B, 64, N] -> k-NN -> [B, 128, N, k] -> Conv2d(128->128) -> MaxPool -> [B, 128, N]
        4. EdgeConv 4: [B, 128, N] -> k-NN -> [B, 256, N, k] -> Conv2d(256->256) -> MaxPool -> [B, 256, N]
        5. Fusion: Cat([x1, x2, x3, x4]) -> [B, 512, N] -> Conv1d(512->1024) -> [B, 1024, N]
        6. Global Pooling: Concat(Max(1024), Avg(1024)) -> [B, 2048]
        7. Projection Head: Linear(2048->512->256->output_size) -> [B, output_size] (default 128)

    Args:
        output_size (int): Output latent context vector dimension (default: 128).
        k (int): Number of nearest neighbors for dynamic graph construction (default: 20).
        dropout (float): Dropout probability in bottleneck projection head (default: 0.0).
    """

    def __init__(self, output_size: int = 128, k: int = 20, dropout: float = 0.0):
        super().__init__()
        self.output_size = output_size
        self.k = k
        self.dropout = dropout

        # --- EdgeConv Block 1 (Input coordinates C=3 -> 64) ---
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # --- EdgeConv Block 2 (C=64 -> 64) ---
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # --- EdgeConv Block 3 (C=64 -> 128) ---
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # --- EdgeConv Block 4 (C=128 -> 256) ---
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # --- Multi-scale feature fusion (64 + 64 + 128 + 256 = 512 -> 1024) ---
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # --- Bottleneck projection MLP: 2048 -> 512 -> 256 -> output_size (128) ---
        proj_layers = [
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        if dropout > 0.0:
            proj_layers.append(nn.Dropout(p=dropout))

        proj_layers.extend([
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ])
        if dropout > 0.0:
            proj_layers.append(nn.Dropout(p=dropout))

        proj_layers.append(nn.Linear(256, output_size, bias=True))

        self.projection_head = nn.Sequential(*proj_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Context Point Cloud encoding.

        Args:
            x (torch.Tensor): Point cloud tensor of shape [B, N, 3] or [B, 3, N].

        Returns:
            torch.Tensor: Context latent embedding z_c of shape [B, output_size].
        """
        # Ensure channel-first orientation: [B, 3, N]
        if x.dim() == 3 and x.size(-1) == 3:
            x = x.transpose(1, 2).contiguous()  # [B, 3, N]

        B, _, N = x.size()

        # 1. EdgeConv Block 1
        x1_edge = get_graph_feature(x, k=self.k)            # [B, 6, N, k]
        x1 = self.conv1(x1_edge)                            # [B, 64, N, k]
        x1 = x1.max(dim=-1, keepdim=False)[0]               # [B, 64, N]

        # 2. EdgeConv Block 2
        x2_edge = get_graph_feature(x1, k=self.k)           # [B, 128, N, k]
        x2 = self.conv2(x2_edge)                            # [B, 64, N, k]
        x2 = x2.max(dim=-1, keepdim=False)[0]               # [B, 64, N]

        # 3. EdgeConv Block 3
        x3_edge = get_graph_feature(x2, k=self.k)           # [B, 128, N, k]
        x3 = self.conv3(x3_edge)                            # [B, 128, N, k]
        x3 = x3.max(dim=-1, keepdim=False)[0]               # [B, 128, N]

        # 4. EdgeConv Block 4
        x4_edge = get_graph_feature(x3, k=self.k)           # [B, 256, N, k]
        x4 = self.conv4(x4_edge)                            # [B, 256, N, k]
        x4 = x4.max(dim=-1, keepdim=False)[0]               # [B, 256, N]

        # 5. Multi-scale concatenation & 1D aggregation
        x_multi = torch.cat([x1, x2, x3, x4], dim=1)        # [B, 512, N]
        x_fused = self.conv_fuse(x_multi)                   # [B, 1024, N]

        # 6. Global pooling (Max + Avg)
        x_max = torch.max(x_fused, dim=-1, keepdim=False)[0] # [B, 1024]
        x_avg = torch.mean(x_fused, dim=-1, keepdim=False)   # [B, 1024]
        x_global = torch.cat([x_max, x_avg], dim=1)         # [B, 2048]

        # 7. Bottleneck projection head
        zc = self.projection_head(x_global)                 # [B, output_size]

        return zc


if __name__ == '__main__':
    print("=" * 65)
    print("  DGCNNContextEncoder Self-Verification")
    print("=" * 65)

    model = DGCNNContextEncoder(output_size=128, k=20)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # Test [B, N, 3] input
    x1 = torch.randn(8, 1024, 3)
    out1 = model(x1)
    print(f"Input shape  : {x1.shape} -> Output shape: {out1.shape}")
    assert out1.shape == (8, 128), f"Expected (8, 128), got {out1.shape}"

    # Test [B, 3, N] input
    x2 = torch.randn(8, 3, 1024)
    out2 = model(x2)
    print(f"Input shape  : {x2.shape} -> Output shape: {out2.shape}")
    assert out2.shape == (8, 128), f"Expected (8, 128), got {out2.shape}"

    # Gradient flow check
    model.train()
    x3 = torch.randn(4, 1024, 3, requires_grad=True)
    out3 = model(x3)
    loss = out3.sum()
    loss.backward()
    assert x3.grad is not None, "Gradients did not flow back to input!"
    print("Gradient backward check: PASSED")
    print("All checks successfully passed!")
