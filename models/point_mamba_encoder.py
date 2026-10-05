"""
point_mamba_encoder.py — Pure PyTorch Point Mamba Context Encoder (Ec) with Selective State Space Model

Drop-in replacement for the Context Encoder (Ec) in the 3D VAE completion pipeline.
Serializes irregular 3D point sets into structured 1D spatial sequences using Morton (Z-order)
or coordinate sorting, followed by bidirectional selective state space modeling (Mamba SSM)
and dual global feature aggregation (Max + Mean pooling) to produce a 128D context latent.

100% Pure PyTorch implementation with zero CUDA C++ extension dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Point Cloud Serialization Utilities (Morton Z-Order & Coordinate Sorting)
# ==============================================================================

def _expand_bits_3d(v: torch.Tensor) -> torch.Tensor:
    """Expands a 10-bit integer into a 30-bit integer by inserting 2 zeros between bits.

    Bit spread for 10-bit integer coordinates [0, 1023]:
        b9 b8 b7 b6 b5 b4 b3 b2 b1 b0 -> b9 0 0 b8 0 0 ... b1 0 0 b0
    """
    v = v.to(torch.int64)
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8))  & 0x0300F00F
    v = (v | (v << 4))  & 0x030C30C3
    v = (v | (v << 2))  & 0x09249249
    return v


def compute_morton_order(xyz: torch.Tensor) -> torch.Tensor:
    """Compute ascending Morton (Z-order curve) permutation indices for 3D points.

    Args:
        xyz (torch.Tensor): Coordinates tensor of shape [B, N, 3].

    Returns:
        torch.Tensor: Sorting indices of shape [B, N].
    """
    B, N, _ = xyz.shape
    device = xyz.device

    # 1. Normalize coordinates per-batch into unit cube [0.0, 1.0]
    min_val = xyz.amin(dim=1, keepdim=True)  # [B, 1, 3]
    max_val = xyz.amax(dim=1, keepdim=True)  # [B, 1, 3]
    span = (max_val - min_val).clamp(min=1e-6)
    normalized = (xyz - min_val) / span      # [B, N, 3] in [0, 1]

    # 2. Quantize to 10-bit integer coordinates [0, 1023]
    quantized = (normalized * 1023.0).clamp(0.0, 1023.0).to(torch.int64)
    x = quantized[..., 0]
    y = quantized[..., 1]
    z = quantized[..., 2]

    # 3. Interleave bits: code = expand(X) | (expand(Y) << 1) | (expand(Z) << 2)
    morton_codes = _expand_bits_3d(x) | (_expand_bits_3d(y) << 1) | (_expand_bits_3d(z) << 2)  # [B, N]

    # 4. Argsort to obtain sequence ordering
    sort_idx = torch.argsort(morton_codes, dim=1)  # [B, N]
    return sort_idx


def compute_coord_order(xyz: torch.Tensor) -> torch.Tensor:
    """Compute lexicographical coordinate sorting indices (X -> Y -> Z).

    Args:
        xyz (torch.Tensor): Coordinates tensor of shape [B, N, 3].

    Returns:
        torch.Tensor: Sorting indices of shape [B, N].
    """
    # Composite rank key: X * 1e6 + Y * 1e3 + Z
    min_val = xyz.amin(dim=1, keepdim=True)
    max_val = xyz.amax(dim=1, keepdim=True)
    span = (max_val - min_val).clamp(min=1e-6)
    norm_xyz = (xyz - min_val) / span  # [B, N, 3] in [0, 1]

    rank_keys = (
        norm_xyz[..., 0] * 1_000_000.0
        + norm_xyz[..., 1] * 1_000.0
        + norm_xyz[..., 2]
    )  # [B, N]

    sort_idx = torch.argsort(rank_keys, dim=1)  # [B, N]
    return sort_idx


def serialize_point_cloud(xyz: torch.Tensor, order: str = 'z_order') -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize unordered 3D point cloud into structured 1D sequence.

    Args:
        xyz (torch.Tensor): Point coordinates tensor of shape [B, N, 3].
        order (str): Serialization strategy ('z_order' or 'coord_sort').

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - sorted_xyz: Serialized coordinates [B, N, 3].
            - sort_idx: Sorting indices [B, N].
    """
    order = order.lower()
    if order in ['z_order', 'morton', 'z']:
        sort_idx = compute_morton_order(xyz)
    elif order in ['coord_sort', 'xyz_sort', 'lexicographical', 'coord']:
        sort_idx = compute_coord_order(xyz)
    else:
        raise ValueError(
            f"Unsupported point serialization order: '{order}'. "
            f"Supported options: 'z_order', 'coord_sort'."
        )

    # Gather points into sorted order
    sorted_xyz = torch.gather(xyz, dim=1, index=sort_idx.unsqueeze(-1).expand(-1, -1, 3))
    return sorted_xyz, sort_idx


# ==============================================================================
# 2. Pure PyTorch Bidirectional Selective State Space Model (BiMambaWrapper)
# ==============================================================================

class PureSelectiveSSM(nn.Module):
    """Pure PyTorch Selective State Space Model (Mamba SSM core).

    Implements continuous-to-discrete Zero-Order Hold (ZOH) selective state space
    scanning with input-dependent parameters (dt, B, C) and 1D depthwise convolution.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        # 1. In-projection for SSM input branch and multiplicative gate branch
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)

        # 2. 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # 3. Input-dependent parameter projections (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for stable continuous timescales
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 4. Learnable state transition matrix A (initialized via HiPPO/log scale)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # 5. Skip connection parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 6. Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for selective state-space scan.

        Args:
            x (torch.Tensor): Input token sequence of shape [B, L, d_model].

        Returns:
            torch.Tensor: Processed sequence of shape [B, L, d_model].
        """
        B, L, _ = x.shape

        # 1. Project to inner feature branches: [B, L, 2 * d_inner]
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)  # [B, L, d_inner], [B, L, d_inner]

        # 2. 1D Depthwise Convolution on x_branch
        x_conv = self.conv1d(x_branch.transpose(1, 2))[..., :L].transpose(1, 2)  # [B, L, d_inner]
        x_act = F.silu(x_conv)

        # 3. Project input to selective state parameters: dt, B, C
        x_dbl = self.x_proj(x_act)  # [B, L, dt_rank + 2 * d_state]
        dt_raw = x_dbl[..., :self.dt_rank]
        B_ssm = x_dbl[..., self.dt_rank:self.dt_rank + self.d_state]          # [B, L, d_state]
        C_ssm = x_dbl[..., self.dt_rank + self.d_state:]                     # [B, L, d_state]

        # Delta computation with softplus: [B, L, d_inner]
        dt = F.softplus(self.dt_proj(dt_raw))

        # Discretize continuous state matrix A: A = -exp(A_log)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # 4. Pure PyTorch Recurrent Scan across sequence length L
        # dA: [B, L, d_inner, d_state] = exp(dt * A)
        # dB_x: [B, L, d_inner, d_state] = dt * B * x
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_x = (dt.unsqueeze(-1) * B_ssm.unsqueeze(2)) * x_act.unsqueeze(-1)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        dA_list = dA.unbind(dim=1)
        dB_x_list = dB_x.unbind(dim=1)
        C_list = C_ssm.unbind(dim=1)

        # Fast unbind iteration
        for da_t, db_x_t, c_t in zip(dA_list, dB_x_list, C_list):
            h = da_t * h + db_x_t
            y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]

        # Add skip connection: D * x
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)

        # 5. Gated multiplicative activation & out-projection
        y = y * F.silu(z_branch)
        out = self.out_proj(y)  # [B, L, d_model]

        return out


class BiMambaWrapper(nn.Module):
    """Bidirectional / Unidirectional Mamba Block with Residual Normalization.

    Args:
        d_model (int): Hidden feature dimension (default: 128).
        d_state (int): State space expansion dimension (default: 16).
        d_conv (int): 1D depthwise convolution width (default: 4).
        expand (int): Inner dimension multiplier (default: 2).
        is_bidirectional (bool): Whether to perform forward and backward scans (default: True).
        dropout (float): Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        is_bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.is_bidirectional = is_bidirectional

        self.ssm_fwd = PureSelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        if self.is_bidirectional:
            self.ssm_bwd = PureSelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm_bwd = None

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for bidirectional/unidirectional SSM block.

        Args:
            x (torch.Tensor): Token sequence [B, L, d_model].

        Returns:
            torch.Tensor: Residual-normalized output [B, L, d_model].
        """
        # 1. Forward scan
        y_fwd = self.ssm_fwd(x)

        # 2. Backward scan (if bidirectional)
        if self.is_bidirectional:
            x_flipped = torch.flip(x, dims=[1])
            y_bwd_flipped = self.ssm_bwd(x_flipped)
            y_bwd = torch.flip(y_bwd_flipped, dims=[1])
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd

        # 3. Residual connection + LayerNorm
        out = self.norm(x + self.dropout(y))
        return out


# ==============================================================================
# 3. Point-Mamba Context Encoder (Ec) with Dual Global Pooling
# ==============================================================================

class PointMambaContextEncoder(nn.Module):
    """Point-Mamba Spatial Context Encoder (Ec) with Dual Global Pooling.

    Architecture:
        1. Coordinate Embedding MLP: [B, N, 3] -> Linear(3->64) -> LayerNorm -> LeakyReLU
                                              -> Linear(64->d_model) -> LayerNorm -> [B, N, d_model]
        2. Spatial Serialization: Morton Z-order curve or coordinate sorting -> ordered sequence
        3. Deep State-Space Stack: num_layers sequential BiMambaWrapper blocks
        4. Dual Global Pooling: Cat(Max-pool, Mean-pool) -> [B, 2 * d_model]
        5. Bottleneck Projection Head: Linear(2 * d_model -> 256) -> LayerNorm -> LeakyReLU
                                       -> Linear(256 -> output_size) -> [B, 128]

    Args:
        output_size (int): Output latent context vector dimension (default: 128).
        d_model (int): Hidden token feature dimension (default: 128).
        d_state (int): State space expansion dimension (default: 16).
        d_conv (int): 1D depthwise convolution width (default: 4).
        expand (int): Inner dimension multiplier (default: 2).
        num_layers (int): Number of stacked Mamba layers (default: 2).
        order (str): Point serialization strategy ('z_order' or 'coord_sort', default: 'z_order').
        scan_dir (str): Scan direction ('bidirectional' or 'unidirectional', default: 'bidirectional').
        dropout (float): Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        output_size: int = 128,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        order: str = 'z_order',
        scan_dir: str = 'bidirectional',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.order = order
        self.scan_dir = scan_dir.lower()
        self.is_bidirectional = (self.scan_dir in ['bidirectional', 'bidi', 'bi', '2way'])

        # 1. Coordinate Embedding MLP: [B, N, 3] -> [B, N, d_model]
        self.coord_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # 2. Stack of BiMambaWrapper blocks
        self.layers = nn.ModuleList([
            BiMambaWrapper(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                is_bidirectional=self.is_bidirectional,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 3. Bottleneck Projection Head (Dual Global Pooling: 2 * d_model -> output_size)
        self.proj_head = nn.Sequential(
            nn.Linear(2 * d_model, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(256, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Point Mamba Context Encoding.

        Args:
            x (torch.Tensor): Point cloud tensor of shape [B, 1024, 3] or [B, 3, 1024].

        Returns:
            torch.Tensor: Context latent embedding z_c of shape [B, output_size] (default: 128).

        Raises:
            ValueError: If input does not have exactly 3 coordinate channels.
        """
        # 1. Dimensionality check
        if x.dim() != 3:
            raise ValueError(
                f"PointMambaContextEncoder expects a 3D tensor of shape [B, 1024, 3] or [B, 3, 1024], "
                f"got tensor with {x.dim()} dimensions: {x.shape}"
            )

        # 2. Input Shape Guard & Dynamic Transposition: [B, 3, N] -> [B, N, 3]
        if x.size(1) == 3 and x.size(2) != 3:
            x = x.transpose(1, 2).contiguous()

        # Strict geometry-only channel validation
        if x.size(-1) != 3:
            raise ValueError(
                f"Geometry-Only Constraint Violated: PointMambaContextEncoder strictly accepts 3 "
                f"coordinate channels (X, Y, Z). Received input with {x.size(-1)} channels: {x.shape}."
            )

        # 3. Spatial Serialization (Morton Z-order curve or Coordinate sorting)
        sorted_xyz, _ = serialize_point_cloud(x, order=self.order)  # [B, N, 3]

        # 4. Coordinate Embedding: [B, N, 3] -> [B, N, d_model]
        tokens = self.coord_embed(sorted_xyz)

        # 5. Deep State-Space Stack
        for layer in self.layers:
            tokens = layer(tokens)  # [B, N, d_model]

        # 6. Dual Global Pooling Head (Max-pool + Mean-pool)
        x_max = torch.max(tokens, dim=1, keepdim=False)[0]   # [B, d_model]
        x_mean = torch.mean(tokens, dim=1, keepdim=False)     # [B, d_model]
        x_global = torch.cat([x_max, x_mean], dim=-1)         # [B, 2 * d_model]

        # 7. Bottleneck Projection -> [B, output_size]
        zc = self.proj_head(x_global)  # [B, 128]

        return zc


if __name__ == '__main__':
    print("=" * 65)
    print("  PointMambaContextEncoder Self-Verification")
    print("=" * 65)

    model = PointMambaContextEncoder(
        output_size=128,
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        order='z_order',
        scan_dir='bidirectional',
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    # 1. Test [B, 1024, 3] with B=4
    x1 = torch.randn(4, 1024, 3)
    out1 = model(x1)
    print(f"Input shape [4, 1024, 3] -> Output shape: {out1.shape}")
    assert out1.shape == (4, 128), f"Expected (4, 128), got {out1.shape}"

    # 2. Test [B, 3, 1024] channel-first format
    x2 = torch.randn(4, 3, 1024)
    out2 = model(x2)
    print(f"Input shape [4, 3, 1024] -> Output shape: {out2.shape}")
    assert out2.shape == (4, 128), f"Expected (4, 128), got {out2.shape}"

    # 3. Test B=1 invariance
    x_single = torch.randn(1, 1024, 3)
    out_single = model(x_single)
    print(f"Input shape [1, 1024, 3] -> Output shape: {out_single.shape}")
    assert out_single.shape == (1, 128), f"Expected (1, 128), got {out_single.shape}"

    # 4. Gradient backward check
    model.train()
    x3 = torch.randn(2, 1024, 3, requires_grad=True)
    out3 = model(x3)
    loss = out3.sum()
    loss.backward()
    assert x3.grad is not None, "Gradients did not flow back to input!"
    print("Gradient backward check: PASSED")
    print("All self-verification checks passed successfully!")
