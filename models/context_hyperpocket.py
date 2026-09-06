"""
context_hyperpocket.py — Context-Aware HyperPocket Generative Point Cloud Model (Phase 3)

Extends the baseline HyperPocket architecture (Wu et al., 2020) by integrating a
deterministic Context Encoder (Ec) for surrounding spatial neighborhood point clouds (Pc).

Architecture:
    - Real Encoder (Ee)     : PointNet encoder for visible context Pe -> ze [B, 128]
    - Random Encoder (Em)   : PointNet VAE encoder for missing target Pm -> zm [B, 128]
    - Context Encoder (Ec)  : PointNet encoder for spatial neighbors Pc -> zc [B, 128]
    - Context HyperNetwork  : Linear(384 -> 64 -> 128 -> 512 -> 1024 -> 2048) -> TargetNetwork weights
    - TargetNetwork         : Implicit decoder MLP (3 -> 32 -> 64 -> 128 -> 64 -> 3)
"""

import torch
import torch.nn as nn

try:
    from .base_hyperpocket import (
        Encoder,
        TargetNetwork,
        generate_random_points,
    )
except ImportError:
    from base_hyperpocket import (
        Encoder,
        TargetNetwork,
        generate_random_points,
    )


class ContextHyperNetwork(nn.Module):
    """Generates dynamically predicted weights for TargetNetwork from 384D conditioned latent code.

    Backbone:
        Linear: 384 -> 64 -> 128 -> 512 -> 1024 -> 2048 (with ReLU)
    Heads:
        Separate Linear layers for each TargetNetwork weight/bias tensor.
    """
    def __init__(self, input_size=384, target_layer_channels=None, use_bias=True):
        super().__init__()
        if target_layer_channels is None:
            target_layer_channels = [32, 64, 128, 64]

        self.use_bias = use_bias

        # TargetNetwork layer channels: [3, 32, 64, 128, 64, 3]
        layer_dims = [3] + target_layer_channels + [3]
        self.out_dims = [
            (layer_dims[i] + int(use_bias)) * layer_dims[i + 1]
            for i in range(len(layer_dims) - 1)
        ]

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 64),  nn.ReLU(inplace=True),
            nn.Linear(64, 128),         nn.ReLU(inplace=True),
            nn.Linear(128, 512),        nn.ReLU(inplace=True),
            nn.Linear(512, 1024),       nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
        )

        self.output_heads = nn.ModuleList([
            nn.Linear(2048, out_dim) for out_dim in self.out_dims
        ])

    def forward(self, latent):
        """
        Args:
            latent (Tensor): Conditioned latent vector [B, 384].
        Returns:
            Tensor: Concatenated TargetNetwork weights [B, total_weights].
        """
        feat = self.backbone(latent)
        return torch.cat([head(feat) for head in self.output_heads], dim=1)


class ContextHyperPocketModel(nn.Module):
    """Context-Aware HyperPocket Model integrating Spatial Context Encoding (Ec).

    Combines:
        - Real Encoder Ee (visible Pe -> ze, 128D)
        - Random Encoder Em (missing Pm -> zm, 128D, VAE)
        - Context Encoder Ec (spatial Pc -> zc, 128D, deterministic)
        - Context HyperNetwork (takes [zm, ze, zc] 384D -> TargetNetwork weights)
        - TargetNetwork (implicit decoder: unit-ball points -> reconstructed 3D shape)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg        = cfg
        self.rand_sz    = cfg.get('random_encoder_output_size', 128)
        self.real_sz    = cfg.get('real_encoder_output_size', 128)
        self.context_sz = cfg.get('context_encoder_output_size', 128)
        self.use_bias   = cfg.get('use_bias', True)
        self.tn_layers  = cfg.get('target_network_layers', [32, 64, 128, 64])
        self.n_points   = cfg.get('n_points', 1024)
        self.progressive_norm_epochs = cfg.get('progressive_norm_epochs', 100)
        self.use_context = cfg.get('use_context', True)

        # 1. Encoders
        # Ee — Real encoder for visible context Pe
        self.real_encoder = Encoder(self.real_sz, use_bias=self.use_bias, is_vae=False)
        # Em — VAE Random encoder for missing target Pm
        self.random_encoder = Encoder(self.rand_sz, use_bias=self.use_bias, is_vae=True)
        # Ec — Context encoder for spatial neighborhood Pc
        self.context_encoder = Encoder(self.context_sz, use_bias=self.use_bias, is_vae=False)

        # 2. Context HyperNetwork (384D input: 128 zm + 128 ze + 128 zc)
        total_latent_dim = self.rand_sz + self.real_sz + self.context_sz
        self.hyper_network = ContextHyperNetwork(
            input_size            = total_latent_dim,
            target_layer_channels = self.tn_layers,
            use_bias              = self.use_bias,
        )

    def forward(self, pe, pm=None, pc=None, epoch=0, device=None, noise=None, use_context=None):
        """
        Args:
            pe          (Tensor): Visible context [B, N, 3].
            pm          (Tensor): Missing target [B, N, 3] (None during evaluation with noise).
            pc          (Tensor): Spatial context neighborhood [B, N, 3] (optional).
            epoch       (int)   : Current training epoch (for progressive norm).
            device              : Torch device.
            noise       (Tensor): Pre-sampled noise [B, rand_sz] to override Em.
            use_context (bool)  : Override self.use_context (set False for zero-context ablation).

        Returns (training):
            reconstruction (Tensor): [B, 3, N] reconstructed missing point cloud.
            mu             (Tensor): [B, rand_sz] — mean of Em.
            logvar         (Tensor): [B, rand_sz] — log-variance of Em.

        Returns (evaluation):
            reconstruction (Tensor): [B, 3, N].
        """
        if device is None:
            device = pe.device

        B = pe.size(0)
        enable_context = self.use_context if use_context is None else use_context

        # 1. Em branch (Missing shape / VAE generative branch)
        if noise is None:
            if pm is None:
                raise ValueError("Either 'pm' or 'noise' must be provided to forward().")
            zm, mu, logvar = self.random_encoder(pm)   # [B, rand_sz]
        else:
            zm = noise                                 # [B, rand_sz]
            mu = logvar = None

        # 2. Ee branch (Visible context deterministic branch)
        ze = self.real_encoder(pe)                     # [B, real_sz]

        # 3. Ec branch (Spatial context neighborhood branch)
        if enable_context and pc is not None:
            zc = self.context_encoder(pc)              # [B, context_sz]
        else:
            # Zero-context vector (for Step 3.4 zero-context ablation)
            zc = torch.zeros(B, self.context_sz, device=device, dtype=ze.dtype)

        # 4. Concatenate: [zm (128), ze (128), zc (128)] -> z_cond [B, 384]
        z_cond = torch.cat([zm, ze, zc], dim=1)        # [B, 384]

        # 5. Generate TargetNetwork weights via ContextHyperNetwork
        tn_weights_batch = self.hyper_network(z_cond)  # [B, total_weight_size]

        # 6. Decode per sample in batch
        reconstruction = torch.zeros(B, 3, self.n_points, device=device)

        for j in range(B):
            tn = TargetNetwork(
                layer_channels = self.tn_layers,
                weights        = tn_weights_batch[j],
                use_bias       = self.use_bias,
            )
            random_pts = generate_random_points(
                self.n_points, epoch, self.progressive_norm_epochs, device
            )                                           # [N, 3]
            recon_j = tn(random_pts)                    # [N, 3]
            reconstruction[j] = recon_j.T               # [3, N]

        if self.training:
            return reconstruction, mu, logvar
        else:
            return reconstruction

