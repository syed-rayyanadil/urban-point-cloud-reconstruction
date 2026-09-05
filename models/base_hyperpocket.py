"""
base_hyperpocket.py — HyperPocket Baseline Architecture

Faithful reimplementation of HyperPocket (Wu et al., 2020, Appendix C):
- Real Encoder (Ee)     : PointNet Conv1d encoder for visible context Pe.
- Random Encoder (Em)   : PointNet Conv1d VAE encoder for missing target Pm.
- HyperNetwork          : Predicts dynamically generated weights from latent code.
- TargetNetwork         : Implicit decoder MLP that transforms random unit-ball points into 3D shapes.
- generate_random_points: Samples random 3D points with progressive sphere normalization.
- HyperPocketModel      : Full model coordinating encoders, hypernetwork, and target network.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Shared PointNet-style encoder used for both Ee (real) and Em (random/VAE).

    Copied architecture from HyperPocket's model/encoder.py:
        Conv1d: 3 → 64 → 128 → 256 → 512 → 512 (with ReLU)
        Max-pool to get global feature → fc(512 → 512) → mu, logvar heads

    For Ee (real encoder):   is_vae=False → returns deterministic mu only.
    For Em (random encoder): is_vae=True  → returns (z, mu, logvar) via reparameterize.
    """
    def __init__(self, output_size, use_bias=True, relu_slope=0.2, is_vae=False):
        super().__init__()
        self.output_size = output_size
        self.is_vae      = is_vae

        # Conv1d backbone (expects [B, 3, N] — channel-first)
        self.conv = nn.Sequential(
            nn.Conv1d(3,   64,  1, bias=use_bias), nn.ReLU(inplace=True),
            nn.Conv1d(64,  128, 1, bias=use_bias), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=use_bias), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=use_bias), nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1, bias=use_bias),
        )

        self.fc        = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.mu_layer  = nn.Linear(512, output_size)
        self.std_layer = nn.Linear(512, output_size)  # Only used when is_vae=True

    def reparameterize(self, mu, logvar):
        """Sample z = mu + eps * exp(logvar) (reparameterization trick)."""
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Args:
            x (Tensor): Point cloud [B, N, 3] — auto-transposed to [B, 3, N] internally.
        Returns:
            (z, mu, exp(logvar)) if is_vae=True
            mu                   if is_vae=False
        """
        if x.size(-1) == 3:
            x = x.transpose(1, 2)   # [B, 3, N]

        feat        = self.conv(x)           # [B, 512, N]
        global_feat = feat.max(dim=2)[0]     # Global max-pool → [B, 512]
        logit       = self.fc(global_feat)   # [B, 512]
        mu          = self.mu_layer(logit)   # [B, output_size]

        if self.is_vae:
            logvar = self.std_layer(logit)   # [B, output_size]
            z      = self.reparameterize(mu, logvar)
            return z, mu, torch.exp(logvar)
        else:
            return mu


class HyperNetwork(nn.Module):
    """Takes the concatenated latent code → generates weights for TargetNetwork.

    Architecture from HyperPocket's model/hyper_network.py:
        Linear: input_size → 64 → 128 → 512 → 1024 → 2048 (with ReLU)
        Then separate Linear heads per TargetNetwork layer to produce weights.
    """
    def __init__(self, input_size, target_layer_channels, use_bias=True):
        super().__init__()
        self.use_bias = use_bias

        # Build full list of in/out channel pairs for TargetNetwork layers
        layer_dims    = [3] + target_layer_channels + [3]
        self.out_dims = [(layer_dims[i] + int(use_bias)) * layer_dims[i+1]
                         for i in range(len(layer_dims) - 1)]

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
            latent (Tensor): [B, input_size]
        Returns:
            Tensor: [B, sum(out_dims)] — concatenated weights for all TargetNetwork layers.
        """
        feat = self.backbone(latent)
        return torch.cat([head(feat) for head in self.output_heads], dim=1)


class TargetNetwork(nn.Module):
    """Implicit decoder — takes random 3D points and predicts XYZ coordinates.

    Faithfully re-implements HyperPocket's model/target_network.py.
    Weights are dynamically provided by the HyperNetwork for each sample.

    Appendix C layer sizes: 3, 32, 64, 128, 64, 3
    """
    def __init__(self, layer_channels, weights, use_bias=True):
        super().__init__()
        self.use_bias   = use_bias
        self.activation = nn.ReLU()

        layer_dims = [3] + layer_channels + [3]
        self.layers = []
        idx = 0

        for i in range(len(layer_dims) - 1):
            in_ch, out_ch = layer_dims[i], layer_dims[i + 1]
            w_size        = in_ch * out_ch
            layer         = {"weight": weights[idx: idx + w_size].view(out_ch, in_ch)}
            idx          += w_size
            if use_bias:
                layer["bias"] = weights[idx: idx + out_ch]
                idx += out_ch
            self.layers.append(layer)

    def forward(self, x):
        """
        Args:
            x (Tensor): Random 3D input points [N, 3].
        Returns:
            Tensor: Reconstructed point coordinates [N, 3].
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.mm(x, layer["weight"].T)
            if self.use_bias:
                x = x + layer["bias"]
            x = self.activation(x)

        # Final linear layer (no activation)
        out_layer = self.layers[-1]
        x = torch.mm(x, out_layer["weight"].T)
        if self.use_bias:
            x = x + out_layer["bias"]
        return x   # [N, 3]


def generate_random_points(n_points, epoch, progressive_norm_epochs, device):
    """Sample random 3D input points from a unit ball for the TargetNetwork.

    Faithful reimplementation of HyperPocket's utils/points.py with
    progressive normalization (Appendix C):
        "target network input normalization so that after 100 epochs,
         the target network input is sampled from a uniform unit 3D ball"

    Points inside the ball that are closer to the center than the current
    normalization coefficient are pushed outward to the sphere surface.
    This progressively constrains the input space over training.

    Args:
        n_points              (int): Number of points to generate.
        epoch                 (int): Current epoch (1-indexed).
        progressive_norm_epochs(int): Number of epochs over which to progressively
                                      normalize (100 per Appendix C).
        device                     : torch device.

    Returns:
        Tensor: [n_points, 3] unit-ball input points with progressive normalization.
    """
    # Sample uniform points from a unit 3D ball
    while True:
        pts = torch.zeros(n_points * 3, 3).uniform_(-1, 1)
        pts = pts[torch.norm(pts, dim=1) < 1]
        if pts.shape[0] >= n_points:
            pts = pts[:n_points]
            break

    # Progressive normalization (per Appendix C)
    if progressive_norm_epochs > 0:
        norm_coef = min(epoch / progressive_norm_epochs, 1.0)
        norms = torch.norm(pts, dim=1)
        mask = norms < norm_coef
        if mask.any():
            # Push interior points to the normalization boundary sphere
            pts[mask] = norm_coef * (pts[mask].T / norms[mask].clamp(min=1e-8)).T

    return pts.to(device)


class HyperPocketModel(nn.Module):
    """Full HyperPocket model combining both encoders, HyperNetwork, and TargetNetwork.

    HyperPocket mode (Appendix C):
        - Em (random encoder, is_vae=True)  encodes Pm (missing) → (z_random, mu, logvar)
        - Ee (real encoder,   is_vae=False) encodes Pe (existing) → real_mu
        - latent = concat(z_random, real_mu)  [256-dim]
        - HyperNetwork(latent) → TargetNetwork weights
        - For each sample in batch: TargetNetwork(random_points) → reconstruction
    """
    def __init__(self, cfg):
        super().__init__()
        rand_sz   = cfg['random_encoder_output_size']
        real_sz   = cfg['real_encoder_output_size']
        use_bias  = cfg['use_bias']
        tn_layers = cfg['target_network_layers']

        # Ee — deterministic real encoder for visible Pe
        self.real_encoder   = Encoder(real_sz, use_bias=use_bias, is_vae=False)
        # Em — VAE random encoder for missing Pm
        self.random_encoder = Encoder(rand_sz, use_bias=use_bias, is_vae=True)
        # HyperNetwork — takes concat(z_random + real_mu) → TargetNetwork weights
        self.hyper_network  = HyperNetwork(
            input_size           = rand_sz + real_sz,
            target_layer_channels= tn_layers,
            use_bias             = use_bias,
        )

        self.tn_layers  = tn_layers
        self.use_bias   = use_bias
        self.n_points   = cfg['n_points']
        self.progressive_norm_epochs = cfg.get('progressive_norm_epochs', 100)

    def forward(self, pe, pm, epoch, device, noise=None):
        """
        Args:
            pe     (Tensor): Visible context,  [B, N, 3].
            pm     (Tensor): Missing target,   [B, N, 3].
            epoch  (int)   : Current epoch number (for progressive normalization).
            device         : torch device.
            noise  (Tensor): Optional pre-sampled noise [B, rand_sz] to override Em.

        Returns (training):
            reconstruction (Tensor): [B, 3, N] reconstructed missing point cloud.
            mu    (Tensor): [B, rand_sz] — mean of Em's latent distribution.
            logvar(Tensor): [B, rand_sz] — log-variance of Em's latent distribution.

        Returns (evaluation):
            reconstruction (Tensor): [B, 3, N]
        """
        B = pe.size(0)

        if noise is None:
            # Em encodes Pm (missing target) — VAE branch (generativity)
            z_random, mu, logvar = self.random_encoder(pm)   # [B, rand_sz]
        else:
            # Override random encoder output with pre-sampled noise
            z_random = noise
            mu = logvar = None

        # Ee encodes Pe (visible context) — deterministic branch
        real_mu = self.real_encoder(pe)                   # [B, real_sz]

        # Concatenate latent codes → HyperPocket latent vector
        latent = torch.cat([z_random, real_mu], dim=1)   # [B, rand_sz + real_sz]

        # HyperNetwork generates TargetNetwork weights per sample in batch
        tn_weights_batch = self.hyper_network(latent)     # [B, total_weight_size]

        # For each sample: create its own TargetNetwork and decode random points → 3D shape
        reconstruction = torch.zeros(B, 3, self.n_points).to(device)

        for j in range(B):
            tn = TargetNetwork(
                layer_channels = self.tn_layers,
                weights        = tn_weights_batch[j],
                use_bias       = self.use_bias,
            )
            # Progressive normalization applied to target network input (Appendix C)
            random_pts        = generate_random_points(
                self.n_points, epoch, self.progressive_norm_epochs, device
            )                                                       # [N, 3]
            recon_j           = tn(random_pts)                      # [N, 3]
            reconstruction[j] = recon_j.T                           # [3, N]

        if self.training:
            return reconstruction, mu, logvar
        else:
            return reconstruction
