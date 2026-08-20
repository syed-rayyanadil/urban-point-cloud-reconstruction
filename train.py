"""
train.py — HyperPocket VAE Training for SensatUrban Urban Point Cloud Completion.

Re-implements the HyperPocket architecture (Wu et al., 2020) in clean PyTorch
for training on the preprocessed SensatUrban urban point cloud dataset.

Architecture (faithful to original HyperPocket — Appendix C):
    - Real Encoder (Ee)     : Encodes visible Pe into a deterministic real_mu vector.
    - Random Encoder (Em)   : Encodes missing Pm into (z, mu, logvar) — VAE branch.
    - HyperNetwork          : Takes concat(z_random, real_mu) → generates weights
                              for the TargetNetwork.
    - TargetNetwork         : Layer sizes: 3, 32, 64, 128, 64, 3 (per Appendix C).

Loss Functions:
    - Reconstruction Loss   : Chamfer Distance (CD) between reconstructed & ground-truth Pm.
    - KL Divergence Loss    : Regularizes random encoder's latent distribution.

Paper Reference:
    Wu et al. (2020) "HyperPocket: Generative Point Cloud Completion"
    Appendix C — Implementation Details:
        - Target network: 3, 32, 64, 128, 64, 3
        - Adam optimizer: lr=0.0001, β1=0.9, β2=0.999
        - Progressive sphere normalization over 100 epochs
        - StepLR scheduler: step=41, γ=0.01
        - Latent vectors |ze| = |zm| = 128

Usage (Kaggle Notebook Cell):
    exec(open('train.py').read())
    # OR: %run train.py
"""

import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add repo path (for Kaggle: adjust if needed)
REPO_PATH = '/kaggle/working/urban-point-cloud-reconstruction'
if os.path.exists(REPO_PATH) and REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from sensat_dataset import get_dataloader, SensatUrbanDataset
from sensat_metrics import _ChamferLoss

# ==========================================
# Weights & Biases (WandB) Setup
# ==========================================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not installed. Run: pip install wandb")
    print("[WARNING] Training will continue without WandB logging.\n")

# ==========================================
# CONFIGURATION — Matches HyperPocket Paper Appendix C
# ==========================================
CONFIG = {
    # Dataset
    'data_root'           : '/kaggle/input/sensaturban-out/SensatUrban_Out',
    'n_points'            : 1024,
    'batch_size'          : 5,
    'num_workers'         : 2,

    # Model — Appendix C: |ze| = |zm| = 128
    'random_encoder_output_size' : 128,   # |zm| — Em latent dim (VAE)
    'real_encoder_output_size'   : 128,   # |ze| — Ee latent dim (deterministic)
    'latent_dim'                 : 256,   # concat(zm, ze) = 128 + 128 = 256
    'use_bias'                   : True,
    'relu_slope'                 : 0.2,
    # Appendix C: "target network architecture — fully connected network
    #              with following layer sizes: 3, 32, 64, 128, 64, 3"
    'target_network_layers'      : [32, 64, 128, 64],

    # Training — Appendix C: Adam lr=0.0001, β1=0.9, β2=0.999
    'epochs'              : 200,
    'learning_rate'       : 1e-4,
    'adam_beta1'          : 0.9,
    'adam_beta2'          : 0.999,
    'loss_coef'           : 0.05,     # CD loss coefficient (HyperPocket default)
    'kl_weight'           : 1.0,      # Weight for KL divergence term

    # Appendix C: StepLR scheduler (step=41, γ=0.01)
    'scheduler_step_size' : 41,
    'scheduler_gamma'     : 0.01,

    # Appendix C: "target network input normalization so that after 100 epochs,
    #              the target network input is sampled from a uniform unit 3D ball"
    'progressive_norm_epochs' : 100,

    # Early Stopping — do NOT trigger before epoch 100 (paper ran at least 100)
    'early_stopping_patience'   : 30,    # Stop if val loss doesn't improve for 30 epochs
    'early_stopping_min_epoch'  : 100,   # Never stop before epoch 100

    # Checkpointing
    'save_freq'           : 10,      # Save checkpoint every N epochs
    'min_save_epoch'      : 10,      # Start saving from this epoch

    # WandB
    'wandb_project'       : 'HyperPocket-SensatUrban',
    'wandb_run_name'      : None,    # Auto-generated if None
    'use_wandb'           : True,    # Set False to disable WandB

    # Paths
    'save_dir'            : '/kaggle/working/checkpoints',
    'log_dir'             : '/kaggle/working/logs',
    'plot_dir'            : '/kaggle/working/plots',
}


# ==========================================
# LOGGING SETUP
# ==========================================
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.log')
    logging.basicConfig(
        level    = logging.INFO,
        format   = '%(asctime)s | %(levelname)s | %(message)s',
        handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(), log_path


# ==========================================
# HYPERPOCKET ARCHITECTURE
# (Faithful reimplementation per Appendix C — no C++ extensions required)
# ==========================================

class Encoder(nn.Module):
    """Shared PointNet-style encoder used for both Ee (real) and Em (random/VAE).

    Copied architecture from HyperPocket's model/encoder.py:
        Conv1d: 3 → 64 → 128 → 256 → 512 → 512 (with ReLU)
        Max-pool to get global feature → fc(512 → 512) → mu, logvar heads

    For Ee (real encoder):  is_vae=False → returns deterministic mu only.
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

        self.fc       = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True))
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
        # Transpose [B, N, 3] → [B, 3, N] for Conv1d
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
        rand_sz  = cfg['random_encoder_output_size']
        real_sz  = cfg['real_encoder_output_size']
        use_bias = cfg['use_bias']
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

    def forward(self, pe, pm, epoch, device):
        """
        Args:
            pe     (Tensor): Visible context,  [B, N, 3].
            pm     (Tensor): Missing target,   [B, N, 3].
            epoch  (int)   : Current epoch number (for progressive normalization).
            device         : torch device.

        Returns (training):
            reconstruction (Tensor): [B, 3, N] reconstructed missing point cloud.
            mu    (Tensor): [B, rand_sz] — mean of Em's latent distribution.
            logvar(Tensor): [B, rand_sz] — log-variance of Em's latent distribution.

        Returns (evaluation):
            reconstruction (Tensor): [B, 3, N]
        """
        B = pe.size(0)

        # Em encodes Pm (missing target) — VAE branch (generativity)
        z_random, mu, logvar = self.random_encoder(pm)   # [B, rand_sz]

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


# ==========================================
# EARLY STOPPING
# ==========================================
class EarlyStopping:
    """Early stopping that respects a minimum epoch threshold.

    Appendix C states training runs for at least 100 epochs
    (progressive normalization completes at epoch 100).
    Early stopping will NOT trigger before min_epoch.
    """
    def __init__(self, patience=30, min_epoch=100):
        self.patience    = patience
        self.min_epoch   = min_epoch
        self.best_loss   = float('inf')
        self.counter     = 0
        self.best_epoch  = 0

    def step(self, val_loss, epoch):
        """Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            return False

        self.counter += 1

        # Never stop before min_epoch
        if epoch < self.min_epoch:
            return False

        return self.counter >= self.patience

    def status_str(self):
        return f'patience {self.counter}/{self.patience} | best @ epoch {self.best_epoch}'


# ==========================================
# VISUALIZATION UTILITIES
# ==========================================
def plot_loss_curves(train_losses, val_losses, save_dir, epoch):
    """Plot and save training & validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')

    epochs    = list(range(1, len(train_losses) + 1))
    labels    = ['Total Loss', 'Chamfer CD Loss', 'KL Divergence']
    colors    = ['#7c4dff', '#00e5ff', '#ff6d00']
    val_color = '#ff4081'

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax = axes[i]
        ax.set_facecolor('#0d0d1a')
        ax.plot(epochs, [l[i] for l in train_losses],
                color=color, linewidth=2, label=f'Train {label}')
        if val_losses and i == 0:
            val_ep = list(range(1, len(val_losses) + 1))
            ax.plot(val_ep, val_losses,
                    color=val_color, linewidth=2, linestyle='--', label='Val CD Loss')
        ax.set_title(label, color='white', fontsize=12)
        ax.set_xlabel('Epoch', color='#aaaaaa')
        ax.set_ylabel('Loss', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
        ax.grid(color='#333355', linestyle='--', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    fig.suptitle(f'HyperPocket VAE — Training Progress (Epoch {epoch})',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'loss_curves_epoch_{epoch:04d}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    return path


def plot_reconstruction_sample(existing_np, gt_np, recon_np, save_dir, epoch, sample_idx=0):
    """Plot 3-panel 3D scatter: Existing Pe | Ground-Truth Pm | Reconstructed Pm."""
    fig = plt.figure(figsize=(21, 6))
    fig.patch.set_facecolor('#1a1a2e')

    panels = [
        (existing_np, 'Existing (Pe)\n[Visible Context]',   'winter'),
        (gt_np,       'Ground Truth (Pm)\n[Missing Target]', 'plasma'),
        (recon_np,    'Reconstructed Pm\n[Model Output]',   'autumn'),
    ]

    for col, (pts, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        ax.set_facecolor('#0d0d1a')

        # Handle shape [3, N] → [N, 3]
        if pts.ndim == 2 and pts.shape[0] == 3:
            pts = pts.T

        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=pts[:, 2], cmap=cmap, s=1.5, alpha=0.85, depthshade=True)
        ax.set_title(title, color='white', fontsize=10, pad=8)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.tick_params(colors='#aaaaaa', labelsize=6)
        ax.set_xlabel('X', color='#aaaaaa', fontsize=7)
        ax.set_ylabel('Y', color='#aaaaaa', fontsize=7)
        ax.set_zlabel('Z', color='#aaaaaa', fontsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#333355')

    fig.suptitle(f'Reconstruction Sample — Epoch {epoch} | Sample {sample_idx}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, f'reconstruction_epoch_{epoch:04d}_sample_{sample_idx}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    return path


# ==========================================
# TRAINING EPOCH LOOP
# ==========================================
def train_epoch(epoch, model, optimizer, loader, device, cd_loss_fn, cfg, log):
    model.train()
    total_loss = total_cd = total_kl = 0.0
    n_batches  = len(loader)

    latest_existing = latest_gt = latest_recon = None

    for batch_idx, (existing, missing, gt, _) in enumerate(loader, 1):
        existing = existing.to(device)   # [B, N, 3]
        missing  = missing.to(device)    # [B, N, 3]
        gt       = gt.to(device)         # [B, N, 3]

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(existing, missing, epoch, device)
        # reconstruction: [B, 3, N] — permute to [B, N, 3] for Chamfer Loss
        recon_for_loss = reconstruction.permute(0, 2, 1)  # [B, N, 3]

        # 1. Chamfer Reconstruction Loss (loss_coef * CD matches HyperPocket default)
        loss_cd  = cfg['loss_coef'] * torch.mean(cd_loss_fn(gt, recon_for_loss))

        # 2. KL Divergence: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar) / B
        loss_kl  = cfg['kl_weight'] * 0.5 * (
            torch.exp(logvar) + torch.square(mu) - 1 - logvar
        ).sum() / existing.size(0)

        loss = loss_cd + loss_kl

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cd   += loss_cd.item()
        total_kl   += loss_kl.item()

        if batch_idx % max(1, n_batches // 5) == 0:
            log.info(f'  Epoch [{epoch}] Batch [{batch_idx}/{n_batches}] '
                     f'Loss={loss.item():.4f} CD={loss_cd.item():.4f} KL={loss_kl.item():.4f}')

        # Save last batch for visualization
        latest_existing = existing.detach().cpu().numpy()
        latest_gt       = gt.detach().cpu().numpy()
        latest_recon    = reconstruction.detach().cpu().numpy()

    avg_loss = total_loss / n_batches
    avg_cd   = total_cd   / n_batches
    avg_kl   = total_kl   / n_batches

    return avg_loss, avg_cd, avg_kl, latest_existing, latest_gt, latest_recon


# ==========================================
# VALIDATION EPOCH LOOP
# ==========================================
def val_epoch(model, loader, device, cd_loss_fn, cfg):
    model.eval()
    total_cd = 0.0

    with torch.no_grad():
        for batch_idx, (existing, missing, gt, _) in enumerate(loader, 1):
            existing = existing.to(device)
            missing  = missing.to(device)
            gt       = gt.to(device)

            reconstruction = model(existing, missing, 0, device)
            recon_for_loss = reconstruction.permute(0, 2, 1)

            loss_cd = cfg['loss_coef'] * torch.mean(cd_loss_fn(gt, recon_for_loss))
            total_cd += loss_cd.item()

    return total_cd / batch_idx


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def train(cfg=CONFIG):
    os.makedirs(cfg['save_dir'], exist_ok=True)
    os.makedirs(cfg['plot_dir'], exist_ok=True)

    log, log_path = setup_logging(cfg['log_dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'

    # ---- WandB Init ----
    use_wandb = cfg.get('use_wandb', True) and WANDB_AVAILABLE
    if use_wandb:
        run_name = cfg.get('wandb_run_name') or f'HP-SensatUrban-{datetime.now().strftime("%m%d-%H%M")}'
        wandb.init(
            project = cfg['wandb_project'],
            name    = run_name,
            config  = cfg,
        )
        log.info(f'  WandB         : ✓ Enabled — project={cfg["wandb_project"]}, run={run_name}')
    else:
        log.info(f'  WandB         : ✗ Disabled')

    log.info('=' * 70)
    log.info('  HyperPocket VAE — SensatUrban Urban Point Cloud Completion')
    log.info('  Configuration aligned with Wu et al. (2020) Appendix C')
    log.info('=' * 70)
    log.info(f'  Device         : {device} ({gpu_name})')
    log.info(f'  Data Root      : {cfg["data_root"]}')
    log.info(f'  Epochs         : {cfg["epochs"]}')
    log.info(f'  Batch Size     : {cfg["batch_size"]}')
    log.info(f'  Learning Rate  : {cfg["learning_rate"]} (β1={cfg["adam_beta1"]}, β2={cfg["adam_beta2"]})')
    log.info(f'  Loss Coef (CD) : {cfg["loss_coef"]}')
    log.info(f'  KL Weight      : {cfg["kl_weight"]}')
    log.info(f'  Latent Dim     : |zm|={cfg["random_encoder_output_size"]} + '
             f'|ze|={cfg["real_encoder_output_size"]} = {cfg["latent_dim"]}')
    log.info(f'  TargetNet Arch : 3 → {cfg["target_network_layers"]} → 3  (per Appendix C)')
    log.info(f'  Scheduler      : StepLR(step={cfg["scheduler_step_size"]}, γ={cfg["scheduler_gamma"]})')
    log.info(f'  Prog. Norm     : {cfg["progressive_norm_epochs"]} epochs')
    log.info(f'  Early Stopping : patience={cfg["early_stopping_patience"]}, '
             f'min_epoch={cfg["early_stopping_min_epoch"]}')
    log.info(f'  Log File       : {log_path}')
    log.info('=' * 70)

    # Save config to JSON for reproducibility
    with open(os.path.join(cfg['save_dir'], 'run_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    # ---- Dataset & DataLoaders ----
    train_loader = get_dataloader(
        split       = 'train',
        data_root   = cfg['data_root'],
        batch_size  = cfg['batch_size'],
        num_workers = cfg['num_workers'],
        as_tuple    = True,
    )

    # Val loader (test split)
    test_dataset = SensatUrbanDataset(
        split     = 'test',
        data_root = cfg['data_root'],
        n_points  = cfg['n_points'],
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size  = cfg['batch_size'],
        shuffle     = False,
        num_workers = cfg['num_workers'],
        pin_memory  = True,
        drop_last   = False,
        collate_fn  = lambda batch: (
            torch.stack([b['Pe']     for b in batch]),
            torch.stack([b['Pm']     for b in batch]),
            torch.stack([b['Target'] for b in batch]),
            None
        )
    )

    log.info(f'  Train batches  : {len(train_loader)}')
    log.info(f'  Val batches    : {len(val_loader)}')

    # ---- Model ----
    model_cfg = {**cfg, 'n_points': cfg['n_points']}
    model     = HyperPocketModel(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  Model Params   : {total_params:,}')

    if use_wandb:
        wandb.config.update({'total_params': total_params})

    # ---- Optimizer (Appendix C: Adam, lr=0.0001, β1=0.9, β2=0.999) ----
    optimizer = optim.Adam(
        model.parameters(),
        lr    = cfg['learning_rate'],
        betas = (cfg['adam_beta1'], cfg['adam_beta2']),
    )

    # ---- Scheduler (Appendix C: StepLR, step=41, γ=0.01) ----
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg['scheduler_step_size'],
        gamma     = cfg['scheduler_gamma'],
    )

    # ---- Loss Function ----
    cd_loss_fn = _ChamferLoss().to(device)

    # ---- Early Stopping ----
    early_stopper = EarlyStopping(
        patience  = cfg['early_stopping_patience'],
        min_epoch = cfg['early_stopping_min_epoch'],
    )

    # ---- Training History ----
    train_losses = []   # List of [total, cd, kl] per epoch
    val_losses   = []   # List of val_cd per epoch
    best_val_loss = float('inf')

    log.info('\n  Starting Training...\n')

    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()

        # --- Train ---
        avg_loss, avg_cd, avg_kl, ex_np, gt_np, rec_np = train_epoch(
            epoch, model, optimizer, train_loader, device, cd_loss_fn, cfg, log
        )
        train_losses.append([avg_loss, avg_cd, avg_kl])
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            f'Epoch [{epoch:03d}/{cfg["epochs"]}] '
            f'Train → Total: {avg_loss:.5f} | CD: {avg_cd:.5f} | KL: {avg_kl:.5f} | '
            f'LR: {current_lr:.2e} | '
            f'Time: {epoch_time:.1f}s'
        )

        # --- Validation ---
        val_cd = val_epoch(model, val_loader, device, cd_loss_fn, cfg)
        val_losses.append(val_cd)
        is_best = val_cd < best_val_loss

        if is_best:
            best_val_loss = val_cd
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'val_loss'   : val_cd,
                'config'     : cfg,
            }, os.path.join(cfg['save_dir'], 'best_model.pth'))

        log.info(
            f'Epoch [{epoch:03d}/{cfg["epochs"]}] '
            f'Val  → CD: {val_cd:.5f} | Best Val CD: {best_val_loss:.5f}'
            + (' ← NEW BEST ✓' if is_best else '')
            + f' | EarlyStop: {early_stopper.status_str()}'
        )

        # --- WandB Logging ---
        if use_wandb:
            wandb_log = {
                'epoch'          : epoch,
                'train/total_loss': avg_loss,
                'train/cd_loss'  : avg_cd,
                'train/kl_loss'  : avg_kl,
                'val/cd_loss'    : val_cd,
                'val/best_cd'    : best_val_loss,
                'learning_rate'  : current_lr,
                'early_stop/patience_counter': early_stopper.counter,
            }
            wandb.log(wandb_log, step=epoch)

        # --- Save Periodic Checkpoint ---
        if epoch >= cfg['min_save_epoch'] and epoch % cfg['save_freq'] == 0:
            ckpt_path = os.path.join(cfg['save_dir'], f'model_epoch_{epoch:04d}.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'val_loss'   : val_cd,
            }, ckpt_path)
            log.info(f'  Checkpoint saved: {ckpt_path}')

        # --- Save Visualizations ---
        if epoch % cfg['save_freq'] == 0 or epoch == 1:
            # Loss curves
            loss_plot = plot_loss_curves(train_losses, val_losses, cfg['plot_dir'], epoch)
            log.info(f'  Loss curve saved: {loss_plot}')

            # Reconstruction sample (first item in last batch)
            if ex_np is not None:
                recon_plot = plot_reconstruction_sample(
                    ex_np[0], gt_np[0], rec_np[0], cfg['plot_dir'], epoch, sample_idx=0
                )
                log.info(f'  Reconstruction plot saved: {recon_plot}')

            # Log images to WandB
            if use_wandb:
                wandb.log({
                    'plots/loss_curves': wandb.Image(loss_plot),
                    'plots/reconstruction': wandb.Image(recon_plot) if ex_np is not None else None,
                }, step=epoch)

        log.info('')  # Blank line between epochs for readability

        # --- Early Stopping Check ---
        should_stop = early_stopper.step(val_cd, epoch)
        if should_stop:
            log.info('=' * 70)
            log.info(f'  EARLY STOPPING triggered at epoch {epoch}!')
            log.info(f'  Val CD has not improved for {early_stopper.patience} epochs '
                     f'(best was at epoch {early_stopper.best_epoch}).')
            log.info('=' * 70)
            if use_wandb:
                wandb.log({'early_stopped_at_epoch': epoch}, step=epoch)
            break

    log.info('=' * 70)
    log.info(f'  Training Complete!')
    log.info(f'  Total Epochs Trained    : {epoch}')
    log.info(f'  Best Validation CD Loss : {best_val_loss:.5f} (epoch {early_stopper.best_epoch})')
    log.info(f'  Best model saved at     : {os.path.join(cfg["save_dir"], "best_model.pth")}')
    log.info(f'  Plots saved at          : {cfg["plot_dir"]}')
    log.info('=' * 70)

    # --- WandB Finish ---
    if use_wandb:
        wandb.finish()

    return model, train_losses, val_losses


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    train(CONFIG)
