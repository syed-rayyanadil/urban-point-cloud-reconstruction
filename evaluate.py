"""
evaluate.py — Quantitative Evaluation for HyperPocket VAE Point Cloud Completion.

Loads a trained model checkpoint from the specified experiment directory,
generates k=10 diverse completion variants per test shape using random latent noise
(σ = 0.05, matching Wu et al., 2020 Appendix C), and computes the 5 evaluation metrics:
CD, EMD, MMD, TMD, and JSD.

Usage:
    # CLI mode (matches train.py):
    python evaluate.py --config base_hyperpocket.exp1_baseline_default
    python evaluate.py --config base_hyperpocket.exp2_reduced_beta
    python evaluate.py --config base_hyperpocket.exp3_annealing

    # In Kaggle Notebook:
    exec(open('evaluate.py').read())
"""

import os
import sys
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add repo path (for Kaggle: adjust if needed)
REPO_PATH = '/kaggle/working/urban-point-cloud-reconstruction'
if os.path.exists(REPO_PATH) and REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from datasets.sensat_dataset import get_dataloader, SensatUrbanDataset
from utils.sensat_metrics import PointCloudEvaluator, _ChamferLoss, _sinkhorn_emd
from models.base_hyperpocket import HyperPocketModel
from configs import get_config, list_available_configs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# DEFAULT EVALUATION CONFIGURATION
# ==========================================
DEFAULT_CONFIG = get_config('base_hyperpocket.default')


def _save_ply(filepath, points):
    """Save Nx3 numpy float array to binary little-endian PLY."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_pts = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_pts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        data = np.zeros(num_pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        data['x'], data['y'], data['z'] = points[:, 0], points[:, 1], points[:, 2]
        data.tofile(f)


def _plot_variants_figure(pe_np, pm_np, variants_np, save_png_path, sample_idx):
    """Plot multi-panel figure: Pe | Ground-Truth Pm | 4 Generated Variants."""
    k = min(4, variants_np.shape[0])
    fig = plt.figure(figsize=(4 * (k + 2), 4))
    fig.patch.set_facecolor('#1a1a2e')

    # Panel 1: Pe
    ax = fig.add_subplot(1, k + 2, 1, projection='3d')
    ax.set_facecolor('#0d0d1a')
    ax.scatter(pe_np[:, 0], pe_np[:, 1], pe_np[:, 2], c=pe_np[:, 2], cmap='winter', s=1.5)
    ax.set_title("Input Visible (Pe)", color='white', fontsize=9)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]); ax.axis('off')

    # Panel 2: GT Pm
    ax = fig.add_subplot(1, k + 2, 2, projection='3d')
    ax.set_facecolor('#0d0d1a')
    ax.scatter(pm_np[:, 0], pm_np[:, 1], pm_np[:, 2], c=pm_np[:, 2], cmap='plasma', s=1.5)
    ax.set_title("Ground-Truth (Pm)", color='white', fontsize=9)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]); ax.axis('off')

    # Panels 3..k+2: Generated Variants
    for vi in range(k):
        v = variants_np[vi]
        ax = fig.add_subplot(1, k + 2, vi + 3, projection='3d')
        ax.set_facecolor('#0d0d1a')
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=v[:, 2], cmap='autumn', s=1.5)
        ax.set_title(f"Gen Variant {vi+1}", color='white', fontsize=9)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1]); ax.axis('off')

    fig.suptitle(f'Sample {sample_idx} — Multimodal Completion Variants (k={variants_np.shape[0]})',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
    plt.savefig(save_png_path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)


def run_evaluation(config_input=None):
    """Run quantitative evaluation on the test dataset.

    Args:
        config_input (str | dict | None): Config key string (e.g. 'base_hyperpocket.default'),
                                          a loaded config dict, or None (uses default).
    """
    if isinstance(config_input, str):
        cfg = get_config(config_input)
    elif isinstance(config_input, dict):
        cfg = config_input.copy()
    else:
        cfg = DEFAULT_CONFIG.copy()

    # Automatically resolve checkpoint path from experiment directory if not explicitly given
    if not cfg.get('model_path'):
        cfg['model_path'] = os.path.join(cfg['save_dir'], 'best_model.pth')

    # Automatically resolve save path for evaluation_results.json
    if not cfg.get('save_path'):
        cfg['save_path'] = cfg.get('eval_save_path') or os.path.join(
            cfg.get('exp_dir', cfg['save_dir']), 'evaluation_results.json'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('  HyperPocket VAE — Quantitative Evaluation Pipeline')
    print('=' * 70)
    print(f'  Device        : {device}')
    print(f'  Model Path    : {cfg["model_path"]}')
    print(f'  Data Root     : {cfg["data_root"]}')
    print(f'  k Variants    : {cfg.get("k_variants", 10)}')
    print(f'  Noise Sigma   : {cfg.get("noise_sigma", 0.05)} (per Appendix C)')
    print(f'  Save Results  : {cfg["save_path"]}')
    print('=' * 70)

    # ---- 1. Load Model ----
    if not os.path.exists(cfg['model_path']):
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: {cfg['model_path']}\n"
            f"Please ensure training has completed for this experiment or pass --checkpoint /path/to/model.pth"
        )

    print("Loading model checkpoint...")
    checkpoint = torch.load(cfg['model_path'], map_location=device)

    train_cfg = checkpoint.get('config', cfg)
    model = HyperPocketModel(train_cfg).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Model loaded successfully!")

    # ---- 2. Load Dataset ----
    print("\nLoading test dataset...")
    test_loader = get_dataloader(
        split       = 'test',
        data_root   = cfg['data_root'],
        batch_size  = cfg.get('batch_size', 8),
        num_workers = cfg.get('num_workers', 2),
        as_tuple    = True,
    )
    print(f"Test dataset loaded with {len(test_loader.dataset)} blocks.")

    # ---- 3. Generative Inference Loop ----
    k_variants = cfg.get('k_variants', 10)
    noise_sigma = cfg.get('noise_sigma', 0.05)
    save_samples_count = cfg.get('save_samples', 5)

    print(f"\nGenerating {k_variants} completions per test block...")
    evaluator = PointCloudEvaluator(device=device, k=k_variants, verbose=False)
    cd_loss_fn = _ChamferLoss().to(device)

    all_generations = []
    all_references  = []
    saved_visuals   = []

    recon_cd_list  = []
    recon_emd_list = []
    tmd_list       = []

    start_time = time.time()
    sample_global_idx = 0
    eval_dir = os.path.dirname(cfg['save_path'])

    with torch.no_grad():
        for batch_idx, (existing, missing, gt, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
            B = existing.size(0)
            existing = existing.to(device)
            missing  = missing.to(device)

            all_references.append(missing.cpu())

            # Generate k diverse completions
            batch_gens = []
            for j in range(k_variants):
                noise = torch.randn(B, train_cfg['random_encoder_output_size'], device=device) * noise_sigma
                recon = model(existing, pm=None, epoch=0, device=device, noise=noise)
                recon_xyz = recon.permute(0, 2, 1)
                batch_gens.append(recon_xyz.cpu())

            batch_gens = torch.stack(batch_gens).transpose(0, 1) # [B, k, N, 3]
            all_generations.append(batch_gens)

            for idx in range(B):
                g_variants = batch_gens[idx].to(device) # [k, N, 3]
                ref_shape  = missing[idx].unsqueeze(0)  # [1, N, 3]

                # 1. CD
                cd_val = 0.0
                for g in g_variants:
                    cd_val += cd_loss_fn(g.unsqueeze(0), ref_shape).item()
                recon_cd_list.append(cd_val / k_variants)

                # 2. EMD
                emd_val = 0.0
                for g in g_variants:
                    emd_batch = _sinkhorn_emd(g.unsqueeze(0), ref_shape)
                    emd_val += emd_batch.item()
                recon_emd_list.append(emd_val / k_variants)

                # 3. TMD
                tmd_val = evaluator.compute_tmd(g_variants)
                tmd_list.append(tmd_val)

                # Export Top N Diverse 3D PLY & PNG Visuals
                if sample_global_idx < save_samples_count:
                    pe_np = existing[idx].cpu().numpy()
                    pm_np = missing[idx].cpu().numpy()
                    var_np = g_variants.cpu().numpy()

                    # Save PNG Visual
                    png_path = os.path.join(eval_dir, 'eval_samples', f'sample_{sample_global_idx:02d}_variants.png')
                    _plot_variants_figure(pe_np, pm_np, var_np, png_path, sample_global_idx)

                    # Save PLY Files for 3D inspection
                    ply_dir = os.path.join(eval_dir, 'eval_samples', f'sample_{sample_global_idx:02d}_ply')
                    _save_ply(os.path.join(ply_dir, 'pe_visible.ply'), pe_np)
                    _save_ply(os.path.join(ply_dir, 'pm_ground_truth.ply'), pm_np)
                    for vi in range(min(4, k_variants)):
                        _save_ply(os.path.join(ply_dir, f'variant_{vi+1}.ply'), var_np[vi])

                    saved_visuals.append(png_path)

                sample_global_idx += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    generation_time = time.time() - start_time
    print(f"\nGenerative inference complete in {generation_time:.1f}s.")

    # Concat all samples
    all_generations = torch.cat(all_generations, dim=0) # [N_samples, k, 1024, 3]
    all_references  = torch.cat(all_references, dim=0)   # [N_samples, 1024, 3]

    print("\nComputing dataset-level MMD and JSD metrics...")
    flat_generations = all_generations.view(-1, cfg.get('n_points', 1024), 3)

    # 4. MMD
    print("  -> Calculating MMD-CD...")
    mmd_cd = evaluator.compute_mmd(flat_generations, all_references)

    # 5. JSD
    print("  -> Calculating JSD...")
    jsd_val = evaluator.compute_jsd(flat_generations, all_references)

    # Results Dictionary
    results = {
        'Reconstruction_CD' : float(np.mean(recon_cd_list)),
        'Reconstruction_EMD': float(np.mean(recon_emd_list)),
        'MMD_CD'            : float(mmd_cd),
        'TMD_Diversity'     : float(np.mean(tmd_list)),
        'JSD'               : float(jsd_val),
        'num_samples'       : len(recon_cd_list),
        'k_variants'        : k_variants,
        'noise_sigma'       : noise_sigma,
        'best_val_epoch'    : checkpoint.get('epoch'),
        'best_val_loss'     : checkpoint.get('val_loss'),
    }

    print('\n' + '=' * 70)
    print('  EVALUATION SUMMARY RESULTS:')
    print('=' * 70)
    print(f'  Reconstruction CD  : {results["Reconstruction_CD"]:.6f}')
    print(f'  Reconstruction EMD : {results["Reconstruction_EMD"]:.6f}')
    print(f'  MMD (Fidelity)     : {results["MMD_CD"]:.6f}')
    print(f'  TMD (Diversity)    : {results["TMD_Diversity"]:.6f}')
    print(f'  JSD (Distribution) : {results["JSD"]:.6f}')
    print('=' * 70)

    # Save to JSON
    os.makedirs(os.path.dirname(cfg['save_path']), exist_ok=True)
    with open(cfg['save_path'], 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {cfg['save_path']}")

    # Optional WandB Summary logging
    try:
        import wandb
        if wandb.run is not None:
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    wandb.summary[f'test/{k}'] = v
            print("Logged test metrics to WandB summary.")
    except ImportError:
        pass

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Point Cloud Completion Models on SensatUrban')
    parser.add_argument(
        '--config',
        type=str,
        default='base_hyperpocket.default',
        help=f'Configuration key to evaluate. Available: {list_available_configs()}'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Optional explicit path to checkpoint .pth file (defaults to best_model.pth in experiment checkpoints folder)'
    )
    parser.add_argument(
        '--k_variants',
        type=int,
        default=10,
        help='Number of diverse completions per test block (default: 10)'
    )
    parser.add_argument(
        '--noise_sigma',
        type=float,
        default=0.05,
        help='Latent noise scale sigma (default: 0.05, matching Appendix C)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation inference (default: 8)'
    )
    args, unknown = parser.parse_known_args()

    selected_cfg = get_config(args.config)
    if args.checkpoint:
        selected_cfg['model_path'] = args.checkpoint
    selected_cfg['k_variants'] = args.k_variants
    selected_cfg['noise_sigma'] = args.noise_sigma
    selected_cfg['batch_size'] = args.batch_size

    print(f"\n[EVAL CONFIG] Loaded: '{args.config}'")
    print(f"  Target Checkpoint : {selected_cfg.get('model_path', os.path.join(selected_cfg['save_dir'], 'best_model.pth'))}")
    print(f"  Results JSON      : {selected_cfg.get('save_path', selected_cfg.get('eval_save_path'))}\n")

    run_evaluation(selected_cfg)
