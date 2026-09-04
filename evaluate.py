"""
evaluate.py — Quantitative Evaluation for HyperPocket VAE Point Cloud Completion.

Loads a trained model checkpoint, generates k=10 diverse completion variants
per test shape using random latent noise (σ = 0.05, matching Appendix C),
and computes the 5 evaluation metrics: CD, EMD, MMD, TMD, and JSD.

Usage (Kaggle Notebook Cell):
    exec(open('evaluate.py').read())
    # OR: %run evaluate.py
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

from sensat_dataset import get_dataloader, SensatUrbanDataset
from sensat_metrics import PointCloudEvaluator, _ChamferLoss, _sinkhorn_emd
from train import HyperPocketModel, CONFIG

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
EVAL_CONFIG = {
    'model_path'   : '/kaggle/working/checkpoints/best_model.pth',
    'data_root'    : '/kaggle/input/datasets/syedrayyanadil/sensaturban-out/SensatUrban_Out',
    'n_points'     : 1024,
    'batch_size'   : 8,          # Batch size for generative inference
    'num_workers'  : 2,
    'k_variants'   : 10,         # k=10 diverse completions per shape (Wu et al. 2020)
    'noise_sigma'  : 0.05,       # Latent noise scale (Appendix C)
    'save_path'    : '/kaggle/working/evaluation_results.json',
    'save_samples' : 5,          # Save top 5 diverse visual samples (.png and .ply)
    'log_to_wandb' : True,       # Log test metrics to wandb summary if active
}


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


def run_evaluation(cfg=EVAL_CONFIG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('  HyperPocket VAE — Quantitative Evaluation Pipeline')
    print('=' * 70)
    print(f'  Device        : {device}')
    print(f'  Model Path    : {cfg["model_path"]}')
    print(f'  Data Root     : {cfg["data_root"]}')
    print(f'  k Variants    : {cfg.get("k_variants", 10)}')
    print(f'  Noise Sigma   : {cfg.get("noise_sigma", 0.05)} (per Appendix C)')
    print('=' * 70)

    # ---- 1. Load Model ----
    if not os.path.exists(cfg['model_path']):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {cfg['model_path']}")

    print("Loading model checkpoint...")
    checkpoint = torch.load(cfg['model_path'], map_location=device)

    train_cfg = checkpoint.get('config', CONFIG)
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
    save_dir = os.path.dirname(cfg.get('save_path', '/kaggle/working/evaluation_results.json'))

    with torch.no_grad():
        for batch_idx, (existing, missing, gt, _) in enumerate(tqdm(test_loader)):
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
                    png_path = os.path.join(save_dir, 'visual_samples', f'sample_{sample_global_idx:02d}_variants.png')
                    _plot_variants_figure(pe_np, pm_np, var_np, png_path, sample_global_idx)

                    # Save PLY Files for 3D inspection
                    ply_dir = os.path.join(save_dir, 'visual_samples', f'sample_{sample_global_idx:02d}_ply')
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
    save_path = cfg.get('save_path', '/kaggle/working/evaluation_results.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {save_path}")

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
    run_evaluation()
