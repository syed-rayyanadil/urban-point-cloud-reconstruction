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

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
EVAL_CONFIG = {
    'model_path' : '/kaggle/working/checkpoints/best_model.pth',
    'data_root'  : '/kaggle/input/datasets/syedrayyanadil/sensaturban-out/SensatUrban_Out',
    'n_points'   : 1024,
    'batch_size' : 8,          # Slightly larger batch size for faster evaluation
    'num_workers': 2,
    'k_variants' : 10,         # k=10 diverse completions per shape (Wu et al. 2020)
    'noise_sigma': 0.05,       # Latent noise scale (Appendix C)
    'save_path'  : '/kaggle/working/evaluation_results.json',
}


def run_evaluation(cfg=EVAL_CONFIG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 70)
    print('  HyperPocket VAE — Quantitative Evaluation Pipeline')
    print('=' * 70)
    print(f'  Device        : {device}')
    print(f'  Model Path    : {cfg["model_path"]}')
    print(f'  Data Root     : {cfg["data_root"]}')
    print(f'  k Variants    : {cfg["k_variants"]}')
    print(f'  Noise Sigma   : {cfg["noise_sigma"]} (per Appendix C)')
    print('=' * 70)

    # ---- 1. Load Model ----
    if not os.path.exists(cfg['model_path']):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {cfg['model_path']}")

    print("Loading model checkpoint...")
    checkpoint = torch.load(cfg['model_path'], map_location=device)
    
    # Reconstruct the model architecture using training configuration
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
        batch_size  = cfg['batch_size'],
        num_workers = cfg['num_workers'],
        as_tuple    = True,
    )
    print(f"Test dataset loaded with {len(test_loader.dataset)} blocks.")

    # ---- 3. Generative Inference Loop ----
    print(f"\nGenerating {cfg['k_variants']} completions per test block...")
    evaluator = PointCloudEvaluator(device=device, k=cfg['k_variants'], verbose=False)
    cd_loss_fn = _ChamferLoss().to(device)

    all_generations = []  # To store generated completions, shape [N_samples, k, 1024, 3]
    all_references = []   # To store ground-truth missing shapes, shape [N_samples, 1024, 3]

    recon_cd_list = []
    recon_emd_list = []
    tmd_list = []

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (existing, missing, gt, _) in enumerate(tqdm(test_loader)):
            # existing (Pe): [B, 1024, 3]
            # missing (Pm) : [B, 1024, 3] — ground-truth missing part
            B = existing.size(0)
            existing = existing.to(device)
            missing = missing.to(device)

            # Store references
            all_references.append(missing.cpu())

            # Generate k diverse completions for each sample in the batch
            batch_gens = []  # shape [k, B, 1024, 3]
            for j in range(cfg['k_variants']):
                # Sample latent noise: z_random ~ N(0, sigma^2)
                noise = torch.randn(B, train_cfg['random_encoder_output_size'], device=device) * cfg['noise_sigma']
                
                # Forward pass with custom noise override
                recon = model(existing, pm=None, epoch=0, device=device, noise=noise) # [B, 3, 1024]
                recon_xyz = recon.permute(0, 2, 1) # [B, 1024, 3]
                batch_gens.append(recon_xyz.cpu())

            # Stack along k dimension: [k, B, 1024, 3] -> transpose to [B, k, 1024, 3]
            batch_gens = torch.stack(batch_gens).transpose(0, 1)
            all_generations.append(batch_gens)

            # Compute Reconstruction CD, EMD, and TMD for each sample in the batch
            for idx in range(B):
                g_variants = batch_gens[idx].to(device) # [k, 1024, 3]
                ref_shape = missing[idx].unsqueeze(0)   # [1, 1024, 3]

                # 1. Reconstruction CD (average CD of all k variants to the GT missing shape)
                cd_val = 0.0
                for g in g_variants:
                    cd_val += cd_loss_fn(g.unsqueeze(0), ref_shape).item()
                recon_cd_list.append(cd_val / cfg['k_variants'])

                # 2. Reconstruction EMD (average EMD of all k variants to the GT missing shape)
                emd_val = 0.0
                for g in g_variants:
                    emd_batch = _sinkhorn_emd(g.unsqueeze(0), ref_shape)
                    emd_val += emd_batch.item()
                recon_emd_list.append(emd_val / cfg['k_variants'])

                # 3. TMD (average pairwise CD between the k variants for diversity)
                tmd_val = evaluator.compute_tmd(g_variants)
                tmd_list.append(tmd_val)

    generation_time = time.time() - start_time
    print(f"\nGenerative inference complete in {generation_time:.1f}s.")

    # Concat all samples
    all_generations = torch.cat(all_generations, dim=0) # [N_samples, k, 1024, 3]
    all_references = torch.cat(all_references, dim=0)   # [N_samples, 1024, 3]

    print("\nComputing dataset-level MMD and JSD metrics...")
    # Reshape all generations for global matching: [N_samples, k, 1024, 3] -> [N_samples * k, 1024, 3]
    flat_generations = all_generations.view(-1, cfg['n_points'], 3)

    # 4. MMD (Fidelity): set-to-set matching
    print("  -> Calculating MMD-CD...")
    mmd_cd = evaluator.compute_mmd(flat_generations, all_references)

    # 5. JSD (Distribution Similarity) over 28^3 grid
    print("  -> Calculating JSD...")
    jsd_val = evaluator.compute_jsd(flat_generations, all_references)

    # Mean over all samples
    mean_recon_cd = np.mean(recon_cd_list)
    mean_recon_emd = np.mean(recon_emd_list)
    mean_tmd = np.mean(tmd_list)

    # ---- 4. Save and Report Results ----
    results = {
        'Reconstruction_CD' : float(mean_recon_cd),
        'Reconstruction_EMD': float(mean_recon_emd),
        'MMD_CD'            : float(mmd_cd),
        'TMD_Diversity'     : float(mean_tmd),
        'JSD'               : float(jsd_val),
        'num_samples'       : len(recon_cd_list),
        'k_variants'        : cfg['k_variants'],
        'noise_sigma'       : cfg['noise_sigma']
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

    with open(cfg['save_path'], 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {cfg['save_path']}\n")

    return results


if __name__ == '__main__':
    run_evaluation()
