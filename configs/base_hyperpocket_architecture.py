"""
base_hyperpocket_architecture.py — HyperPocket Baseline Architecture Configurations

Contains the base HyperPocket architecture definition and experimental
variations (default paper settings, reduced beta, and beta-annealing).
"""

from .base_config import GLOBAL_BASE_CONFIG

# Base Model Architecture for HyperPocket
HYPERPOCKET_BASE = {
    **GLOBAL_BASE_CONFIG,
    'model_name'                 : 'base_hyperpocket',
    'random_encoder_output_size' : 128,   # |zm| — Em latent dim (VAE)
    'real_encoder_output_size'   : 128,   # |ze| — Ee latent dim (deterministic)
    'latent_dim'                 : 256,   # concat(zm, ze) = 128 + 128 = 256
    'use_bias'                   : True,
    'relu_slope'                 : 0.2,
    'target_network_layers'      : [32, 64, 128, 64],
}

# Experiments for Base HyperPocket Architecture
EXPERIMENTS = {
    # 1. Default Baseline (Faithful to HyperPocket Paper Appendix C)
    'default': {
        **HYPERPOCKET_BASE,
        'loss_coef'           : 0.05,     # CD loss coefficient
        'kl_weight'           : 1.0,      # Weight for KL divergence term
        'kl_anneal'           : False,    # Constant KL weight
        'wandb_run_name'      : 'HP-SensatUrban-Baseline-Default',
    },

    # 2. Experiment 1: Reduced Fixed KL Weight
    'exp1_reduced_beta': {
        **HYPERPOCKET_BASE,
        'loss_coef'           : 1.0,      # Normalized/Full Chamfer weight
        'kl_weight'           : 0.001,    # Scaled down beta to prevent posterior collapse
        'kl_anneal'           : False,
        'wandb_run_name'      : 'HP-SensatUrban-Exp1-ReducedBeta-0.001',
    },

    # 3. Experiment 2: Beta-Annealing Schedule
    'exp2_annealing': {
        **HYPERPOCKET_BASE,
        'loss_coef'           : 1.0,
        'kl_weight'           : 0.001,    # Target beta after warm-up
        'kl_anneal'           : True,     # Enable gradual beta warm-up
        'kl_anneal_warmup'    : 15,       # Epochs 1-15: beta = 0
        'kl_anneal_end'       : 40,       # Epochs 16-40: beta 0 -> 0.001
        'wandb_run_name'      : 'HP-SensatUrban-Exp2-BetaAnnealing',
    },
}
