"""
context_hyperpocket_architecture.py — Context-Aware HyperPocket Configurations (Phase 4)

Contains the ContextHyperPocket architecture definition and experimental
variations (default paper settings with context, reduced beta, and beta-annealing).
"""

from .base_config import GLOBAL_BASE_CONFIG

# Base Model Architecture for Context-Aware HyperPocket
CONTEXT_HYPERPOCKET_BASE = {
    **GLOBAL_BASE_CONFIG,
    'model_name'                 : 'context_hyperpocket',
    'random_encoder_output_size' : 128,   # |zm| — Em latent dim (VAE)
    'real_encoder_output_size'   : 128,   # |ze| — Ee latent dim (deterministic)
    'context_encoder_output_size': 128,   # |zc| — Ec latent dim (deterministic)
    'latent_dim'                 : 384,   # concat(zm, ze, zc) = 128 + 128 + 128 = 384
    'use_bias'                   : True,
    'relu_slope'                 : 0.2,
    'target_network_layers'      : [32, 64, 128, 64],
    'use_context'                : True,
    'encoder_type'               : 'pointnet',   # 'pointnet' | 'dgcnn' | 'cross_attention' (Ec architecture)
    'context_encoder_type'       : 'pointnet',   # Alias / specific selector for Ec
    'dgcnn_k'                    : 20,           # k-NN neighbors for DGCNN Ec
    'dgcnn_dropout'              : 0.0,          # Dropout for DGCNN projection head
    'ca_d_model'                 : 128,          # Cross-Attention token dimension
    'ca_num_heads'               : 4,            # Cross-Attention multi-head count
    'ca_num_queries'             : 1,            # Cross-Attention learnable query tokens count
    'ca_dim_feedforward'         : 512,          # Cross-Attention FFN hidden dimension
    'ca_dropout'                 : 0.0,          # Cross-Attention dropout probability
    'neighbor_map_path'          : 'datasets/neighbor_map.json',
    'tile_centroids_path'        : 'datasets/tile_centroids.json',
}

# Experiments for Context-Aware HyperPocket Architecture
EXPERIMENTS = {
    # 1. Experiment 1: Context Default (Faithful HyperPocket loss parameters + Spatial Context)
    'exp1_context_default': {
        **CONTEXT_HYPERPOCKET_BASE,
        'loss_coef'     : 0.05,     # CD loss coefficient
        'kl_weight'     : 1.0,      # Weight for KL divergence term
        'kl_anneal'     : False,    # Constant KL weight
        'wandb_run_name': 'CHP-SensatUrban-Exp1-ContextDefault',
    },
    'default': {
        **CONTEXT_HYPERPOCKET_BASE,
        'loss_coef'     : 0.05,
        'kl_weight'     : 1.0,
        'kl_anneal'     : False,
        'wandb_run_name': 'CHP-SensatUrban-Exp1-ContextDefault',
    },

    # 2. Experiment 2: Context Reduced Beta (Loss Coef = 1.0, Beta = 0.001)
    'exp2_context_reduced_beta': {
        **CONTEXT_HYPERPOCKET_BASE,
        'loss_coef'     : 1.0,      # Full/Normalized Chamfer weight
        'kl_weight'     : 0.001,    # Scaled down beta to prevent posterior collapse
        'kl_anneal'     : False,
        'wandb_run_name': 'CHP-SensatUrban-Exp2-ContextReducedBeta-0.001',
    },

    # 3. Experiment 3: Context Beta-Annealing Schedule
    'exp3_context_annealing': {
        **CONTEXT_HYPERPOCKET_BASE,
        'loss_coef'        : 1.0,
        'kl_weight'        : 0.001,    # Target beta after warm-up
        'kl_anneal'        : True,     # Enable gradual beta warm-up
        'kl_anneal_warmup' : 15,       # Epochs 1-15: beta = 0
        'kl_anneal_end'    : 40,       # Epochs 16-40: beta 0 -> 0.001
        'wandb_run_name'   : 'CHP-SensatUrban-Exp3-ContextBetaAnnealing',
    },

    # 4. Experiment 4: Context DGCNN Encoder (Phase 5 - Dynamic Graph CNN Ec + Exp 1 Loss Parameters)
    'exp4_dgcnn_context': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'dgcnn',
        'context_encoder_type': 'dgcnn',
        'dgcnn_k'             : 20,
        'loss_coef'           : 0.05,     # CD loss coefficient (Faithful HyperPocket / Exp 1)
        'kl_weight'           : 1.0,      # Weight for KL divergence term
        'kl_anneal'           : False,    # Constant KL weight
        'wandb_run_name'      : 'CHP-SensatUrban-Exp4-DGCNN-Context',
    },

    # 5. Experiment 5: Context Cross-Attention Transformer (Phase 6 - Set Transformer Ec)
    'exp5_cross_attention_context': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'cross_attention',
        'context_encoder_type': 'cross_attention',
        'loss_coef'           : 0.05,     # Proven winning loss setup
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp5-CrossAttention-Context',
    },
}
