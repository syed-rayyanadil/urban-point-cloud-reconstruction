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
    'dgcnn_channels'             : [64, 64, 128, 256],# Feature channels for 4 EdgeConv layers
    'dgcnn_pooling'              : 'dual_max_avg',   # 'dual_max_avg' (2048D) | 'max_only' (1024D)
    'ca_d_model'                 : 128,          # Cross-Attention token dimension
    'ca_num_heads'               : 4,            # Cross-Attention multi-head count
    'ca_num_queries'             : 1,            # Cross-Attention learnable query tokens count
    'ca_dim_feedforward'         : 512,          # Cross-Attention FFN hidden dimension
    'ca_dropout'                 : 0.0,          # Cross-Attention dropout probability
    'mamba_d_model'              : 128,          # Point Mamba token feature dimension
    'mamba_d_state'              : 16,           # Point Mamba state space expansion dimension
    'mamba_d_conv'               : 4,            # Point Mamba 1D depthwise conv kernel width
    'mamba_expand'               : 2,            # Point Mamba inner dimension expansion factor
    'mamba_num_layers'           : 1,            # Point Mamba stacked layer count (1 layer = fwd + bwd scans)
    'mamba_order'                : 'z_order',    # Point serialization: 'z_order' | 'coord_sort'
    'mamba_scan'                 : 'bidirectional', # Scanning direction: 'bidirectional' | 'unidirectional'
    'mamba_dropout'              : 0.0,          # Point Mamba dropout rate
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

    # --- DGCNN Ablation Family (Exp 4a - 4d) ---
    'exp4a_dgcnn_k10': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'dgcnn',
        'context_encoder_type': 'dgcnn',
        'dgcnn_k'             : 10,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp4a-DGCNN-k10',
    },
    'exp4b_dgcnn_k40': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'dgcnn',
        'context_encoder_type': 'dgcnn',
        'dgcnn_k'             : 40,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp4b-DGCNN-k40',
    },
    'exp4c_dgcnn_light_channels': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'dgcnn',
        'context_encoder_type': 'dgcnn',
        'dgcnn_k'             : 20,
        'dgcnn_channels'      : [32, 32, 64, 128],
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp4c-DGCNN-LightChannels',
    },
    'exp4d_dgcnn_max_only': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'dgcnn',
        'context_encoder_type': 'dgcnn',
        'dgcnn_k'             : 20,
        'dgcnn_pooling'       : 'max_only',
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp4d-DGCNN-MaxOnly',
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

    # --- Cross-Attention Transformer Ablation Family (Exp 5a - 5d) ---
    'exp5a_ca_dim64': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'cross_attention',
        'context_encoder_type': 'cross_attention',
        'ca_d_model'          : 64,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp5a-CA-Dim64',
    },
    'exp5b_ca_dim256': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'cross_attention',
        'context_encoder_type': 'cross_attention',
        'ca_d_model'          : 256,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp5b-CA-Dim256',
    },
    'exp5c_ca_heads2': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'cross_attention',
        'context_encoder_type': 'cross_attention',
        'ca_num_heads'        : 2,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp5c-CA-Heads2',
    },
    'exp5d_ca_heads8': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'cross_attention',
        'context_encoder_type': 'cross_attention',
        'ca_num_heads'        : 8,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp5d-CA-Heads8',
    },

    # 6. Experiment 6: Point Mamba Context Encoder (Phase 7 - State Space Model Ec)
    'exp6_mamba_context': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'bidirectional',
        'mamba_order'         : 'z_order',
        'mamba_d_model'       : 128,
        'mamba_d_state'       : 16,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6-PointMamba-Context',
    },

    # --- Point Mamba Ablation Family (Exp 6a - 6e) ---
    'exp6a_mamba_unidirectional': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'unidirectional',
        'mamba_order'         : 'z_order',
        'mamba_d_model'       : 128,
        'mamba_d_state'       : 16,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6a-Mamba-Unidirectional',
    },
    'exp6b_mamba_coord_sort': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'bidirectional',
        'mamba_order'         : 'coord_sort',
        'mamba_d_model'       : 128,
        'mamba_d_state'       : 16,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6b-Mamba-CoordSort',
    },
    'exp6c_mamba_dim256': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'bidirectional',
        'mamba_order'         : 'z_order',
        'mamba_d_model'       : 256,
        'mamba_d_state'       : 16,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6c-Mamba-Dim256',
    },
    'exp6d_mamba_dim64': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'bidirectional',
        'mamba_order'         : 'z_order',
        'mamba_d_model'       : 64,
        'mamba_d_state'       : 16,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6d-Mamba-Dim64',
    },
    'exp6e_mamba_state32': {
        **CONTEXT_HYPERPOCKET_BASE,
        'encoder_type'        : 'mamba',
        'context_encoder_type': 'mamba',
        'mamba_scan'          : 'bidirectional',
        'mamba_order'         : 'z_order',
        'mamba_d_model'       : 128,
        'mamba_d_state'       : 32,
        'loss_coef'           : 0.05,
        'kl_weight'           : 1.0,
        'kl_anneal'           : False,
        'wandb_run_name'      : 'CHP-SensatUrban-Exp6e-Mamba-State32',
    },
}
