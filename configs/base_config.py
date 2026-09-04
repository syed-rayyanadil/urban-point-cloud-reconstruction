"""
base_config.py — Global Base Configuration

Contains project-wide default parameters shared across all point cloud
completion models (dataset paths, optimizer, progressive norm, WandB).
"""

GLOBAL_BASE_CONFIG = {
    # Dataset & DataLoader
    'data_root'                 : '/kaggle/input/sensaturban-out/SensatUrban_Out',
    'n_points'                  : 1024,
    'batch_size'                : 5,
    'num_workers'               : 2,

    # Optimizer & Scheduler Defaults (Adam + StepLR per HyperPocket Appendix C)
    'epochs'                    : 200,
    'learning_rate'             : 1e-4,
    'adam_beta1'                : 0.9,
    'adam_beta2'                : 0.999,
    'scheduler_step_size'       : 41,
    'scheduler_gamma'           : 0.01,

    # Progressive Sphere Normalization
    'progressive_norm_epochs'   : 100,

    # Early Stopping
    'early_stopping_patience'   : 30,
    'early_stopping_min_epoch'  : 100,

    # Checkpointing & Visualizations
    'save_freq'                 : 10,
    'min_save_epoch'            : 10,

    # WandB
    'wandb_project'             : 'HyperPocket-SensatUrban',
    'wandb_run_name'            : None,
    'use_wandb'                 : True,

    # Output Paths
    'save_dir'                  : '/kaggle/working/checkpoints',
    'log_dir'                   : '/kaggle/working/logs',
    'plot_dir'                  : '/kaggle/working/plots',
}
