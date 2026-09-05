"""
base_config.py — Global Base Configuration

Contains project-wide default parameters shared across all point cloud
completion models (dataset paths, optimizer, progressive norm, WandB).
"""

import os

GLOBAL_BASE_CONFIG = {
    # Dataset & DataLoader
    'data_root'                 : '/kaggle/input/sensaturban-out/SensatUrban_Out',
    'n_points'                  : 1024,
    'batch_size'                : 5,
    'num_workers'               : 4,

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

    # Base experiment directories: experiments/<model_name>/<exp_name>/
    'experiments_base_dir'      : '/kaggle/working/experiments',
}


def get_experiment_dirs(model_name: str, exp_name: str, base_dir: str = '/kaggle/working/experiments') -> dict:
    """Return structured paths for an experiment: experiments/<model_name>/<exp_name>/"""
    exp_dir = os.path.join(base_dir, model_name, exp_name)
    return {
        'exp_dir'        : exp_dir,
        'save_dir'       : os.path.join(exp_dir, 'checkpoints'),
        'log_dir'        : os.path.join(exp_dir, 'logs'),
        'plot_dir'       : os.path.join(exp_dir, 'plots'),
        'eval_save_path' : os.path.join(exp_dir, 'evaluation_results.json'),
    }
