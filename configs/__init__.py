"""
configs/__init__.py — Central Configuration Registry

Allows loading any configuration by name across all model architectures:
e.g. get_config('base_hyperpocket.default')
     get_config('base_hyperpocket.exp1_baseline_default')
     get_config('base_hyperpocket.exp2_reduced_beta')
     get_config('base_hyperpocket.exp3_annealing')
"""

from .base_config import GLOBAL_BASE_CONFIG, get_experiment_dirs
from .base_hyperpocket_architecture import EXPERIMENTS as HYPERPOCKET_EXPERIMENTS

CONFIG_REGISTRY = {
    'base_hyperpocket': HYPERPOCKET_EXPERIMENTS,
}


def get_config(config_key: str = 'base_hyperpocket.default') -> dict:
    """Retrieve a configuration dictionary by dot-separated key (e.g. 'model.experiment').

    If only model name or experiment name is given, defaults intelligently.
    Automatically assigns output paths under experiments/<model_name>/<exp_name>/.
    """
    parts = config_key.split('.')
    if len(parts) == 2:
        model_name, exp_name = parts
    elif len(parts) == 1:
        if parts[0] in CONFIG_REGISTRY:
            model_name = parts[0]
            exp_name = 'default'
        elif parts[0] in HYPERPOCKET_EXPERIMENTS:
            model_name = 'base_hyperpocket'
            exp_name = parts[0]
        else:
            raise KeyError(f'Unknown config key: {config_key}. Available: {list_available_configs()}')
    else:
        raise ValueError(f'Invalid config key format: {config_key}. Use "model_name.exp_name"')

    if model_name not in CONFIG_REGISTRY:
        raise KeyError(f'Model "{model_name}" not in registry. Available models: {list(CONFIG_REGISTRY.keys())}')

    if exp_name not in CONFIG_REGISTRY[model_name]:
        raise KeyError(
            f'Experiment "{exp_name}" not found for {model_name}. '
            f'Available: {list(CONFIG_REGISTRY[model_name].keys())}'
        )

    cfg = CONFIG_REGISTRY[model_name][exp_name].copy()

    # Automatically set structured experiment directories: experiments/<model_name>/<exp_name>/
    exp_dirs = get_experiment_dirs(
        model_name = model_name,
        exp_name   = exp_name,
        base_dir   = cfg.get('experiments_base_dir', '/kaggle/working/experiments')
    )
    cfg.update(exp_dirs)
    cfg['model_name'] = model_name
    cfg['exp_name']   = exp_name

    return cfg


def list_available_configs() -> list:
    """List all registered config keys."""
    available = []
    for model_name, exps in CONFIG_REGISTRY.items():
        for exp_name in exps.keys():
            available.append(f'{model_name}.{exp_name}')
    return available
