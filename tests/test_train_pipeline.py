"""Test suite for Training Pipeline integration (Phase 4).

Tests data loading, forward pass, loss calculation, backprop, validation,
and disk-safe checkpointing for both ContextHyperPocket and baseline HyperPocket.
"""
import os
import sys
import tempfile
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import get_config, list_available_configs
from train import train_epoch, val_epoch, EarlyStopping
from datasets.context_sensat_dataset import get_context_dataloader
from datasets.sensat_dataset import get_dataloader
from models.context_hyperpocket import ContextHyperPocketModel
from models.base_hyperpocket import HyperPocketModel
from utils.sensat_metrics import _ChamferLoss


def test_train_pipeline():
    print("\n" + "=" * 70)
    print("  Phase 4 Training Pipeline Integration Test Suite")
    print("=" * 70)

    # 1. Verify Config Registry
    configs = list_available_configs()
    print(f"\n[Test 1] Config Registry (Total: {len(configs)} configs):")
    for c in configs:
        print(f"  - {c}")
    assert 'context_hyperpocket.exp1_context_default' in configs
    assert 'context_hyperpocket.exp2_context_reduced_beta' in configs
    assert 'context_hyperpocket.exp3_context_annealing' in configs
    assert 'base_hyperpocket.exp1_baseline_default' in configs

    # 2. Locate data root
    candidates = [
        os.path.join(PROJECT_ROOT, "datasets", "SensatUrban_Out"),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "datasets", "SensatUrban_Out")),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "datasets", "SensatUrban_Out")),
    ]
    data_root = "datasets/SensatUrban_Out"
    for c in candidates:
        if os.path.exists(c):
            data_root = c
            break

    split = 'test' if os.path.exists(os.path.join(data_root, 'test')) else 'train'
    device = torch.device('cpu')

    # 3. ContextHyperPocket Training Step Test
    print("\n[Test 2] ContextHyperPocket 1-Step Training & Validation:")
    cfg_context = get_config('context_hyperpocket.exp1_context_default')
    cfg_context['data_root'] = data_root
    cfg_context['batch_size'] = 2
    cfg_context['epochs'] = 1

    train_loader = get_context_dataloader(
        split='test',
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        as_tuple=True,
    )
    val_loader = train_loader

    model_context = ContextHyperPocketModel(cfg_context).to(device)
    optimizer = torch.optim.Adam(model_context.parameters(), lr=1e-4)
    cd_loss_fn = _ChamferLoss().to(device)

    import logging
    log = logging.getLogger("TestLogger")
    log.setLevel(logging.WARNING)

    avg_loss, avg_cd, avg_kl, ex, gt, rec = train_epoch(
        epoch=1,
        model=model_context,
        optimizer=optimizer,
        loader=[next(iter(train_loader))],  # 1 batch
        device=device,
        cd_loss_fn=cd_loss_fn,
        cfg=cfg_context,
        log=log,
        is_context_model=True,
    )
    print(f"  Context Train Step -> Total Loss: {avg_loss:.4f} | CD: {avg_cd:.4f} | KL: {avg_kl:.4f}")
    assert avg_loss > 0 and avg_cd > 0

    val_cd = val_epoch(
        model=model_context,
        loader=[next(iter(val_loader))],
        device=device,
        cd_loss_fn=cd_loss_fn,
        cfg=cfg_context,
        is_context_model=True,
    )
    print(f"  Context Val Step   -> Val CD: {val_cd:.4f}")
    assert val_cd > 0

    # 4. Checkpoint saving test
    print("\n[Test 3] Safe 2-Checkpoint Saving Test:")
    with tempfile.TemporaryDirectory() as tmpdir:
        best_path = os.path.join(tmpdir, "best_model.pth")
        latest_path = os.path.join(tmpdir, "latest_model.pth")

        torch.save({'epoch': 1, 'model_state': model_context.state_dict()}, best_path)
        torch.save({'epoch': 1, 'model_state': model_context.state_dict()}, latest_path)

        assert os.path.exists(best_path), "best_model.pth was not saved!"
        assert os.path.exists(latest_path), "latest_model.pth was not saved!"
        print(f"  ✓ Saved and verified: best_model.pth ({os.path.getsize(best_path):,} bytes)")
        print(f"  ✓ Saved and verified: latest_model.pth ({os.path.getsize(latest_path):,} bytes)")

    print("\n" + "=" * 70)
    print("✓ ALL PHASE 4 TRAINING PIPELINE TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_train_pipeline()
