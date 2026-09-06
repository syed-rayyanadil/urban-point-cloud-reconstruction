"""Test suite for SensatUrban baseline dataset loader.

Mirrors datasets/sensat_dataset.py.
"""
import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.sensat_dataset import get_dataloader


def test_sensat_dataset():
    print("\n--- SensatUrban Dataset Quick Sanity Test ---\n")

    # Locate data root
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


    # --- Test 1: Default dict format [B, N, 3] ---
    print("[Test 1] Dict format, [B, N, 3]:")
    train_loader = get_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0)
    batch = next(iter(train_loader))
    print(f"  Pe     : {batch['Pe'].shape}    # Expected [5, 1024, 3]")
    print(f"  Pm     : {batch['Pm'].shape}    # Expected [5, 1024, 3]")
    print(f"  Target : {batch['Target'].shape}")
    print(f"  Pe range: [{batch['Pe'].min():.3f}, {batch['Pe'].max():.3f}]  (should be in [-1, 1])")
    print(f"  Dtype  : {batch['Pe'].dtype}")

    assert batch['Pe'].shape == (5, 1024, 3), f"Expected [5, 1024, 3], got {batch['Pe'].shape}"
    assert batch['Pm'].shape == (5, 1024, 3), f"Expected [5, 1024, 3], got {batch['Pm'].shape}"

    # --- Test 2: HyperPocket tuple format (existing, missing, gt, _) ---
    print("\n[Test 2] HyperPocket tuple format (existing, missing, gt, _):")
    loader_tuple = get_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0, as_tuple=True)
    existing, missing, gt, _ = next(iter(loader_tuple))
    print(f"  existing : {existing.shape}    # Expected [5, 1024, 3]")
    print(f"  missing  : {missing.shape}")
    print(f"  gt       : {gt.shape}")
    print(f"  _        : {_}               # None placeholder")

    assert existing.shape == (5, 1024, 3), f"Expected [5, 1024, 3], got {existing.shape}"

    # --- Test 3: Channel-first [B, 3, N] for 1D convolutions ---
    print("\n[Test 3] Transposed [B, 3, N] for 1D conv layers:")
    loader_t = get_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0, transpose=True, as_tuple=True)
    existing_t, missing_t, gt_t, _ = next(iter(loader_t))
    print(f"  existing : {existing_t.shape}    # Expected [5, 3, 1024]")
    print(f"  missing  : {missing_t.shape}")

    assert existing_t.shape == (5, 3, 1024), f"Expected [5, 3, 1024], got {existing_t.shape}"

    print("\n--- All tests complete ---")


if __name__ == "__main__":
    test_sensat_dataset()
