"""Test suite for Context-Aware SensatUrban dataset and dataloader.

Mirrors datasets/context_sensat_dataset.py.
"""
import os
import sys
import torch

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.context_sensat_dataset import get_context_dataloader


def test_context_sensat_dataset():
    print("\n" + "=" * 70)
    print("  Context-Aware SensatUrban Dataset & DataLoader Self-Test Suite")
    print("=" * 70)

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


    # 1. Context Dict format Test (batch_size=5)
    print("\n[Test 1] Context Dict format (batch_size=5):")
    loader_dict = get_context_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0, as_tuple=False)
    batch_dict  = next(iter(loader_dict))
    print(f"  Pe     : {batch_dict['Pe'].shape}      # Expected: [5, 1024, 3]")
    print(f"  Pm     : {batch_dict['Pm'].shape}      # Expected: [5, 1024, 3]")
    print(f"  Target : {batch_dict['Target'].shape}  # Expected: [5, 1024, 3]")
    print(f"  Pc     : {batch_dict['Pc'].shape}      # Expected: [5, 1024, 3]")

    assert batch_dict['Pe'].shape == (5, 1024, 3), f"Unexpected Pe shape: {batch_dict['Pe'].shape}"
    assert batch_dict['Pm'].shape == (5, 1024, 3), f"Unexpected Pm shape: {batch_dict['Pm'].shape}"
    assert batch_dict['Target'].shape == (5, 1024, 3), f"Unexpected Target shape: {batch_dict['Target'].shape}"
    assert batch_dict['Pc'].shape == (5, 1024, 3), f"Unexpected Pc shape: {batch_dict['Pc'].shape}"

    # 2. Context Tuple format Test (batch_size=5, as_tuple=True)
    print("\n[Test 2] Context Tuple format (batch_size=5, as_tuple=True):")
    loader_tuple = get_context_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0, as_tuple=True)
    Pe, Pm, Target, Pc = next(iter(loader_tuple))
    print(f"  Pe     : {Pe.shape}      # Expected: [5, 1024, 3]")
    print(f"  Pm     : {Pm.shape}      # Expected: [5, 1024, 3]")
    print(f"  Target : {Target.shape}  # Expected: [5, 1024, 3]")
    print(f"  Pc     : {Pc.shape}      # Expected: [5, 1024, 3]")

    assert Pe.shape == (5, 1024, 3), f"Unexpected Pe shape: {Pe.shape}"
    assert Pm.shape == (5, 1024, 3), f"Unexpected Pm shape: {Pm.shape}"
    assert Target.shape == (5, 1024, 3), f"Unexpected Target shape: {Target.shape}"
    assert Pc.shape == (5, 1024, 3), f"Unexpected Pc shape: {Pc.shape}"

    # 3. Transpose format Test (transpose=True, as_tuple=True)
    print("\n[Test 3] Context Transposed format (transpose=True, as_tuple=True):")
    loader_t = get_context_dataloader(data_root=data_root, split=split, batch_size=5, num_workers=0, transpose=True, as_tuple=True)
    Pe_t, Pm_t, Target_t, Pc_t = next(iter(loader_t))
    print(f"  Pe_t   : {Pe_t.shape}    # Expected: [5, 3, 1024]")
    print(f"  Pc_t   : {Pc_t.shape}    # Expected: [5, 3, 1024]")

    assert Pe_t.shape == (5, 3, 1024), f"Unexpected Pe_t shape: {Pe_t.shape}"
    assert Pc_t.shape == (5, 3, 1024), f"Unexpected Pc_t shape: {Pc_t.shape}"

    print("\n" + "=" * 70)
    print("✓ ALL CONTEXT DATALOADER TESTS PASSED! Pe, Pm, Target, Pc = [5, 1024, 3]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_context_sensat_dataset()
