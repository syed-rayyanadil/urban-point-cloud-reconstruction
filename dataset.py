"""
dataset.py — Data Loading & Hyperplane Partitioning (Pe/Pm) for SensatUrban.

Loads preprocessed .npy blocks from SensatUrban_Out/, splits each block into
a visible portion (Pe) and missing portion (Pm) via random hyperplane cutting,
resamples both partitions to a fixed size N=1024, and formats everything as
PyTorch tensors ready for the HyperPocket dual-encoder VAE.

Data contract — each batch returned by the DataLoader can be unpacked as:
    (existing, missing, gt, _)
    existing : FloatTensor [B, N, 3]  — Pe, visible context (input to encoder Ee)
    missing  : FloatTensor [B, N, 3]  — Pm, missing target  (input to encoder Em)
    gt       : FloatTensor [B, N, 3]  — full original block (reconstruction loss GT)
    _        : None placeholder       — reserved for compatibility with HyperPocket

All spatial coordinates are pre-normalised to [-1, 1]^3 (unit sphere).
Transposing to [B, 3, N] for 1D convolution layers is handled on demand
by passing transpose=True to get_dataloader().

Usage:
    from dataset import get_dataloader
    train_loader = get_dataloader(split='train')              # dict keys: Pe, Pm, Target
    train_loader = get_dataloader(split='train', as_tuple=True) # (existing, missing, gt, _)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CONFIGURATION
# ==========================================
PREPROCESSED_DATA_DIR = "SensatUrban_Out"  # READ-ONLY input: preprocessed .npy blocks
                                            # Written by preprocess.py, read here.
N_POINTS              = 1024               # Fixed point count: Pe, Pm, and Target
MIN_POINTS_SIDE       = 128               # Min points on each side of the hyperplane cut
MAX_RETRY_CUT         = 10               # Max random plane attempts before fallback
BATCH_SIZE            = 5                # Matches HyperPocket repository configuration
NUM_WORKERS           = 2               # CPU workers for DataLoader parallelism


# ==========================================
# HELPER: FIXED-SIZE RESAMPLING
# ==========================================
def resample_pcd(points, n_points=N_POINTS):
    """Resample a point cloud to exactly n_points.

    - If points > n_points : random downsample WITHOUT replacement.
    - If points < n_points : random upsample WITH replacement.
    - If points == n_points: return as-is (no copy overhead).

    Args:
        points   (np.ndarray): Input point cloud, shape [M, 3].
        n_points (int)       : Target fixed size.

    Returns:
        np.ndarray: Resampled point cloud, shape [n_points, 3].
    """
    current = len(points)
    if current == n_points:
        return points
    replace = current < n_points
    indices = np.random.choice(current, n_points, replace=replace)
    return points[indices]


# ==========================================
# HELPER: HYPERPLANE CUTTING (Pe vs Pm)
# ==========================================
def hyperplane_cut(points, min_points_side=MIN_POINTS_SIDE, max_retry=MAX_RETRY_CUT):
    """Divide a point cloud into Pe (visible) and Pm (missing) using a
    random 3D cutting plane passing through the block's centroid.

    The plane is defined by a random unit normal vector n and passes through
    the centroid of the point cloud:
        Pe = {p : dot(p - centroid, n) >= 0}
        Pm = {p : dot(p - centroid, n) <  0}

    A retry loop of up to max_retry attempts is used to ensure both sides
    contain at least min_points_side points. If no valid cut is found after
    max_retry attempts, a deterministic fallback split along the median X
    coordinate is applied.

    Args:
        points          (np.ndarray): Input point cloud, shape [N, 3].
        min_points_side (int)       : Minimum points required on each side.
        max_retry       (int)       : Maximum random plane attempts.

    Returns:
        Pe (np.ndarray): Visible portion, shape [M_e, 3].
        Pm (np.ndarray): Missing portion, shape [M_m, 3].
    """
    centroid = points.mean(axis=0)   # shape [3]

    for _ in range(max_retry):
        # Random unit normal vector for the cutting plane
        normal = np.random.randn(3).astype(np.float32)
        norm   = np.linalg.norm(normal)
        if norm < 1e-8:
            continue                  # Near-zero vector, skip this attempt
        normal /= norm

        # Signed distances from each point to the plane
        signed_dist = (points - centroid) @ normal   # shape [N]

        Pe = points[signed_dist >= 0]
        Pm = points[signed_dist <  0]

        # Enforce minimum point threshold on both sides
        if len(Pe) >= min_points_side and len(Pm) >= min_points_side:
            return Pe, Pm

    # Fallback: deterministic split along median X coordinate
    median_x = np.median(points[:, 0])
    Pe = points[points[:, 0] >= median_x]
    Pm = points[points[:, 0] <  median_x]

    # Edge-case safety: if median split also fails, split by index
    if len(Pe) == 0 or len(Pm) == 0:
        mid = len(points) // 2
        Pe  = points[:mid]
        Pm  = points[mid:]

    return Pe, Pm


# ==========================================
# CUSTOM PYTORCH DATASET
# ==========================================
class SensatUrbanDataset(Dataset):
    """PyTorch Dataset for preprocessed SensatUrban .npy blocks.

    Each .npy file contains a normalized point cloud of shape [1024, 3]
    (float32, already centered at origin and scaled to unit sphere [-1, 1]^3).
    Only pure 3D geometry (XYZ) is used — semantic labels are not loaded.

    For the training split:
        __getitem__ applies random hyperplane cutting to produce Pe and Pm,
        then resamples both back to N_POINTS.

    For the test split (no ground-truth labels):
        Pe and Pm are returned as safe zero-filled placeholder tensors of shape
        [N_POINTS, 3] to prevent DataLoader's collate_fn from crashing.

    Tensor layout: [N, 3] by default. Pass transpose=True to get [3, N]
    if downstream 1D convolution layers require channel-first format.
    """

    def __init__(self, split='train', data_root=PREPROCESSED_DATA_DIR,
                 n_points=N_POINTS, transpose=False):
        """
        Args:
            split     (str)  : 'train' or 'test'.
            data_root (str)  : Root directory containing 'train/' and 'test/' subdirs.
                               Reads from PREPROCESSED_DATA_DIR by default (SensatUrban_Out/).
            n_points  (int)  : Fixed point count for Pe, Pm, and Target tensors.
            transpose (bool) : If True, returns tensors as [3, N] instead of [N, 3].
                               Set True if downstream 1D convolutions require channel-first.
        """
        super().__init__()

        self.split     = split
        self.n_points  = n_points
        self.is_train  = (split == 'train')
        self.transpose = transpose

        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"[SensatUrbanDataset] Split directory not found: '{split_dir}'\n"
                f"Run preprocess.py first to generate the .npy block files."
            )

        # Collect all valid .npy block files (skip macOS hidden ._files)
        self.file_paths = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.npy') and not f.startswith('._')
        ])

        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"[SensatUrbanDataset] No .npy files found in '{split_dir}'.\n"
                f"Run preprocess.py first."
            )

        print(f"[SensatUrbanDataset] '{split}' split — {len(self.file_paths)} blocks "
              f"| transpose={self.transpose}")

    def __len__(self):
        return len(self.file_paths)

    def _maybe_transpose(self, t):
        """Transpose [N, 3] → [3, N] if self.transpose is True."""
        return t.permute(1, 0) if self.transpose else t

    def __getitem__(self, idx):
        """Load one block, apply hyperplane cut, resample, and return tensors.

        Returns:
            dict with keys:
                "Pe"    : FloatTensor [N, 3] or [3, N] — visible context partition.
                "Pm"    : FloatTensor [N, 3] or [3, N] — missing target partition.
                "Target": FloatTensor [N, 3] or [3, N] — full original block.

        Can also be unpacked in HyperPocket's format via get_dataloader(as_tuple=True):
            (existing, missing, gt, _)  →  (Pe, Pm, Target, None)
        """
        # Load preprocessed block — pure XYZ, no semantic labels, shape [1024, 3]
        points = np.load(self.file_paths[idx]).astype(np.float32)  # [N, 3]

        # Resample Target to guarantee exact N_POINTS (handles edge cases)
        target = resample_pcd(points, self.n_points)                # [N, 3]

        # Apply random 3D hyperplane cut through the block centroid.
        # Both train and test/validation splits are sliced dynamically.
        Pe_raw, Pm_raw = hyperplane_cut(
            target,
            min_points_side=MIN_POINTS_SIDE,
            max_retry=MAX_RETRY_CUT
        )
        # Resample both partitions independently back to N_POINTS
        Pe = resample_pcd(Pe_raw, self.n_points)   # [N, 3]
        Pm = resample_pcd(Pm_raw, self.n_points)   # [N, 3]

        # Build tensors — transpose to [3, N] if requested for 1D convolutions
        Pe_t     = self._maybe_transpose(torch.from_numpy(Pe).float())
        Pm_t     = self._maybe_transpose(torch.from_numpy(Pm).float())
        target_t = self._maybe_transpose(torch.from_numpy(target).float())

        return {
            "Pe"    : Pe_t,      # [N, 3] or [3, N]
            "Pm"    : Pm_t,      # [N, 3] or [3, N]
            "Target": target_t,  # [N, 3] or [3, N]
        }


# ==========================================
# DATALOADER FACTORY
# ==========================================
def _collate_as_tuple(batch):
    """Custom collate function to output batches as HyperPocket's expected tuple:
        (existing, missing, gt, _)
    where _ is a None placeholder for compatibility.
    """
    Pe     = torch.stack([b['Pe']     for b in batch])   # [B, N, 3] or [B, 3, N]
    Pm     = torch.stack([b['Pm']     for b in batch])   # [B, N, 3] or [B, 3, N]
    target = torch.stack([b['Target'] for b in batch])   # [B, N, 3] or [B, 3, N]
    return Pe, Pm, target, None


def get_dataloader(split='train', data_root=PREPROCESSED_DATA_DIR,
                   batch_size=BATCH_SIZE, n_points=N_POINTS,
                   num_workers=NUM_WORKERS, pin_memory=True,
                   transpose=False, as_tuple=False):
    """Create and return a configured PyTorch DataLoader for SensatUrban blocks.

    Args:
        split       (str)  : 'train' or 'test'.
        data_root   (str)  : Root directory of preprocessed .npy blocks
                             (defaults to PREPROCESSED_DATA_DIR = 'SensatUrban_Out/').
        batch_size  (int)  : Samples per batch. Default=5 matches HyperPocket config.
        n_points    (int)  : Fixed point count per sample (N=1024).
        num_workers (int)  : CPU worker processes for parallel data loading.
        pin_memory  (bool) : Pin CPU memory for fast CPU-to-GPU transfer.
        transpose   (bool) : If True, tensors are [B, 3, N] instead of [B, N, 3].
                             Set True if downstream 1D convolutions require channel-first.
        as_tuple    (bool) : If True, batches are returned as HyperPocket's tuple:
                             (existing, missing, gt, None) instead of a dict.

    Returns:
        DataLoader: Configured PyTorch DataLoader ready for training/evaluation.
    """
    dataset = SensatUrbanDataset(
        split     = split,
        data_root = data_root,
        n_points  = n_points,
        transpose = transpose,
    )

    is_train = (split == 'train')

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,             # Shuffle only for training
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = is_train,             # Drop incomplete final batch during training
        collate_fn  = _collate_as_tuple if as_tuple else None,
    )

    print(f"[DataLoader] '{split}' — {len(dataset)} blocks | "
          f"batch_size={batch_size} | shuffle={is_train} | "
          f"transpose={transpose} | as_tuple={as_tuple}")

    return loader


# ==========================================
# QUICK STANDALONE TEST
# ==========================================
if __name__ == "__main__":
    print("\n--- SensatUrban Dataset Quick Sanity Test ---\n")

    # --- Test 1: Default dict format [B, N, 3] ---
    print("[Test 1] Dict format, [B, N, 3]:")
    train_loader = get_dataloader(split='train')
    batch = next(iter(train_loader))
    print(f"  Pe     : {batch['Pe'].shape}    # Expected [5, 1024, 3]")
    print(f"  Pm     : {batch['Pm'].shape}    # Expected [5, 1024, 3]")
    print(f"  Target : {batch['Target'].shape}")
    print(f"  Pe range: [{batch['Pe'].min():.3f}, {batch['Pe'].max():.3f}]  (should be in [-1, 1])")
    print(f"  Dtype  : {batch['Pe'].dtype}")

    # --- Test 2: HyperPocket tuple format (existing, missing, gt, _) ---
    print("\n[Test 2] HyperPocket tuple format (existing, missing, gt, _):")
    loader_tuple = get_dataloader(split='train', as_tuple=True)
    existing, missing, gt, _ = next(iter(loader_tuple))
    print(f"  existing : {existing.shape}    # Expected [5, 1024, 3]")
    print(f"  missing  : {missing.shape}")
    print(f"  gt       : {gt.shape}")
    print(f"  _        : {_}               # None placeholder")

    # --- Test 3: Channel-first [B, 3, N] for 1D convolutions ---
    print("\n[Test 3] Transposed [B, 3, N] for 1D conv layers:")
    loader_t = get_dataloader(split='train', transpose=True, as_tuple=True)
    existing_t, missing_t, gt_t, _ = next(iter(loader_t))
    print(f"  existing : {existing_t.shape}    # Expected [5, 3, 1024]")
    print(f"  missing  : {missing_t.shape}")

    print("\n--- All tests complete ---")
