"""
context_sensat_dataset.py — Context-Aware SensatUrban Dataset & DataLoader (Phase 2)

Loads preprocessed .npy blocks along with their K=4 nearest spatial neighbor blocks,
aligns the neighbor point clouds in their true relative physical coordinates:
    P_neighbor_aligned = P_neighbor + (Centroid_neighbor - Centroid_target)
concatenates and resamples them to N=1024 points (Pc).

Data contract:
    - as_tuple=True : returns (Pe, Pm, Target, Pc)
        Pe     : FloatTensor [B, N, 3] (or [B, 3, N]) — visible context (input to encoder Ee)
        Pm     : FloatTensor [B, N, 3] (or [B, 3, N]) — missing target  (input to encoder Em)
        Target : FloatTensor [B, N, 3] (or [B, 3, N]) — full original block (reconstruction GT)
        Pc     : FloatTensor [B, N, 3] (or [B, 3, N]) — spatial context point cloud (input to encoder Ec)
    - as_tuple=False: returns dict with keys {"Pe", "Pm", "Target", "Pc"}
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Import base helpers from baseline dataset module
try:
    from .sensat_dataset import (
        SensatUrbanDataset,
        resample_pcd,
        hyperplane_cut,
        PREPROCESSED_DATA_DIR,
        N_POINTS,
        BATCH_SIZE,
        NUM_WORKERS,
        MIN_POINTS_SIDE,
        MAX_RETRY_CUT,
    )
except ImportError:
    from sensat_dataset import (
        SensatUrbanDataset,
        resample_pcd,
        hyperplane_cut,
        PREPROCESSED_DATA_DIR,
        N_POINTS,
        BATCH_SIZE,
        NUM_WORKERS,
        MIN_POINTS_SIDE,
        MAX_RETRY_CUT,
    )


def _resolve_file_path(candidates, description="file"):
    """Find the first existing path from candidate locations."""
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    return None


def _resolve_data_root(data_root):
    """Resolve data root directory across local and Kaggle paths."""
    candidates = [
        data_root,
        "SensatUrban_Out",
        "../SensatUrban_Out",
        "../../datasets/SensatUrban_Out",
        "datasets/SensatUrban_Out",
        "/kaggle/input/sensaturban-out/SensatUrban_Out",
        "/kaggle/input/datasets/syedrayyanadil/sensaturban-out/SensatUrban_Out",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return data_root


class ContextSensatUrbanDataset(Dataset):
    """Context-Aware PyTorch Dataset for SensatUrban .npy blocks with K=4 spatial neighbors."""

    def __init__(
        self,
        split='train',
        data_root=PREPROCESSED_DATA_DIR,
        n_points=N_POINTS,
        transpose=False,
        neighbor_map_path="datasets/neighbor_map.json",
        tile_centroids_path="datasets/tile_centroids.json",
    ):
        super().__init__()

        self.split       = split
        self.n_points    = n_points
        self.is_train    = (split == 'train')
        self.transpose   = transpose

        resolved_root = _resolve_data_root(data_root)
        split_dir     = os.path.join(resolved_root, split)

        if not os.path.exists(split_dir):
            raise FileNotFoundError(
                f"[ContextSensatUrbanDataset] Split directory not found: '{split_dir}'\n"
                f"Please verify preprocessed dataset location."
            )

        # Collect all valid .npy block files in this split
        self.file_paths = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith('.npy') and not f.startswith('._')
        ])

        if len(self.file_paths) == 0:
            raise RuntimeError(
                f"[ContextSensatUrbanDataset] No .npy files found in '{split_dir}'."
            )

        # Build fast index mapping filename -> absolute path across the entire data_root
        self.npy_path_map = {}
        for root, _, files in os.walk(resolved_root):
            for f in files:
                if f.endswith('.npy') and not f.startswith('._'):
                    self.npy_path_map[f] = os.path.join(root, f)

        # Load Spatial Context metadata
        resolved_nb_path = _resolve_file_path([
            neighbor_map_path,
            "datasets/neighbor_map.json",
            "neighbor_map.json",
            os.path.join(os.path.dirname(__file__), "neighbor_map.json"),
            os.path.join(os.path.dirname(__file__), "..", "datasets", "neighbor_map.json"),
        ], "neighbor_map.json")

        resolved_c_path = _resolve_file_path([
            tile_centroids_path,
            "datasets/tile_centroids.json",
            "tile_centroids.json",
            os.path.join(os.path.dirname(__file__), "tile_centroids.json"),
            os.path.join(os.path.dirname(__file__), "..", "datasets", "tile_centroids.json"),
        ], "tile_centroids.json")

        if not resolved_nb_path or not resolved_c_path:
            raise FileNotFoundError(
                f"[ContextSensatUrbanDataset] Spatial metadata files missing:\n"
                f"  neighbor_map   : {resolved_nb_path}\n"
                f"  tile_centroids : {resolved_c_path}"
            )

        with open(resolved_nb_path, 'r') as f:
            self.neighbor_map = json.load(f)

        with open(resolved_c_path, 'r') as f:
            self.tile_centroids = json.load(f)

        print(f"[ContextSensatUrbanDataset] '{split}' split — {len(self.file_paths)} blocks | "
              f"transpose={self.transpose} | context=True")

    def __len__(self):
        return len(self.file_paths)

    def _maybe_transpose(self, t):
        """Transpose [N, 3] -> [3, N] if self.transpose is True."""
        return t.permute(1, 0) if self.transpose else t

    def __getitem__(self, idx):
        """Load one block, apply hyperplane cut, resample, and load spatial context."""
        file_path = self.file_paths[idx]
        filename  = os.path.basename(file_path)

        # 1. Load target block [N, 3]
        points = np.load(file_path).astype(np.float32)
        target = resample_pcd(points, self.n_points)

        # 2. Hyperplane cut into visible (Pe) and missing (Pm)
        Pe_raw, Pm_raw = hyperplane_cut(
            target,
            min_points_side=MIN_POINTS_SIDE,
            max_retry=MAX_RETRY_CUT
        )
        Pe = resample_pcd(Pe_raw, self.n_points)
        Pm = resample_pcd(Pm_raw, self.n_points)

        # 3. Spatial Context Loading (Pc)
        target_centroid = self.tile_centroids.get(filename)
        neighbor_fns    = self.neighbor_map.get(filename, [])

        neighbor_point_clouds = []
        if target_centroid is not None and len(neighbor_fns) > 0:
            t_c = np.array(target_centroid, dtype=np.float32)

            for n_fn in neighbor_fns:
                n_path = self.npy_path_map.get(n_fn)
                if n_path and os.path.exists(n_path):
                    n_pts = np.load(n_path).astype(np.float32)
                    n_centroid = self.tile_centroids.get(n_fn)
                    if n_centroid is not None:
                        n_c = np.array(n_centroid, dtype=np.float32)
                        offset = n_c - t_c
                        n_aligned = n_pts + offset
                    else:
                        n_aligned = n_pts
                    neighbor_point_clouds.append(n_aligned)
                else:
                    # Safe fallback if neighbor file is missing
                    neighbor_point_clouds.append(Pe.copy())

        if len(neighbor_point_clouds) > 0:
            combined_neighbors = np.concatenate(neighbor_point_clouds, axis=0) # [4 * 1024, 3]
        else:
            combined_neighbors = Pe.copy()

        # Downsample combined neighbor cloud to exact N_POINTS (N_c=1024)
        Pc = resample_pcd(combined_neighbors, self.n_points)

        # Transpose tensors if requested
        Pe_t     = self._maybe_transpose(torch.from_numpy(Pe).float())
        Pm_t     = self._maybe_transpose(torch.from_numpy(Pm).float())
        target_t = self._maybe_transpose(torch.from_numpy(target).float())
        Pc_t     = self._maybe_transpose(torch.from_numpy(Pc).float())

        return {
            "Pe"    : Pe_t,      # [N, 3] or [3, N]
            "Pm"    : Pm_t,      # [N, 3] or [3, N]
            "Target": target_t,  # [N, 3] or [3, N]
            "Pc"    : Pc_t,      # [N, 3] or [3, N]
        }


# ==========================================
# DATALOADER COLLATORS
# ==========================================
def _collate_as_tuple(batch):
    """Collate into tuple: (Pe, Pm, Target, Pc)"""
    Pe     = torch.stack([b['Pe']     for b in batch])
    Pm     = torch.stack([b['Pm']     for b in batch])
    target = torch.stack([b['Target'] for b in batch])
    Pc     = torch.stack([b['Pc']     for b in batch])
    return Pe, Pm, target, Pc


def _collate_as_dict(batch):
    """Collate into dictionary {"Pe", "Pm", "Target", "Pc"}"""
    return {
        'Pe'    : torch.stack([b['Pe']     for b in batch]),
        'Pm'    : torch.stack([b['Pm']     for b in batch]),
        'Target': torch.stack([b['Target'] for b in batch]),
        'Pc'    : torch.stack([b['Pc']     for b in batch]),
    }


# ==========================================
# DATALOADER FACTORY
# ==========================================
def get_context_dataloader(
    split='train',
    data_root=PREPROCESSED_DATA_DIR,
    batch_size=BATCH_SIZE,
    n_points=N_POINTS,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    transpose=False,
    as_tuple=False,
    neighbor_map_path="datasets/neighbor_map.json",
    tile_centroids_path="datasets/tile_centroids.json",
):
    """Create and return a configured Context-Aware PyTorch DataLoader for SensatUrban blocks."""
    dataset = ContextSensatUrbanDataset(
        split               = split,
        data_root           = data_root,
        n_points            = n_points,
        transpose           = transpose,
        neighbor_map_path   = neighbor_map_path,
        tile_centroids_path = tile_centroids_path,
    )

    is_train = (split == 'train')
    collate_fn = _collate_as_tuple if as_tuple else _collate_as_dict

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = is_train,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = is_train,
        collate_fn  = collate_fn,
    )

    print(f"[ContextDataLoader] '{split}' — {len(dataset)} blocks | "
          f"batch_size={batch_size} | shuffle={is_train} | "
          f"transpose={transpose} | as_tuple={as_tuple}")

    return loader

