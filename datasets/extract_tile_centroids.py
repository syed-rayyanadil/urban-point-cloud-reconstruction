"""
extract_tile_centroids.py — One-off script to extract real-world tile centroids.

Reads .ply point clouds from SensatUrban splits ('train' and 'test'),
applies the exact 20cm voxel downsampling and 30m spatial block slicing
from preprocess.py, and computes the un-normalized real-world centroid [X, Y, Z]
in meters for every valid block.

Output:
    Saves 'tile_centroids.json' mapping block filenames to [X, Y, Z] coordinates:
    {
        "birmingham_block_0000.npy": [X, Y, Z],
        "birmingham_block_0001.npy": [X, Y, Z],
        ...
    }

Usage:
    python extract_tile_centroids.py
    # OR with explicit data directory:
    python extract_tile_centroids.py --data_dir SensatUrban_Out --out tile_centroids.json
"""

import os
import sys
import gc
import json
import argparse
import numpy as np

# ==========================================
# DEFAULT PREPROCESSING PARAMETERS
# (Identical to preprocess.py)
# ==========================================
GRID_SIZE        = 0.20     # 20cm voxel grid subsampling
BLOCK_SIZE       = 30.0     # 30m x 30m spatial XY partitioning
MIN_POINTS_BLOCK = 512      # Minimum points required to keep a block


# ==========================================
# PLY I/O (Pure NumPy)
# ==========================================
def read_ply_minimal(filepath):
    """Reads a binary .ply file. Dynamically handles missing 'class' columns for test files."""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header.append(line)
            if line == 'end_header':
                break

        dtype_list = []
        num_points = 0
        has_class  = False

        for line in header:
            if line.startswith('element vertex'):
                num_points = int(line.split()[-1])
            elif line.startswith('property'):
                parts     = line.split()
                ply_type  = parts[1]
                prop_name = parts[2]

                if prop_name == 'class':
                    has_class = True

                np_type = 'f4' if ply_type in ['float', 'float32'] else \
                          'u1' if ply_type in ['uchar', 'uint8']   else 'i4'
                dtype_list.append((prop_name, np_type))

        data = np.fromfile(f, dtype=np.dtype(dtype_list), count=num_points)
        xyz = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)

        if has_class:
            labels = data['class'].astype(np.uint8)
        else:
            labels = np.zeros(num_points, dtype=np.uint8)

        return xyz, labels, has_class


# ==========================================
# SUBSAMPLING LOGIC (Exact match to preprocess.py)
# ==========================================
def grid_subsample(points, labels, grid_size=0.2):
    """Memory-efficient grid subsampling for massive point clouds."""
    voxel_coords = np.floor(points / grid_size).astype(np.int32)
    coord_min = np.min(voxel_coords, axis=0)
    shifted   = voxel_coords - coord_min

    coord_max = np.max(shifted, axis=0).astype(np.uint64)
    stride_y  = coord_max[0] + 1
    stride_z  = stride_y * (coord_max[1] + 1)

    packed_coords = (shifted[:, 0].astype(np.uint64) +
                     shifted[:, 1].astype(np.uint64) * stride_y +
                     shifted[:, 2].astype(np.uint64) * stride_z)

    del voxel_coords, shifted

    unique_voxels, inverse_indices = np.unique(packed_coords, return_inverse=True)
    num_voxels = len(unique_voxels)
    del packed_coords

    sub_xyz = np.zeros((num_voxels, 3), dtype=np.float32)
    sub_xyz[:, 0] = np.bincount(inverse_indices, weights=points[:, 0])
    sub_xyz[:, 1] = np.bincount(inverse_indices, weights=points[:, 1])
    sub_xyz[:, 2] = np.bincount(inverse_indices, weights=points[:, 2])

    counts = np.bincount(inverse_indices, minlength=num_voxels)[:, None]
    sub_xyz /= counts

    num_classes = int(labels.max()) + 1
    label_votes = np.zeros((num_voxels, num_classes), dtype=np.int32)
    for cls in range(num_classes):
        is_cls = (labels == cls).astype(np.int32)
        label_votes[:, cls] = np.bincount(inverse_indices, weights=is_cls, minlength=num_voxels)
    sub_labels = np.argmax(label_votes, axis=1).astype(np.uint8)

    return sub_xyz, sub_labels


# ==========================================
# SPATIAL BLOCK PARTITIONING (Exact match to preprocess.py)
# ==========================================
def spatial_block_partition(xyz, labels, block_size=30.0, min_points=512):
    """Slide a regular block_size x block_size grid over the XY plane."""
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    x_starts = np.arange(x_min, x_max + block_size, block_size)
    y_starts = np.arange(y_min, y_max + block_size, block_size)

    valid_blocks = []

    for x0 in x_starts:
        x1 = x0 + block_size
        for y0 in y_starts:
            y1 = y0 + block_size

            mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            )

            block_xyz = xyz[mask]
            block_labels = labels[mask]

            if len(block_xyz) < min_points:
                continue

            valid_blocks.append((block_xyz, block_labels))

    return valid_blocks


# ==========================================
# CENTROID EXTRACTION PIPELINE
# ==========================================
def extract_centroids(data_dir='SensatUrban_Out', output_json='tile_centroids.json'):
    """Extract real-world centroids for each 30m block across train/test splits."""
    splits = ['train', 'test']
    tile_centroids = {}

    print('=' * 70)
    print('  SensatUrban Real-World Tile Centroid Extraction')
    print('=' * 70)
    print(f'  Dataset Path : {os.path.abspath(data_dir)}')
    print(f'  Output JSON  : {os.path.abspath(output_json)}')
    print(f'  Voxel Grid   : {GRID_SIZE} m')
    print(f'  Block Size   : {BLOCK_SIZE} m x {BLOCK_SIZE} m')
    print(f'  Min Points   : {MIN_POINTS_BLOCK}')
    print('=' * 70)

    total_blocks_found = 0

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"\n[INFO] Split directory not found, skipping: {split_dir}")
            continue

        ply_files = sorted([
            f for f in os.listdir(split_dir)
            if f.lower().endswith('.ply') and not f.startswith('._')
        ])

        if not ply_files:
            print(f"\n[INFO] No .ply files found in: {split_dir}")
            continue

        print(f"\n--- Processing '{split}' split ({len(ply_files)} PLY file(s)) ---")

        for file_idx, filename in enumerate(ply_files):
            input_path = os.path.join(split_dir, filename)
            stem = os.path.splitext(filename)[0]
            print(f"\n[{file_idx + 1}/{len(ply_files)}] Reading: {filename}...")

            # 1. Load raw PLY
            xyz, labels, _ = read_ply_minimal(input_path)
            print(f"  -> Loaded {len(xyz):,} raw points")

            # 2. Voxel Grid Subsampling (20cm)
            sub_xyz, sub_labels = grid_subsample(xyz, labels, grid_size=GRID_SIZE)
            print(f"  -> Subsampled to {len(sub_xyz):,} points")

            del xyz, labels
            gc.collect()

            # 3. 30m Spatial Block Slicing
            valid_blocks = spatial_block_partition(
                sub_xyz, sub_labels,
                block_size=BLOCK_SIZE,
                min_points=MIN_POINTS_BLOCK
            )
            print(f"  -> Sliced into {len(valid_blocks)} valid blocks (>= {MIN_POINTS_BLOCK} pts)")

            del sub_xyz, sub_labels
            gc.collect()

            # 4. Compute real-world centroid [X, Y, Z] before normalization
            for b_idx, (block_xyz, _) in enumerate(valid_blocks):
                # Real-world centroid in meters
                centroid = block_xyz.mean(axis=0)
                block_filename = f"{stem}_block_{b_idx:04d}.npy"

                tile_centroids[block_filename] = [
                    float(centroid[0]),
                    float(centroid[1]),
                    float(centroid[2]),
                ]
                total_blocks_found += 1

                del block_xyz
                gc.collect()

            print(f"  -> Recorded centroids for {len(valid_blocks)} blocks from {stem}")

    # 5. Save to JSON
    out_dir = os.path.dirname(output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(tile_centroids, f, indent=2)

    print('\n' + '=' * 70)
    print(f'  Extraction Complete!')
    print(f'  Total Centroids Extracted : {total_blocks_found}')
    print(f'  JSON Saved To             : {os.path.abspath(output_json)}')
    print('=' * 70)

    return tile_centroids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract real-world 3D block centroids from SensatUrban PLY files before normalization.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='SensatUrban_Out',
        help="Root folder containing 'train/' and 'test/' splits (default: 'SensatUrban_Out')"
    )
    parser.add_argument(
        '--out',
        type=str,
        default='tile_centroids.json',
        help="Path to save output JSON mapping (default: 'tile_centroids.json')"
    )
    args = parser.parse_args()

    # Fallback to local paths if default not found
    resolved_data_dir = args.data_dir
    if not os.path.exists(resolved_data_dir):
        alt_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'SensatUrban_Out'),
            os.path.join(os.path.dirname(__file__), '..', 'SensatUrban'),
            'SensatUrban',
            '../SensatUrban',
        ]
        for p in alt_paths:
            if os.path.exists(p):
                resolved_data_dir = p
                break

    extract_centroids(data_dir=resolved_data_dir, output_json=args.out)
