import os
import gc
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
BASE_INPUT_FOLDER  = "../SensatUrban"       # Relative path for portable hard drive
BASE_OUTPUT_FOLDER = "SensatUrban_Out"      # Output directory for processed .npy blocks
GRID_SIZE          = 0.20                   # 20cm voxel downsampling (SensatUrban standard)
BLOCK_SIZE         = 30.0                   # 30m x 30m spatial XY partitioning
MIN_POINTS_BLOCK   = 512                    # Drop blocks with fewer than this many points
N_POINTS           = 1024                   # Fixed point count per block (resampling target)

# ==========================================
# PLY I/O HELPER FUNCTIONS (Pure Numpy)
# ==========================================
def read_ply_minimal(filepath):
    """Reads a binary .ply file. Dynamically handles missing 'class' columns for test files."""
    with open(filepath, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
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
            # Test set has no ground-truth labels — fill with zeros
            labels = np.zeros(num_points, dtype=np.uint8)

        return xyz, labels, has_class


# ==========================================
# SUBSAMPLING LOGIC
# ==========================================
def grid_subsample(points, labels, grid_size=0.2):
    """Memory-efficient grid subsampling for massive point clouds.
    Works for ALL semantic classes (no class filtering applied here).
    """

    # 1. Discretize points into voxel coordinates (use int32 to save memory)
    voxel_coords = np.floor(points / grid_size).astype(np.int32)

    # 2. Shift coordinates to be strictly non-negative to allow 1D packing
    coord_min = np.min(voxel_coords, axis=0)
    shifted   = voxel_coords - coord_min

    # 3. Pack 3D coordinates into a single 1D uint64 array
    # This completely avoids the memory explosion of np.unique(axis=0)
    coord_max = np.max(shifted, axis=0).astype(np.uint64)
    stride_y  = coord_max[0] + 1
    stride_z  = stride_y * (coord_max[1] + 1)

    packed_coords = (shifted[:, 0].astype(np.uint64) +
                     shifted[:, 1].astype(np.uint64) * stride_y +
                     shifted[:, 2].astype(np.uint64) * stride_z)

    # Free memory immediately before the heavy unique operation
    del voxel_coords, shifted

    # 4. Use 1D unique (vastly more memory-efficient and much faster)
    unique_voxels, inverse_indices = np.unique(packed_coords, return_inverse=True)
    num_voxels = len(unique_voxels)
    del packed_coords   # Free packed array

    # 5. Compute mean XYZ per voxel using np.bincount (faster than np.add.at)
    sub_xyz = np.zeros((num_voxels, 3), dtype=np.float32)
    sub_xyz[:, 0] = np.bincount(inverse_indices, weights=points[:, 0])
    sub_xyz[:, 1] = np.bincount(inverse_indices, weights=points[:, 1])
    sub_xyz[:, 2] = np.bincount(inverse_indices, weights=points[:, 2])

    counts   = np.bincount(inverse_indices, minlength=num_voxels)[:, None]
    sub_xyz /= counts

    # 6. Majority-vote label across ALL 13 classes (generic, no hardcoding)
    num_classes = int(labels.max()) + 1
    label_votes = np.zeros((num_voxels, num_classes), dtype=np.int32)
    for cls in range(num_classes):
        is_cls = (labels == cls).astype(np.int32)
        label_votes[:, cls] = np.bincount(inverse_indices, weights=is_cls, minlength=num_voxels)
    sub_labels = np.argmax(label_votes, axis=1).astype(np.uint8)

    return sub_xyz, sub_labels


# ==========================================
# SPATIAL BLOCK PARTITIONING
# ==========================================
def spatial_block_partition(xyz, labels, block_size=30.0, min_points=512,
                             return_discarded=False):
    """Slide a regular block_size x block_size grid over the XY plane.
    Returns a list of (block_xyz, block_labels) tuples that meet the
    minimum point threshold.

    Args:
        xyz              (np.ndarray): Subsampled XYZ point cloud, shape [M, 3].
        labels           (np.ndarray): Per-point semantic labels, shape [M].
        block_size       (float)     : Tile side length in metres (XY plane).
        min_points       (int)       : Minimum points required to keep a tile.
        return_discarded (bool)      : If True, also return the (x0, y0) origins
                                       of discarded tiles (used by visualize.py).

    Returns:
        valid_blocks (list)         : List of (block_xyz, block_labels) tuples.
        discarded_origins (list)    : Only returned when return_discarded=True.
                                      List of (x0, y0) origins of sparse/empty tiles.
    """
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    x_max, y_max = xyz[:, 0].max(), xyz[:, 1].max()

    # Tile boundaries along X and Y
    # Extend stop by one extra block_size so the last partial tile always
    # covers every point up to the true x_max / y_max boundary.
    x_starts = np.arange(x_min, x_max + block_size, block_size)
    y_starts = np.arange(y_min, y_max + block_size, block_size)

    valid_blocks      = []
    discarded_origins = []   # (x0, y0) of tiles that failed the threshold
    valid_origins     = []   # (x0, y0) of tiles that passed the threshold

    for x0 in x_starts:
        x1 = x0 + block_size
        for y0 in y_starts:
            y1 = y0 + block_size

            # Boolean mask: points that fall inside this XY tile
            mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x1) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y1)
            )

            block_xyz    = xyz[mask]
            block_labels = labels[mask]

            # Enforce minimum point threshold — discard under-populated tiles
            if len(block_xyz) < min_points:
                discarded_origins.append((x0, y0))
                continue

            valid_blocks.append((block_xyz, block_labels))
            valid_origins.append((x0, y0))

    if return_discarded:
        return valid_blocks, valid_origins, discarded_origins

    return valid_blocks


# ==========================================
# FIXED-SIZE RESAMPLING (N = 1024)
# ==========================================
def resample_block(block_xyz, block_labels, n_points=1024):
    """Enforce an exact fixed size of n_points on a single block.
    - More than n_points  -> random downsampling (no replacement)
    - Fewer than n_points -> random upsampling with replacement
    """
    current_size = len(block_xyz)

    if current_size > n_points:
        # Random downsampling — pick n_points indices without replacement
        indices = np.random.choice(current_size, n_points, replace=False)
    else:
        # Random upsampling with replacement to reach exactly n_points
        indices = np.random.choice(current_size, n_points, replace=True)

    return block_xyz[indices], block_labels[indices]


# ==========================================
# NORMALIZATION (Centroid + Unit Sphere)
# ==========================================
def normalize_block(block_xyz):
    """Center block at origin and normalize to unit sphere.
    Step 1: P_centered  = P - P_centroid
    Step 2: P_normalized = P_centered / max(||P_centered||_2)
    Returns normalized XYZ array of shape [N, 3].
    """
    # Step 1: Centroid centering
    centroid   = block_xyz.mean(axis=0)            # shape [3]
    p_centered = block_xyz - centroid              # shape [N, 3]

    # Step 2: Unit sphere scaling
    distances  = np.linalg.norm(p_centered, axis=1)  # shape [N]
    max_dist   = distances.max()

    if max_dist > 0:
        p_normalized = p_centered / max_dist
    else:
        p_normalized = p_centered                  # edge-case: all identical points

    return p_normalized.astype(np.float32)         # shape [N, 3]


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def preprocess_sensaturban():
    splits = ['train', 'test']

    for split in splits:
        split_in_dir  = os.path.join(BASE_INPUT_FOLDER, split)
        split_out_dir = os.path.join(BASE_OUTPUT_FOLDER, split)

        if not os.path.exists(split_in_dir):
            print(f"Directory not found, skipping: {split_in_dir}")
            continue

        os.makedirs(split_out_dir, exist_ok=True)
        ply_files   = [f for f in os.listdir(split_in_dir)
                       if f.endswith('.ply') and not f.startswith('._')]
        total_files = len(ply_files)

        print(f"\n--- Processing '{split}' split ({total_files} files) ---")

        for i, filename in enumerate(ply_files):
            print(f"\n[{i+1}/{total_files}] Processing {filename}...")
            input_path  = os.path.join(split_in_dir, filename)
            stem        = os.path.splitext(filename)[0]  # filename without extension

            # ------------------------------------------------------------------
            # Phase 1: Load ALL 13 classes — no filtering
            # ------------------------------------------------------------------
            xyz, labels, has_class = read_ply_minimal(input_path)
            initial_count = len(xyz)
            print(f"  Phase 1 | Loaded {initial_count:,} points (all classes kept)")

            # ------------------------------------------------------------------
            # Phase 3: Voxel grid subsampling (GRID_SIZE = 0.2 m)
            # ------------------------------------------------------------------
            sub_xyz, sub_labels = grid_subsample(xyz, labels, grid_size=GRID_SIZE)
            sub_count = len(sub_xyz)
            print(f"  Phase 3 | After grid subsampling (grid={GRID_SIZE}m): {sub_count:,} points")

            del xyz, labels
            gc.collect()

            # ------------------------------------------------------------------
            # Step 4a: Spatial block partitioning (30 m x 30 m XY tiles)
            # ------------------------------------------------------------------
            valid_blocks = spatial_block_partition(
                sub_xyz, sub_labels,
                block_size=BLOCK_SIZE,
                min_points=MIN_POINTS_BLOCK
            )
            print(f"  Step 4a | Spatial blocks (>= {MIN_POINTS_BLOCK} pts): {len(valid_blocks)}")

            del sub_xyz, sub_labels
            gc.collect()

            # ------------------------------------------------------------------
            # Step 4b + Step 5: Resample to N=1024, normalize, save as .npy
            # ------------------------------------------------------------------
            saved_count = 0
            for b_idx, (block_xyz, block_labels) in enumerate(valid_blocks):

                # Step 4b: Fixed-size resampling to exactly N_POINTS
                resampled_xyz, _ = resample_block(block_xyz, block_labels, n_points=N_POINTS)

                # Step 5: Centroid centering + unit sphere normalization
                normalized_xyz = normalize_block(resampled_xyz)   # shape [1024, 3]

                # Save as .npy — shape [1024, 3], dtype float32
                out_filename = f"{stem}_block_{b_idx:04d}.npy"
                out_path     = os.path.join(split_out_dir, out_filename)
                np.save(out_path, normalized_xyz)

                saved_count += 1

                del block_xyz, block_labels, resampled_xyz, normalized_xyz
                gc.collect()

            print(f"  Step 5  | Saved {saved_count} blocks -> {split_out_dir}/")

        print(f"\n=== '{split}' split complete ===")

    print("\n\nAll splits processed successfully.")


if __name__ == "__main__":
    preprocess_sensaturban()