"""
visualize.py — Data Preprocessing Visualization Script.

Generates and saves 5 diagnostic visualizations for the SensatUrban
preprocessing and data loading pipeline:

  1. Before / After 3D Point Cloud (side-by-side in 1 image)
  2. Point Count Distribution Histogram (per block, with 512-pt threshold line)
  3. Semantic Class Breakdown Bar Chart (all 13 classes across all blocks)
  4. Spatial Block Partition Map (top-down XY, green=kept / red=discarded)
  5. Pe vs Pm Hyperplane Slicing Sanity Check (3-panel 3D scatter)

Output structure (nothing is ever overwritten):
  outputs/visualizations/
  └── preprocessing_[YYYY-MM-DD_HH-MM]/
      ├── 3d_before_after/
      │   └── {stem}_before_after.png
      ├── histograms/
      │   └── point_count_distribution.png
      ├── class_charts/
      │   └── semantic_class_breakdown.png
      ├── block_maps/
      │   └── {stem}_partition_map.png
      └── pe_pm_split/
          └── preprocessing_sanity_check.png

Usage:
    python visualize.py
"""

import os
import sys
import gc
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
BASE_INPUT_FOLDER  = "../SensatUrban"       # Relative path for portable hard drive
BASE_VIZ_OUTPUT    = os.path.join("outputs", "visualizations")
SAMPLE_SPLIT       = "train"                # Which split to sample from
SAMPLE_FILE_INDEX  = 0                      # Index of the .ply file to use as the visual sample
MAX_STAT_FILES     = 5                      # Maximum number of PLY files to parse for stats
                                            # (Set to None to parse all files in split)

# ==========================================
# IMPORTS FROM THIS PROJECT
# ==========================================
# Add project root to path so imports work regardless of where you run this from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sensat_dataset import hyperplane_cut, resample_pcd

from preprocess import (
    read_ply_minimal,
    grid_subsample,
    spatial_block_partition,
    resample_block,
    normalize_block,
    GRID_SIZE,
    BLOCK_SIZE,
    MIN_POINTS_BLOCK,
    N_POINTS,
)
from utils.plot_utils import (
    get_run_dir,
    make_subdir,
    plot_pointcloud_before_after_3d,
    plot_pe_pm_split_3d,
    plot_histogram,
    plot_bar_chart,
    plot_partition_map_2d,
    SENSATURBAN_CLASS_NAMES,
)


# ==========================================
# HELPER: COLLECT STATS ACROSS FILES
# ==========================================
def _collect_dataset_stats(split_dir, max_files=MAX_STAT_FILES):
    """Walk .ply files in split_dir through the pipeline and collect:
      - per-block point counts (before resampling)     → for histogram
      - per-class point totals (across sub-sampled clouds) → for bar chart

    Args:
        split_dir (str): Path to split directory containing .ply files.
        max_files (int): Limit number of files to process for speed.

    Returns:
        block_point_counts (list[int]): Raw point count of each valid block.
        class_totals       (dict)     : {class_id: total_points} across parsed files.
    """
    ply_files = sorted([
        f for f in os.listdir(split_dir)
        if f.endswith('.ply') and not f.startswith('._')
    ])

    if max_files is not None and max_files > 0:
        ply_files = ply_files[:max_files]

    block_point_counts = []
    class_totals       = {cid: 0 for cid in range(13)}
    total_files        = len(ply_files)

    print(f"\n  Collecting stats across {total_files} sampled file(s) in '{split_dir}' ...")

    for i, filename in enumerate(ply_files):
        print(f"    [{i+1}/{total_files}] {filename}")
        input_path = os.path.join(split_dir, filename)

        # Phase 1: Load
        xyz, labels, _ = read_ply_minimal(input_path)

        # Phase 3: Grid subsampling
        sub_xyz, sub_labels = grid_subsample(xyz, labels, grid_size=GRID_SIZE)
        del xyz, labels
        gc.collect()

        # Accumulate class totals from this file
        for cid in range(13):
            class_totals[cid] += int(np.sum(sub_labels == cid))

        # Step 4: Partition — collect block sizes (no return_discarded needed here)
        valid_blocks = spatial_block_partition(
            sub_xyz, sub_labels,
            block_size=BLOCK_SIZE,
            min_points=MIN_POINTS_BLOCK,
            return_discarded=False
        )

        for (block_xyz, _) in valid_blocks:
            block_point_counts.append(len(block_xyz))

        del sub_xyz, sub_labels, valid_blocks
        gc.collect()

    return block_point_counts, class_totals


# ==========================================
# MAIN VISUALIZATION PIPELINE
# ==========================================
def visualize_preprocessing():
    split_dir = os.path.join(BASE_INPUT_FOLDER, SAMPLE_SPLIT)

    if not os.path.exists(split_dir):
        print(f"[ERROR] Input directory not found: {split_dir}")
        print(f"        Please run preprocess.py first, or check BASE_INPUT_FOLDER.")
        return

    ply_files = sorted([
        f for f in os.listdir(split_dir)
        if f.endswith('.ply') and not f.startswith('._')
    ])

    if not ply_files:
        print(f"[ERROR] No .ply files found in: {split_dir}")
        return

    # ---------------------------------------------------------------
    # Create timestamped output directory for this run
    # ---------------------------------------------------------------
    run_dir = get_run_dir(BASE_VIZ_OUTPUT, stage_name="preprocessing")

    dir_3d_before_after = make_subdir(run_dir, "3d_before_after")
    dir_histograms      = make_subdir(run_dir, "histograms")
    dir_class_charts    = make_subdir(run_dir, "class_charts")
    dir_block_maps      = make_subdir(run_dir, "block_maps")

    print(f"\n{'='*60}")
    print(f"  Preprocessing Visualizations")
    print(f"  Output: {run_dir}")
    print(f"{'='*60}")

    # ---------------------------------------------------------------
    # Visualization 2 & 3: Collect stats across ALL files (histogram + bar chart)
    # ---------------------------------------------------------------
    block_point_counts, class_totals = _collect_dataset_stats(split_dir)

    # --- Visualization 2: Point Count Distribution Histogram ---
    print(f"\n  [2/5] Saving Point Count Distribution Histogram ...")
    histogram_path = os.path.join(dir_histograms, "point_count_distribution.png")
    plot_histogram(
        values          = block_point_counts,
        title           = (f"Point Count Distribution per Spatial Block\n"
                           f"(before fixed-size resampling to N={N_POINTS})\n"
                           f"Total valid blocks: {len(block_point_counts):,}"),
        xlabel          = "Points per Block (raw)",
        ylabel          = "Number of Blocks",
        save_path       = histogram_path,
        threshold       = MIN_POINTS_BLOCK,
        threshold_label = f"Min threshold = {MIN_POINTS_BLOCK} pts",
        color           = '#7c4dff',
    )

    # --- Visualization 3: Semantic Class Breakdown Bar Chart ---
    print(f"\n  [3/5] Saving Semantic Class Breakdown Bar Chart ...")
    label_ids = [cid for cid in range(13) if class_totals[cid] > 0]
    counts    = [class_totals[cid] for cid in label_ids]
    chart_path = os.path.join(dir_class_charts, "semantic_class_breakdown.png")
    plot_bar_chart(
        label_ids   = label_ids,
        counts      = counts,
        title       = ("Semantic Class Distribution After Grid Subsampling\n"
                       f"(All 13 SensatUrban Classes — {SAMPLE_SPLIT} split)"),
        save_path   = chart_path,
        class_names = SENSATURBAN_CLASS_NAMES,
    )

    # ---------------------------------------------------------------
    # Visualizations 1 & 4: Use a single sample file
    # ---------------------------------------------------------------
    idx         = min(SAMPLE_FILE_INDEX, len(ply_files) - 1)
    sample_file = ply_files[idx]
    stem        = os.path.splitext(sample_file)[0]
    input_path  = os.path.join(split_dir, sample_file)

    print(f"\n  Sample file for 3D + partition map: '{sample_file}'")

    # Phase 1: Load sample
    xyz_raw_full, labels_full, _ = read_ply_minimal(input_path)
    print(f"    Loaded {len(xyz_raw_full):,} raw points.")

    # Phase 3: Grid subsampling on sample
    sub_xyz, sub_labels = grid_subsample(xyz_raw_full, labels_full, grid_size=GRID_SIZE)
    del xyz_raw_full, labels_full
    gc.collect()
    print(f"    After grid subsampling: {len(sub_xyz):,} points.")

    # Step 4: Partition with return_discarded=True for the partition map
    valid_blocks, valid_origins, discarded_origins = spatial_block_partition(
        sub_xyz, sub_labels,
        block_size=BLOCK_SIZE,
        min_points=MIN_POINTS_BLOCK,
        return_discarded=True
    )
    print(f"    Valid blocks: {len(valid_blocks)} | Discarded: {len(discarded_origins)}")

    # Pick the first valid block as the visual sample for the 3D plot
    if valid_blocks:
        sample_block_xyz, sample_block_labels = valid_blocks[0]
    else:
        print("    [WARNING] No valid blocks found in sample file. Skipping 3D plot.")
        sample_block_xyz = None

    if sample_block_xyz is not None:
        # Step 4b: Resample to N_POINTS
        resampled_xyz, _ = resample_block(
            sample_block_xyz, sample_block_labels, n_points=N_POINTS
        )

        # Step 5: Normalize
        normalized_xyz = normalize_block(resampled_xyz)

        # --- Visualization 1/5: Before / After 3D ---
        print(f"\n  [1/5] Saving Before / After 3D Point Cloud ...")
        before_after_path = os.path.join(dir_3d_before_after, f"{stem}_before_after.png")
        plot_pointcloud_before_after_3d(
            xyz_raw        = sample_block_xyz,
            xyz_normalized = normalized_xyz,
            stem           = stem,
            save_path      = before_after_path,
        )

        # --- Visualization 5: Pe vs Pm Slicing Sanity Check ---
        print(f"\n  [5/5] Saving Pe/Pm Hyperplane Slicing Sanity Check ...")
        dir_pe_pm_split = make_subdir(run_dir, "pe_pm_split")
        
        # Apply hyperplane cut
        pe_raw, pm_raw = hyperplane_cut(normalized_xyz)
        
        # Resample both sides to N_POINTS just like dataset.py does
        from dataset import resample_pcd
        pe = resample_pcd(pe_raw, N_POINTS)
        pm = resample_pcd(pm_raw, N_POINTS)
        
        pe_pm_path = os.path.join(dir_pe_pm_split, "preprocessing_sanity_check.png")
        plot_pe_pm_split_3d(
            xyz_target = normalized_xyz,
            xyz_pe     = pe,
            xyz_pm     = pm,
            stem       = stem,
            save_path  = pe_pm_path,
        )

        del resampled_xyz, sample_block_xyz, pe, pm
        del normalized_xyz
        gc.collect()

    # --- Visualization 4/5: Block Partition Map ---
    print(f"\n  [4/5] Saving Block Partition Map ...")
    partition_map_path = os.path.join(dir_block_maps, f"{stem}_partition_map.png")
    plot_partition_map_2d(
        xyz                    = sub_xyz,
        block_size             = BLOCK_SIZE,
        valid_tile_origins     = valid_origins,
        discarded_tile_origins = discarded_origins,
        stem                   = stem,
        save_path              = partition_map_path,
    )

    del sub_xyz, sub_labels, valid_blocks
    gc.collect()

    print(f"\n{'='*60}")
    print(f"  All visualizations saved to:")
    print(f"  {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    visualize_preprocessing()
