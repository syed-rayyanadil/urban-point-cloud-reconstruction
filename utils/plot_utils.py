"""
plot_utils.py — Reusable plotting utilities for GenerativeVaeReconstruction3DPointcloud.

All functions:
  - Accept raw NumPy arrays (no pipeline-specific dependencies).
  - Save images to disk using the non-interactive 'Agg' backend (VSCode-safe).
  - Are designed to be imported by ANY future pipeline script (visualize.py,
    train.py, metrics evaluation scripts, etc.).

Functions:
  - get_run_dir()                    : Create a timestamped stage output folder.
  - make_subdir()                    : Create a named subfolder inside a run dir.
  - plot_pointcloud_before_after_3d(): Side-by-side 3D scatter (Before vs After).
  - plot_histogram()                 : Generic histogram with optional threshold line.
  - plot_bar_chart()                 : Generic bar chart (class distributions, metrics, etc.).
  - plot_partition_map_2d()          : Top-down XY block partition map (green=kept, red=discarded).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — saves to disk, never opens a window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ==========================================
# SENSATURBAN CLASS NAME LOOKUP
# (Kept here so any future script can import it from one place)
# ==========================================
SENSATURBAN_CLASS_NAMES = {
    0:  'Ground',
    1:  'High Vegetation',
    2:  'Buildings',
    3:  'Walls',
    4:  'Bridge',
    5:  'Parking',
    6:  'Rail',
    7:  'Traffic Roads',
    8:  'Street Furniture',
    9:  'Cars',
    10: 'Footpath',
    11: 'Bikes',
    12: 'Water',
}

# ==========================================
# RUN DIRECTORY MANAGEMENT
# ==========================================
def get_run_dir(base_output_dir, stage_name):
    """Create and return a timestamped output folder for a specific pipeline stage.

    Folder naming format:
        {stage_name}_[YYYY-MM-DD_HH-MM]

    Example:
        preprocessing_[2026-08-09_23-55]

    Running the same script twice always produces a NEW folder — nothing is
    ever overwritten.

    Args:
        base_output_dir (str): Root output directory (e.g. 'outputs/visualizations').
        stage_name      (str): Stage name in snake_case (e.g. 'preprocessing', 'metrics', 'training').

    Returns:
        str: Absolute path to the newly created run directory.
    """
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"{stage_name}_[{timestamp}]"
    run_dir     = os.path.join(base_output_dir, folder_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_subdir(run_dir, subdir_name):
    """Create a named subdirectory inside a run directory.

    Args:
        run_dir     (str): Path to the parent run directory.
        subdir_name (str): Name of the subdirectory to create.

    Returns:
        str: Absolute path to the created subdirectory.
    """
    path = os.path.join(run_dir, subdir_name)
    os.makedirs(path, exist_ok=True)
    return path


# ==========================================
# DARK THEME HELPER
# ==========================================
_DARK_BG    = '#1a1a2e'   # Figure background
_DARK_AXES  = '#0d0d1a'   # Axes background
_GRID_COLOR = '#333355'   # Grid & spine color
_LABEL_CLR  = '#aaaaaa'   # Axis label color
_TEXT_WHITE = '#ffffff'   # Title & value text


def _apply_dark_axes(ax):
    """Apply consistent dark theme styling to a 2D matplotlib Axes object."""
    ax.set_facecolor(_DARK_AXES)
    ax.tick_params(colors=_LABEL_CLR)
    ax.grid(color=_GRID_COLOR, linestyle='--', alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)


# ==========================================
# 3D POINT CLOUD: BEFORE / AFTER (1 IMAGE)
# ==========================================
def plot_pointcloud_before_after_3d(xyz_raw, xyz_normalized, stem, save_path):
    """Save a single side-by-side figure comparing the raw tile vs the final
    normalized block.

    Left panel  : Raw point cloud slice (after spatial tiling, before resampling).
    Right panel : Normalized block (1024 pts, centered at origin, unit sphere).

    Points are coloured by height (Z coordinate) using the 'plasma' colormap.

    Args:
        xyz_raw        (np.ndarray): Raw XYZ array, shape [M, 3].
        xyz_normalized (np.ndarray): Normalized XYZ array, shape [N, 3].
        stem           (str)       : Source filename stem for the plot title.
        save_path      (str)       : Full path (including filename) to save the PNG.
    """
    fig = plt.figure(figsize=(17, 7))
    fig.patch.set_facecolor(_DARK_BG)

    panels = [
        (xyz_raw,        f"RAW  (after spatial tiling)\nPoints: {len(xyz_raw):,}"),
        (xyz_normalized, f"NORMALIZED  (centroid-centred + unit sphere)\nPoints: {len(xyz_normalized):,}"),
    ]

    for col_idx, (data, subtitle) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection='3d')
        ax.set_facecolor(_DARK_AXES)

        heights = data[:, 2]
        sc = ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c=heights, cmap='plasma', s=1.8, alpha=0.85, depthshade=True
        )

        cbar = plt.colorbar(sc, ax=ax, shrink=0.45, pad=0.1)
        cbar.set_label('Height (Z)', color=_LABEL_CLR, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=_LABEL_CLR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_LABEL_CLR, fontsize=7)

        ax.set_title(subtitle, color=_TEXT_WHITE, fontsize=10, pad=10)
        ax.set_xlabel('X', color=_LABEL_CLR, fontsize=8)
        ax.set_ylabel('Y', color=_LABEL_CLR, fontsize=8)
        ax.set_zlabel('Z', color=_LABEL_CLR, fontsize=8)
        ax.tick_params(colors=_LABEL_CLR, labelsize=6)

        # Transparent pane walls with subtle edges
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(_GRID_COLOR)

    fig.suptitle(
        f'Point Cloud Pipeline — Before vs After\n{stem}',
        color=_TEXT_WHITE, fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [Saved] {os.path.basename(save_path)}")


# ==========================================
# 3D POINT CLOUD: Pe vs Pm SPLIT (3 PANELS)
# ==========================================
def plot_pe_pm_split_3d(xyz_target, xyz_pe, xyz_pm, stem, save_path):
    """Save a 3-panel side-by-side figure showing the hyperplane cut.

    Left   : Full Block (Target)
    Middle : Visible Part (Pe) - colored blue
    Right  : Missing Part (Pm) - colored red/orange

    Args:
        xyz_target (np.ndarray): Full XYZ array, shape [N, 3].
        xyz_pe     (np.ndarray): Pe XYZ array, shape [N, 3].
        xyz_pm     (np.ndarray): Pm XYZ array, shape [N, 3].
        stem       (str)       : Source filename stem for the plot title.
        save_path  (str)       : Full path (including filename) to save the PNG.
    """
    fig = plt.figure(figsize=(24, 7))
    fig.patch.set_facecolor(_DARK_BG)

    panels = [
        (xyz_target, f"1. FULL BLOCK (Target)\nPoints: {len(xyz_target):,}", 'plasma'),
        (xyz_pe,     f"2. VISIBLE PART (Pe)\nPoints: {len(xyz_pe):,} (resampled)", 'winter'),
        (xyz_pm,     f"3. MISSING PART (Pm)\nPoints: {len(xyz_pm):,} (resampled)", 'autumn'),
    ]

    for col_idx, (data, subtitle, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, col_idx + 1, projection='3d')
        ax.set_facecolor(_DARK_AXES)

        heights = data[:, 2]
        sc = ax.scatter(
            data[:, 0], data[:, 1], data[:, 2],
            c=heights, cmap=cmap, s=1.8, alpha=0.85, depthshade=True
        )

        ax.set_title(subtitle, color=_TEXT_WHITE, fontsize=11, pad=10)
        ax.set_xlabel('X', color=_LABEL_CLR, fontsize=8)
        ax.set_ylabel('Y', color=_LABEL_CLR, fontsize=8)
        ax.set_zlabel('Z', color=_LABEL_CLR, fontsize=8)
        ax.tick_params(colors=_LABEL_CLR, labelsize=6)

        # Ensure consistent [-1, 1] bounds since data is normalized
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # Transparent pane walls with subtle edges
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(_GRID_COLOR)

    fig.suptitle(
        f'Hyperplane Slicing Partitioning (Pe vs Pm)\n{stem}',
        color=_TEXT_WHITE, fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [Saved] {os.path.basename(save_path)}")


# ==========================================
# HISTOGRAM (Generic — reusable for any numeric distribution)
# ==========================================
def plot_histogram(values, title, xlabel, ylabel, save_path,
                   threshold=None, threshold_label=None, color='#7c4dff'):
    """Save a histogram of any numeric distribution.

    Optionally draws a vertical dashed threshold line (e.g. min_points = 512).
    Reusable for: point count distributions, loss curves per epoch,
    metric value distributions, etc.

    Args:
        values          (array-like): 1D array of numeric values to histogram.
        title           (str)       : Plot title.
        xlabel          (str)       : X-axis label.
        ylabel          (str)       : Y-axis label.
        save_path       (str)       : Full path to save the PNG.
        threshold       (float)     : Optional X-position for a vertical line.
        threshold_label (str)       : Legend label for the threshold line.
        color           (str)       : Bar fill colour (hex).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(_DARK_BG)
    _apply_dark_axes(ax)

    ax.hist(values, bins=40, color=color, edgecolor='#ffffff18', alpha=0.87)

    if threshold is not None:
        label = threshold_label or f'Threshold = {threshold:,}'
        ax.axvline(x=threshold, color='#ff6b6b', linestyle='--',
                   linewidth=2.0, label=label)
        ax.legend(facecolor=_DARK_BG, edgecolor=_GRID_COLOR,
                  labelcolor=_TEXT_WHITE, fontsize=10)

    ax.set_title(title,  color=_TEXT_WHITE, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel(xlabel, color=_LABEL_CLR,  fontsize=11)
    ax.set_ylabel(ylabel, color=_LABEL_CLR,  fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [Saved] {os.path.basename(save_path)}")


# ==========================================
# BAR CHART (Generic — reusable for class distributions, metric comparisons, etc.)
# ==========================================
def plot_bar_chart(label_ids, counts, title, save_path, class_names=None):
    """Save a bar chart for any categorical distribution.

    When class_names is provided (e.g. SENSATURBAN_CLASS_NAMES), integer IDs
    are replaced with human-readable names on the X axis.

    Reusable for: semantic class distributions, per-class metric comparisons,
    dataset split statistics, etc.

    Args:
        label_ids    (list[int]) : Ordered list of category IDs.
        counts       (list[int]) : Point/sample count per category.
        title        (str)       : Plot title.
        save_path    (str)       : Full path to save the PNG.
        class_names  (dict)      : Optional {id: name} mapping for X labels.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(_DARK_BG)
    _apply_dark_axes(ax)

    x_labels = [class_names.get(lid, str(lid)) for lid in label_ids] \
               if class_names else [str(lid) for lid in label_ids]

    bar_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(label_ids)))
    bars = ax.bar(x_labels, counts, color=bar_colors,
                  edgecolor='#ffffff18', alpha=0.88)

    # Value annotation on top of each bar
    max_count = max(counts) if counts else 1
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max_count * 0.012,
            f'{int(count):,}',
            ha='center', va='bottom',
            color=_TEXT_WHITE, fontsize=8, fontweight='bold'
        )

    ax.set_title(title, color=_TEXT_WHITE, fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('Semantic Class',  color=_LABEL_CLR, fontsize=11)
    ax.set_ylabel('Point Count',     color=_LABEL_CLR, fontsize=11)
    plt.xticks(rotation=30, ha='right', color=_LABEL_CLR, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [Saved] {os.path.basename(save_path)}")


# ==========================================
# 2D BLOCK PARTITION MAP (Top-Down XY View)
# ==========================================
def plot_partition_map_2d(xyz, block_size, valid_tile_origins,
                          discarded_tile_origins, stem, save_path):
    """Save a bird's-eye XY view of the city scan with the block grid overlaid.

    Green tiles  = valid blocks (kept, >= min_points threshold).
    Red tiles    = discarded blocks (too sparse, < min_points threshold).
    Background   = raw point cloud (dim, for spatial reference).

    Args:
        xyz                   (np.ndarray): Full XYZ array after grid subsampling, shape [M, 3].
        block_size            (float)     : Side length of each tile in metres.
        valid_tile_origins    (list)      : List of (x0, y0) tuples for kept tiles.
        discarded_tile_origins(list)      : List of (x0, y0) tuples for discarded tiles.
        stem                  (str)       : Source filename stem for the plot title.
        save_path             (str)       : Full path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(13, 11))
    fig.patch.set_facecolor(_DARK_BG)
    _apply_dark_axes(ax)

    # --- Background: raw point cloud (dim scatter for spatial reference) ---
    ax.scatter(xyz[:, 0], xyz[:, 1], s=0.08, c='#445566', alpha=0.25, rasterized=True)

    # --- Discarded tiles (red) ---
    for (x0, y0) in discarded_tile_origins:
        rect = mpatches.Rectangle(
            (x0, y0), block_size, block_size,
            linewidth=0.7, edgecolor='#ff6b6b', facecolor='#ff6b6b1a'
        )
        ax.add_patch(rect)

    # --- Valid tiles (green) ---
    for (x0, y0) in valid_tile_origins:
        rect = mpatches.Rectangle(
            (x0, y0), block_size, block_size,
            linewidth=0.7, edgecolor='#06d6a0', facecolor='#06d6a01a'
        )
        ax.add_patch(rect)

    # --- Legend ---
    legend_handles = [
        mpatches.Patch(facecolor='#06d6a044', edgecolor='#06d6a0',
                       label=f'Valid blocks  : {len(valid_tile_origins)}'),
        mpatches.Patch(facecolor='#ff6b6b44', edgecolor='#ff6b6b',
                       label=f'Discarded blocks : {len(discarded_tile_origins)}'),
    ]
    ax.legend(handles=legend_handles, facecolor=_DARK_BG,
              edgecolor=_GRID_COLOR, labelcolor=_TEXT_WHITE,
              fontsize=10, loc='upper right')

    ax.set_title(
        f'Spatial Block Partition Map  ({block_size:.0f}m × {block_size:.0f}m grid)\n{stem}',
        color=_TEXT_WHITE, fontsize=13, fontweight='bold', pad=14
    )
    ax.set_xlabel('X (metres)', color=_LABEL_CLR, fontsize=11)
    ax.set_ylabel('Y (metres)', color=_LABEL_CLR, fontsize=11)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    [Saved] {os.path.basename(save_path)}")
