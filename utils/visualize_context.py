"""
visualize_context.py — Step 1.3: Context Visualization Sanity Check

Loads preprocessed target tiles and their K=4 nearest spatial neighbors,
aligns the neighbor point clouds in their true relative physical locations using:
    P_neighbor_aligned = P_neighbor + (Centroid_neighbor - Centroid_target)
and plots a 3D scatter plot verifying spatial alignment (Target: Blue, Neighbors: Red/Orange).

Output:
    experiments/context_sanity_check.png

Usage:
    python utils/visualize_context.py
    # OR with custom paths:
    python utils/visualize_context.py --data_dir ../../datasets/SensatUrban_Out --out experiments/context_sanity_check.png
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def resolve_path(candidates, description="file"):
    """Find the first existing path from a list of candidate locations."""
    for p in candidates:
        if p and os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Could not find {description}. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def find_npy_file(data_dir, filename):
    """Locate a .npy file inside data_dir (checking train/, test/, or root)."""
    checks = [
        os.path.join(data_dir, 'train', filename),
        os.path.join(data_dir, 'test', filename),
        os.path.join(data_dir, filename),
    ]
    for c in checks:
        if os.path.exists(c):
            return c
    return None


def plot_context_sanity_check(
    centroids_path='datasets/tile_centroids.json',
    neighbor_map_path='datasets/neighbor_map.json',
    data_dir='SensatUrban_Out',
    output_path='experiments/context_sanity_check.png',
    num_samples=3
):
    print('=' * 70)
    print('  SensatUrban Context Visualization Sanity Check (Step 1.3)')
    print('=' * 70)

    # 1. Resolve JSON paths
    centroids_file = resolve_path([
        centroids_path,
        'datasets/tile_centroids.json',
        'tile_centroids.json',
        os.path.join(os.path.dirname(__file__), '..', 'datasets', 'tile_centroids.json'),
    ], "tile_centroids.json")

    neighbor_file = resolve_path([
        neighbor_map_path,
        'datasets/neighbor_map.json',
        'neighbor_map.json',
        os.path.join(os.path.dirname(__file__), '..', 'datasets', 'neighbor_map.json'),
    ], "neighbor_map.json")

    # 2. Resolve dataset directory
    resolved_data_dir = None
    data_dir_candidates = [
        data_dir,
        'SensatUrban_Out',
        '../SensatUrban_Out',
        '../../datasets/SensatUrban_Out',
        'datasets/SensatUrban_Out',
        '/kaggle/input/sensaturban-out/SensatUrban_Out',
        '/kaggle/input/datasets/syedrayyanadil/sensaturban-out/SensatUrban_Out',
    ]
    for d in data_dir_candidates:
        if d and os.path.exists(d):
            resolved_data_dir = os.path.abspath(d)
            break

    if not resolved_data_dir:
        raise FileNotFoundError(
            f"Could not find SensatUrban_Out dataset directory. Tried:\n"
            + "\n".join(f"  - {c}" for c in data_dir_candidates)
        )

    print(f"  Tile Centroids : {centroids_file}")
    print(f"  Neighbor Map   : {neighbor_file}")
    print(f"  Data Directory : {resolved_data_dir}")
    print(f"  Output Image   : {os.path.abspath(output_path)}")
    print('=' * 70)

    # 3. Load JSON files
    with open(centroids_file, 'r') as f:
        centroids = json.load(f)

    with open(neighbor_file, 'r') as f:
        neighbor_map = json.load(f)

    # 4. Find valid target candidates where target AND all 4 neighbors exist on disk
    valid_candidates = []
    for target_fn, neighbor_fns in neighbor_map.items():
        target_path = find_npy_file(resolved_data_dir, target_fn)
        if not target_path:
            continue

        all_neighbors_exist = all(
            find_npy_file(resolved_data_dir, n_fn) is not None
            for n_fn in neighbor_fns
        )
        if all_neighbors_exist:
            valid_candidates.append(target_fn)

    if not valid_candidates:
        raise RuntimeError(
            f"No target tiles with all 4 neighbors present were found in {resolved_data_dir}."
        )

    print(f"\n[INFO] Found {len(valid_candidates)} complete target-neighbor groups available on disk.")

    # Select sample targets (spaced out for variety)
    step = max(1, len(valid_candidates) // num_samples)
    chosen_targets = [valid_candidates[i * step] for i in range(min(num_samples, len(valid_candidates)))]
    actual_samples = len(chosen_targets)

    # 5. Create 3D multi-panel visualization
    fig = plt.figure(figsize=(7 * actual_samples, 7))
    fig.patch.set_facecolor('#1a1a2e')

    neighbor_colors = ['#ff1744', '#ff5252', '#ff9100', '#ffab40'] # Red to warm amber shades
    target_color    = '#00e5ff'                                    # Electric Cyan / Blue

    for panel_idx, target_fn in enumerate(chosen_targets):
        target_path = find_npy_file(resolved_data_dir, target_fn)
        target_pts  = np.load(target_path) # [1024, 3]
        target_c    = np.array(centroids[target_fn], dtype=np.float64) # [3]

        neighbor_fns = neighbor_map[target_fn]

        ax = fig.add_subplot(1, actual_samples, panel_idx + 1, projection='3d')
        ax.set_facecolor('#0d0d1a')

        # Plot target block in Blue/Cyan (at relative origin [0, 0, 0])
        ax.scatter(
            target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
            c=target_color, s=4.0, alpha=0.95, label='Target Block (Center)',
            depthshade=True
        )

        # Plot each of the 4 neighbors with physical offset alignment
        for n_idx, n_fn in enumerate(neighbor_fns):
            n_path = find_npy_file(resolved_data_dir, n_fn)
            n_pts  = np.load(n_path) # [1024, 3]
            n_c    = np.array(centroids[n_fn], dtype=np.float64) # [3]

            # Physical centroid offset (in meters)
            offset = n_c - target_c
            distance = np.linalg.norm(offset[:2]) # 2D ground distance

            # True relative physical alignment
            n_aligned = n_pts + offset

            ax.scatter(
                n_aligned[:, 0], n_aligned[:, 1], n_aligned[:, 2],
                c=neighbor_colors[n_idx % len(neighbor_colors)],
                s=2.5, alpha=0.75,
                label=f'Neighbor {n_idx + 1} (dist={distance:.1f}m)',
                depthshade=True
            )

        # Aesthetics and axes styling
        ax.set_title(
            f"Sample {panel_idx + 1}: {target_fn[:25]}...\nReal-World Centroid: [{target_c[0]:.1f}, {target_c[1]:.1f}, {target_c[2]:.1f}]m",
            color='white', fontsize=11, fontweight='bold', pad=12
        )
        ax.set_xlabel('X (meters offset)', color='#aaaaaa', labelpad=8)
        ax.set_ylabel('Y (meters offset)', color='#aaaaaa', labelpad=8)
        ax.set_zlabel('Z (meters offset)', color='#aaaaaa', labelpad=8)
        ax.tick_params(colors='#aaaaaa')
        ax.grid(color='#333355', linestyle='--', alpha=0.4)

        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        ax.legend(
            loc='upper right', facecolor='#1a1a2e', edgecolor='#444466',
            labelcolor='white', fontsize=8, framealpha=0.9
        )

    fig.suptitle(
        "SensatUrban Spatial Context Sanity Check — Target Block & K=4 Real-World Aligned Neighbors",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout()

    # 6. Save image
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)

    print('\n' + '=' * 70)
    print(f'✓ Visual Sanity Check Image Generated Successfully!')
    print(f'  Saved to: {os.path.abspath(output_path)}')
    print('=' * 70)

    return os.path.abspath(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize Target Point Cloud Block and its K=4 Spatial Neighbors.'
    )
    parser.add_argument(
        '--centroids',
        type=str,
        default='datasets/tile_centroids.json',
        help='Path to tile_centroids.json (default: datasets/tile_centroids.json)'
    )
    parser.add_argument(
        '--neighbor_map',
        type=str,
        default='datasets/neighbor_map.json',
        help='Path to neighbor_map.json (default: datasets/neighbor_map.json)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='SensatUrban_Out',
        help='Path to SensatUrban_Out dataset folder (default: SensatUrban_Out)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='experiments/context_sanity_check.png',
        help='Output PNG path (default: experiments/context_sanity_check.png)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of target samples to visualize (default: 3)'
    )
    args = parser.parse_args()

    plot_context_sanity_check(
        centroids_path=args.centroids,
        neighbor_map_path=args.neighbor_map,
        data_dir=args.data_dir,
        output_path=args.out,
        num_samples=args.num_samples,
    )
