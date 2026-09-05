"""
build_neighbor_index.py — Step 1.2: Build Spatial KD-Tree Index

Loads tile centers/centroids, groups tiles by city (e.g. 'birmingham' vs 'cambridge'),
builds a 2D spatial KD-Tree (scipy.spatial.cKDTree) on ground (X, Y) coordinates,
queries K=4 nearest spatial neighbors for every tile, and saves the mapping to
neighbor_map.json.

Output format (neighbor_map.json):
{
    "birmingham_block_0_block_0000.npy": [
        "birmingham_block_0_block_0001.npy",
        "birmingham_block_0_block_0002.npy",
        "birmingham_block_0_block_0003.npy",
        "birmingham_block_0_block_0004.npy"
    ],
    ...
}

Usage:
    python datasets/build_neighbor_index.py
    # OR with explicit arguments:
    python datasets/build_neighbor_index.py --input datasets/tile_centroids.json --output datasets/neighbor_map.json --k 4
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.spatial import cKDTree


def resolve_input_path(input_arg):
    """Find the tile centers/centroids JSON file across common paths."""
    candidates = [
        input_arg,
        'datasets/tile_centers.json',
        'datasets/tile_centroids.json',
        'tile_centers.json',
        'tile_centroids.json',
        os.path.join(os.path.dirname(__file__), 'tile_centers.json'),
        os.path.join(os.path.dirname(__file__), 'tile_centroids.json'),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find tile centers/centroids JSON file. Tried: {candidates}"
    )


def build_neighbor_map(input_path, output_path, k_neighbors=4):
    print('=' * 70)
    print('  SensatUrban Spatial KD-Tree Neighbor Index Construction')
    print('=' * 70)
    print(f'  Input File    : {os.path.abspath(input_path)}')
    print(f'  Output File   : {os.path.abspath(output_path)}')
    print(f'  K Neighbors   : {k_neighbors}')
    print('=' * 70)

    # 1. Load tile centers / centroids
    with open(input_path, 'r') as f:
        tile_centers = json.load(f)

    total_tiles = len(tile_centers)
    print(f"\n[1/4] Loaded {total_tiles:,} tile centers from JSON.")

    # 2. Group tiles by city
    # City prefix is the first token before the underscore (e.g. 'birmingham' or 'cambridge')
    city_groups = {}
    for filename, coords in tile_centers.items():
        city = filename.split('_')[0].lower()
        if city not in city_groups:
            city_groups[city] = {
                'filenames': [],
                'xy': [],
                'xyz': [],
            }
        city_groups[city]['filenames'].append(filename)
        city_groups[city]['xy'].append([coords[0], coords[1]]) # 2D ground coordinates
        city_groups[city]['xyz'].append(coords)

    print(f"\n[2/4] Discovered {len(city_groups)} distinct cities:")
    for city, data in sorted(city_groups.items()):
        print(f"  -> {city.capitalize()}: {len(data['filenames']):,} tiles")

    # 3. Build KD-Tree per city and query K nearest neighbors
    print(f"\n[3/4] Building 2D KD-Trees and querying K={k_neighbors} nearest neighbors...")
    neighbor_map = {}
    all_neighbor_distances = []

    for city, data in sorted(city_groups.items()):
        filenames = data['filenames']
        xy_coords = np.array(data['xy'], dtype=np.float64)

        if len(filenames) <= k_neighbors:
            raise ValueError(
                f"City '{city}' has only {len(filenames)} tiles, which is <= k_neighbors ({k_neighbors})."
            )

        # Build 2D cKDTree on (X, Y)
        tree = cKDTree(xy_coords)

        # Query k_neighbors + 1 (since the first neighbor at distance 0 is the tile itself)
        distances, indices = tree.query(xy_coords, k=k_neighbors + 1)

        for i, target_fn in enumerate(filenames):
            # Exclude self (index 0) and take the K nearest neighbors
            neighbor_indices = indices[i][1:k_neighbors + 1]
            neighbor_dists   = distances[i][1:k_neighbors + 1]

            neighbor_fns = [filenames[idx] for idx in neighbor_indices]
            neighbor_map[target_fn] = neighbor_fns
            all_neighbor_distances.extend(neighbor_dists.tolist())

    # 4. Save neighbor map to JSON
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(neighbor_map, f, indent=2)

    print(f"\n[4/4] Saved neighbor index mapping to: {os.path.abspath(output_path)}")

    # ==========================================
    # SANITY CHECKS & VERIFICATION
    # ==========================================
    print('\n' + '=' * 70)
    print('  SANITY CHECKS & VALIDATION')
    print('=' * 70)

    # Check 1: Total tiles count matches input
    assert len(neighbor_map) == total_tiles, (
        f"Mismatch in total tiles: expected {total_tiles}, got {len(neighbor_map)}"
    )
    print(f"✓ Check 1: Total indexed tiles = {len(neighbor_map):,} (100% matched)")

    # Check 2: Every tile has EXACTLY K neighbors
    all_k_correct = all(len(neighbors) == k_neighbors for neighbors in neighbor_map.values())
    assert all_k_correct, f"Error: Some tiles do not have exactly {k_neighbors} neighbors!"
    print(f"✓ Check 2: Every tile has EXACTLY {k_neighbors} neighbors")

    # Check 3: No tile is its own neighbor
    no_self_neighbor = all(target not in neighbors for target, neighbors in neighbor_map.items())
    assert no_self_neighbor, "Error: Self-neighbor detected in neighbor mapping!"
    print(f"✓ Check 3: No tile is mapped as its own neighbor (self-exclusion verified)")

    # Check 4: Cross-city isolation
    no_cross_city = all(
        target.split('_')[0].lower() == nb.split('_')[0].lower()
        for target, neighbors in neighbor_map.items()
        for nb in neighbors
    )
    assert no_cross_city, "Error: Cross-city neighbor matching detected!"
    print(f"✓ Check 4: 100% city isolation (Birmingham & Cambridge never mixed)")

    # Distance statistics
    mean_dist = np.mean(all_neighbor_distances)
    min_dist  = np.min(all_neighbor_distances)
    max_dist  = np.max(all_neighbor_distances)
    print(f"\nNeighbor Distance Summary (in meters):")
    print(f"  Min Distance  : {min_dist:.2f} m")
    print(f"  Mean Distance : {mean_dist:.2f} m")
    print(f"  Max Distance  : {max_dist:.2f} m")
    print('=' * 70)
    print("✓ ALL SANITY CHECKS PASSED SUCCESSFULLY!\n")

    return neighbor_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build 2D Spatial KD-Tree Neighbor Map for SensatUrban Tiles.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/tile_centroids.json',
        help='Path to input tile centers/centroids JSON (default: datasets/tile_centers.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/neighbor_map.json',
        help='Path to save neighbor map JSON (default: datasets/neighbor_map.json)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        help='Number of nearest neighbors to retrieve per tile (default: 4)'
    )
    args = parser.parse_args()

    input_file = resolve_input_path(args.input)
    build_neighbor_map(
        input_path=input_file,
        output_path=args.output,
        k_neighbors=args.k,
    )
