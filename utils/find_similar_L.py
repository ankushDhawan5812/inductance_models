"""
Find trajectory pairs with similar inductance profiles but different XY paths.
Prints top pairs and generates comparison plots.

Usage:
    python find_similar_L.py
    python find_similar_L.py --n_sample 500 --top_k 10
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d


DATA_DIR = Path("data/constant_v_traj")


def resample(arr, n):
    """Resample a 1D or 2D array to n points using linear interpolation."""
    old_t = np.linspace(0, 1, len(arr))
    new_t = np.linspace(0, 1, n)
    if arr.ndim == 1:
        return interp1d(old_t, arr)(new_t)
    return np.column_stack([interp1d(old_t, arr[:, i])(new_t) for i in range(arr.shape[1])])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/constant_v_traj")
    parser.add_argument("--n_sample", type=int, default=300, help="Number of trajectories to compare")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top pairs to show")
    parser.add_argument("--resample_n", type=int, default=50, help="Resample length for comparison")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Sample trajectories
    all_ids = list(manifest.keys())
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(all_ids, min(args.n_sample, len(all_ids)), replace=False)

    # Load and resample to common length
    n = args.resample_n
    L_resampled = []
    pos_resampled = []
    valid_ids = []
    for tid in sample_ids:
        data = np.load(data_dir / f"{tid}.npz")
        L = data["inductance"]
        pos = data["positions"]
        if len(L) < 5:
            continue
        L_resampled.append(resample(L, n))
        pos_resampled.append(resample(pos, n))
        valid_ids.append(tid)

    L_mat = np.array(L_resampled)    # (N, n)
    pos_mat = np.array(pos_resampled)  # (N, n, 2)
    N = len(valid_ids)

    print(f"Comparing {N} trajectories (resampled to {n} points each)")

    # Compute pairwise distances (only between trajectories of similar length)
    orig_lengths = np.array([len(np.load(data_dir / f"{tid}.npz")["inductance"]) for tid in valid_ids])
    length_tol = 0.1  # only compare trajectories within +/- 10% length

    print("Computing pairwise distances (similar-length pairs only)...")
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            len_i, len_j = orig_lengths[i], orig_lengths[j]
            if abs(len_i - len_j) > length_tol * max(len_i, len_j):
                continue
            l_rmse = np.sqrt(np.mean((L_mat[i] - L_mat[j]) ** 2))
            pos_dist = np.mean(np.linalg.norm(pos_mat[i] - pos_mat[j], axis=1))
            pairs.append((l_rmse, pos_dist, i, j))

    print(f"  {len(pairs)} similar-length pairs (out of {N*(N-1)//2} total)")

    pairs = np.array(pairs, dtype=[('l_rmse', float), ('pos_dist', float), ('i', int), ('j', int)])

    # Compute L RMSE percentiles for binning
    all_l_rmse = pairs['l_rmse']
    all_pos_dist = pairs['pos_dist']
    p25, p50, p75 = np.percentile(all_l_rmse, [25, 50, 75])
    print(f"\nL RMSE distribution: p25={p25:.6f}, p50={p50:.6f}, p75={p75:.6f}")
    print(f"  min={all_l_rmse.min():.6f}, max={all_l_rmse.max():.6f}")

    def pick_diverse(candidates, k):
        """Greedily pick k pairs where no trajectory ID appears more than once."""
        candidates_sorted = candidates.copy()
        candidates_sorted.sort(order='pos_dist')
        candidates_sorted = candidates_sorted[::-1]  # highest pos_dist first
        selected = []
        used_ids = set()
        for p in candidates_sorted:
            i, j = int(p['i']), int(p['j'])
            if i not in used_ids and j not in used_ids:
                selected.append(p)
                used_ids.update([i, j])
                if len(selected) >= k:
                    break
        return np.array(selected, dtype=pairs.dtype)

    # Three categories:
    # 1. Similar L, different XY (small L RMSE, large pos dist)
    # 2. Moderate L difference, different XY (median L RMSE, large pos dist)
    # 3. Large L difference, different XY (large L RMSE, large pos dist)
    categories = {}

    # Similar: bottom 20% L RMSE
    n_bin = max(100, len(pairs) // 5)
    pairs.sort(order='l_rmse')
    categories["Similar L, Different XY"] = pick_diverse(pairs[:n_bin], args.top_k)

    # Moderate: middle 20% L RMSE (40th-60th percentile)
    p40_idx = int(0.4 * len(pairs))
    p60_idx = int(0.6 * len(pairs))
    categories["Moderate L Difference, Different XY"] = pick_diverse(pairs[p40_idx:p60_idx], args.top_k)

    # Large: top 20% L RMSE
    categories["Large L Difference, Different XY"] = pick_diverse(pairs[-n_bin:], args.top_k)

    for cat_name, cat_pairs in categories.items():
        print(f"\n--- {cat_name} ---")
        print(f"{'Pair':>35s}  {'L RMSE':>10s}  {'Mean XY dist':>12s}")
        print("-" * 65)
        for p in cat_pairs:
            tid_i = valid_ids[int(p['i'])]
            tid_j = valid_ids[int(p['j'])]
            print(f"{tid_i} vs {tid_j}  {p['l_rmse']:10.6f}  {p['pos_dist']:10.3f} mm")

    # Pre-compute global axis limits across all plotted pairs
    all_plot_pairs = []
    for cat_pairs in categories.values():
        for p in cat_pairs[:min(args.top_k, 3)]:
            all_plot_pairs.append(p)

    xy_min, xy_max = np.inf, -np.inf
    L_min, L_max = np.inf, -np.inf
    max_timesteps = 0
    for p in all_plot_pairs:
        i, j = int(p['i']), int(p['j'])
        for idx in [i, j]:
            tid = valid_ids[idx]
            d = np.load(data_dir / f"{tid}.npz")
            pos = d["positions"]
            L = d["inductance"]
            xy_min = min(xy_min, pos.min())
            xy_max = max(xy_max, pos.max())
            L_min = min(L_min, L.min())
            L_max = max(L_max, L.max())
            max_timesteps = max(max_timesteps, len(L))

    # Add padding
    xy_pad = (xy_max - xy_min) * 0.05
    L_pad = (L_max - L_min) * 0.05

    # Plot each category as a separate figure
    k = min(args.top_k, 3)  # rows per figure
    cat_filenames = {
        "Similar L, Different XY": "L_comparison_similar.png",
        "Moderate L Difference, Different XY": "L_comparison_moderate.png",
        "Large L Difference, Different XY": "L_comparison_large.png",
    }

    for cat_name, cat_pairs in categories.items():
        fig, axes = plt.subplots(k, 3, figsize=(16, 4 * k))
        if k == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(cat_name, fontsize=14, y=1.01)

        for pair_idx, p in enumerate(cat_pairs[:k]):
            i, j = int(p['i']), int(p['j'])
            tid_i, tid_j = valid_ids[i], valid_ids[j]

            d1 = np.load(data_dir / f"{tid_i}.npz")
            d2 = np.load(data_dir / f"{tid_j}.npz")

            # XY paths
            ax = axes[pair_idx, 0]
            ax.plot(d1["positions"][:, 0], d1["positions"][:, 1], 'b.-', markersize=3, label=tid_i)
            ax.plot(d2["positions"][:, 0], d2["positions"][:, 1], 'r.-', markersize=3, label=tid_j)
            ax.plot(0, 0, 'ko', markersize=5)
            ax.set_xlim(xy_min - xy_pad, xy_max + xy_pad)
            ax.set_ylim(xy_min - xy_pad, xy_max + xy_pad)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"XY paths (dist: {p['pos_dist']:.2f} mm)", fontsize=9)

            # Inductance profiles
            ax = axes[pair_idx, 1]
            ax.plot(d1["inductance"], 'b.-', markersize=3, label=tid_i)
            ax.plot(d2["inductance"], 'r.-', markersize=3, label=tid_j)
            ax.set_xlim(0, max_timesteps)
            ax.set_ylim(L_min - L_pad, L_max + L_pad)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("L")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Inductance (RMSE: {p['l_rmse']:.6f})", fontsize=9)

            # Resampled overlay
            ax = axes[pair_idx, 2]
            ax.plot(L_mat[i], 'b-', alpha=0.8, label=f"{tid_i}")
            ax.plot(L_mat[j], 'r-', alpha=0.8, label=f"{tid_j}")
            ax.fill_between(range(len(L_mat[i])),
                            L_mat[i], L_mat[j], alpha=0.2, color='gray')
            ax.set_ylim(L_min - L_pad, L_max + L_pad)
            ax.set_xlabel("Normalized timestep")
            ax.set_ylabel("L")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title("Resampled L overlay", fontsize=9)

        plt.tight_layout()
        out_path = f"output/{cat_filenames[cat_name]}"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
