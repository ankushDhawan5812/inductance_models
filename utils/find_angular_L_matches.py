"""
For trajectories at regular angular intervals (every pi/12), find the most
L-similar trajectory that is at least pi/4 away in angle.

Usage:
    python find_angular_L_matches.py
    python find_angular_L_matches.py --data_dir data/constant_v_traj --n_sample 1000
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d


def resample(arr, n):
    old_t = np.linspace(0, 1, len(arr))
    new_t = np.linspace(0, 1, n)
    if arr.ndim == 1:
        return interp1d(old_t, arr)(new_t)
    return np.column_stack([interp1d(old_t, arr[:, i])(new_t) for i in range(arr.shape[1])])


def trajectory_angle(positions):
    """Compute the dominant angle of a trajectory from its endpoint relative to origin."""
    endpoint = positions[-1]
    return np.arctan2(endpoint[1], endpoint[0])


def angular_distance(a1, a2):
    """Shortest angular distance between two angles."""
    diff = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/constant_v_traj")
    parser.add_argument("--n_sample", type=int, default=500)
    parser.add_argument("--resample_n", type=int, default=50)
    parser.add_argument("--length_tol", type=float, default=0.15,
                        help="Only match trajectories within this fraction of length")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    all_ids = list(manifest.keys())
    rng = np.random.RandomState(42)
    sample_ids = rng.choice(all_ids, min(args.n_sample, len(all_ids)), replace=False)

    # Load all sampled trajectories
    n = args.resample_n
    trajs = {}
    for tid in sample_ids:
        data = np.load(data_dir / f"{tid}.npz")
        L = data["inductance"]
        pos = data["positions"]
        if len(L) < 5:
            continue
        trajs[tid] = {
            "L_raw": L,
            "pos_raw": pos,
            "L_resampled": resample(L, n),
            "pos_resampled": resample(pos, n),
            "angle": trajectory_angle(pos),
            "orig_len": len(L),
        }

    tids = list(trajs.keys())
    angles = np.array([trajs[tid]["angle"] for tid in tids])
    print(f"Loaded {len(tids)} trajectories")

    # Pick reference trajectories at every pi/12 (15 degrees)
    angle_step = np.pi / 12
    min_angular_sep = np.pi / 16  # match must be at least 45 (22.5) degrees away

    ref_angles = np.arange(-np.pi, np.pi, angle_step)
    matches = []

    for ref_angle in ref_angles:
        # Find trajectory closest to this reference angle
        angle_diffs = np.array([angular_distance(angles[i], ref_angle) for i in range(len(tids))])
        ref_idx = np.argmin(angle_diffs)
        ref_tid = tids[ref_idx]
        ref_data = trajs[ref_tid]

        # Find most L-similar trajectory that's >= pi/4 away and similar length
        best_rmse = np.inf
        best_tid = None
        for j, tid in enumerate(tids):
            if tid == ref_tid:
                continue
            # Check angular separation
            if angular_distance(angles[j], angles[ref_idx]) < min_angular_sep:
                continue
            # Check length similarity
            len_ref = ref_data["orig_len"]
            len_j = trajs[tid]["orig_len"]
            if abs(len_ref - len_j) > args.length_tol * max(len_ref, len_j):
                continue
            # Compute L RMSE
            rmse = np.sqrt(np.mean((ref_data["L_resampled"] - trajs[tid]["L_resampled"]) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_tid = tid

        if best_tid is not None:
            match_angle = trajs[best_tid]["angle"]
            pos_dist = np.mean(np.linalg.norm(
                ref_data["pos_resampled"] - trajs[best_tid]["pos_resampled"], axis=1))
            matches.append({
                "ref_tid": ref_tid,
                "ref_angle": np.degrees(ref_data["angle"]),
                "match_tid": best_tid,
                "match_angle": np.degrees(match_angle),
                "angular_sep": np.degrees(angular_distance(ref_data["angle"], match_angle)),
                "l_rmse": best_rmse,
                "pos_dist": pos_dist,
            })

    print(f"\nFound {len(matches)} matches (every {np.degrees(angle_step):.0f} deg, "
          f"match >= {np.degrees(min_angular_sep):.0f} deg away)")
    print(f"\n{'Ref angle':>10s}  {'Match angle':>11s}  {'Sep':>6s}  {'L RMSE':>10s}  {'XY dist':>8s}  Pair")
    print("-" * 85)
    for m in matches:
        print(f"{m['ref_angle']:>9.1f}°  {m['match_angle']:>10.1f}°  {m['angular_sep']:>5.1f}°  "
              f"{m['l_rmse']:10.6f}  {m['pos_dist']:7.2f}mm  "
              f"{m['ref_tid']} vs {m['match_tid']}")

    # --- Plots ---
    # Global axis limits
    xy_min, xy_max = np.inf, -np.inf
    L_min, L_max = np.inf, -np.inf
    max_ts = 0
    for m in matches:
        for tid in [m["ref_tid"], m["match_tid"]]:
            d = trajs[tid]
            xy_min = min(xy_min, d["pos_raw"].min())
            xy_max = max(xy_max, d["pos_raw"].max())
            L_min = min(L_min, d["L_raw"].min())
            L_max = max(L_max, d["L_raw"].max())
            max_ts = max(max_ts, len(d["L_raw"]))
    xy_pad = (xy_max - xy_min) * 0.05
    L_pad = (L_max - L_min) * 0.05

    # Overview plot: polar showing all reference angles and their matches
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
    for m in matches:
        ref_r = 1.0
        match_r = 1.0
        ref_a = np.radians(m["ref_angle"])
        match_a = np.radians(m["match_angle"])
        ax.annotate("", xy=(match_a, match_r), xytext=(ref_a, ref_r),
                     arrowprops=dict(arrowstyle="->", color='red', alpha=0.5, lw=1.5))
        ax.plot(ref_a, ref_r, 'bo', markersize=6)
        ax.plot(match_a, match_r, 'r^', markersize=6)
    ax.set_title(f"Angular matches: blue = reference, red = closest L match\n"
                 f"(must be >= {np.degrees(min_angular_sep):.0f} deg apart)", pad=20)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("output/angular_L_overview_11_25.png", dpi=150)
    print(f"\nSaved: output/angular_L_overview_11_25.png")
    plt.close()

    # Detail plots: 4 per page
    n_pages = (len(matches) + 3) // 4
    for page in range(n_pages):
        page_matches = matches[page * 4:(page + 1) * 4]
        n_rows = len(page_matches)
        fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(f"Angular L Matches (page {page + 1}/{n_pages})", fontsize=14, y=1.01)

        for row, m in enumerate(page_matches):
            ref = trajs[m["ref_tid"]]
            match = trajs[m["match_tid"]]

            # XY paths
            ax = axes[row, 0]
            ax.plot(ref["pos_raw"][:, 0], ref["pos_raw"][:, 1], 'b.-', markersize=3,
                    label=f'{m["ref_tid"]} ({m["ref_angle"]:.0f}°)')
            ax.plot(match["pos_raw"][:, 0], match["pos_raw"][:, 1], 'r.-', markersize=3,
                    label=f'{m["match_tid"]} ({m["match_angle"]:.0f}°)')
            ax.plot(0, 0, 'ko', markersize=5)
            ax.set_xlim(xy_min - xy_pad, xy_max + xy_pad)
            ax.set_ylim(xy_min - xy_pad, xy_max + xy_pad)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_aspect("equal")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"XY paths (sep: {m['angular_sep']:.0f}°, dist: {m['pos_dist']:.1f}mm)", fontsize=9)

            # Raw inductance
            ax = axes[row, 1]
            ax.plot(ref["L_raw"], 'b.-', markersize=3, label=m["ref_tid"])
            ax.plot(match["L_raw"], 'r.-', markersize=3, label=m["match_tid"])
            ax.set_xlim(0, max_ts)
            ax.set_ylim(L_min - L_pad, L_max + L_pad)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("L")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"Inductance (RMSE: {m['l_rmse']:.6f})", fontsize=9)

            # Resampled overlay
            ax = axes[row, 2]
            ax.plot(ref["L_resampled"], 'b-', alpha=0.8, label=m["ref_tid"])
            ax.plot(match["L_resampled"], 'r-', alpha=0.8, label=m["match_tid"])
            ax.fill_between(range(n), ref["L_resampled"], match["L_resampled"],
                            alpha=0.2, color='gray')
            ax.set_ylim(L_min - L_pad, L_max + L_pad)
            ax.set_xlabel("Normalized timestep")
            ax.set_ylabel("L")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title("Resampled L overlay", fontsize=9)

        plt.tight_layout()
        out_path = f"output/angular_L_matches_page{page + 1}11_25.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
