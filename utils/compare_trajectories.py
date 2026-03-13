"""
Compare two trajectories side by side: XY path + inductance profile.

Usage:
    python compare_trajectories.py curved_00000 linear_00001
    python compare_trajectories.py curved_00000 curved_00001 --data_dir data/constant_v_traj
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_traj(data_dir, traj_id):
    data = np.load(Path(data_dir) / f"{traj_id}.npz")
    return data["inductance"], data["positions"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj1", help="First trajectory ID")
    parser.add_argument("traj2", help="Second trajectory ID")
    parser.add_argument("--data_dir", default="data/constant_v_traj")
    parser.add_argument("-o", "--output", default=None, help="Save path (default: show)")
    args = parser.parse_args()

    L1, pos1 = load_traj(args.data_dir, args.traj1)
    L2, pos2 = load_traj(args.data_dir, args.traj2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{args.traj1}  vs  {args.traj2}", fontsize=13)

    # 1. XY paths
    ax = axes[0]
    ax.plot(pos1[:, 0], pos1[:, 1], 'b.-', label=args.traj1, markersize=3, alpha=0.7)
    ax.plot(pos2[:, 0], pos2[:, 1], 'r.-', label=args.traj2, markersize=3, alpha=0.7)
    ax.plot(0, 0, 'ko', markersize=6, label='origin')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("XY Trajectories")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Inductance profiles
    ax = axes[1]
    ax.plot(L1, 'b.-', label=args.traj1, markersize=3)
    ax.plot(L2, 'r.-', label=args.traj2, markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Inductance (L)")
    ax.set_title("Inductance Profiles")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. If same length, show pointwise difference
    ax = axes[2]
    min_len = min(len(L1), len(L2))
    diff = L1[:min_len] - L2[:min_len]
    pos_dist = np.linalg.norm(pos1[:min_len] - pos2[:min_len], axis=1)
    ax.scatter(pos_dist, np.abs(diff), s=10, alpha=0.6)
    ax.set_xlabel("Position distance (mm)")
    ax.set_ylabel("|ΔL|")
    ax.set_title(f"Pointwise: pos dist vs |ΔL| (first {min_len} steps)")
    ax.grid(True, alpha=0.3)

    # Print stats
    print(f"\n{args.traj1}: {len(L1)} steps, L range [{L1.min():.6f}, {L1.max():.6f}]")
    print(f"{args.traj2}: {len(L2)} steps, L range [{L2.min():.6f}, {L2.max():.6f}]")
    print(f"\nPointwise (first {min_len} steps):")
    print(f"  Mean |ΔL|:          {np.abs(diff).mean():.6f}")
    print(f"  Max  |ΔL|:          {np.abs(diff).max():.6f}")
    print(f"  Mean pos distance:  {pos_dist.mean():.3f} mm")
    print(f"  Max  pos distance:  {pos_dist.max():.3f} mm")

    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
        print(f"\nSaved to {args.output}")
    else:
        plt.savefig("output/compare_trajectories.png", dpi=150)
        print(f"\nSaved to output/compare_trajectories.png")
    plt.show()


if __name__ == "__main__":
    main()
