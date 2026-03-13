"""
Sanity check: compare inductance profiles across trajectories of similar length.
Shows whether the inductance manifold carries meaningful positional information.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data/constant_v_traj")

# Load manifest
with open(DATA_DIR / "manifest.json") as f:
    manifest = json.load(f)

# Group trajectories by length
by_length = defaultdict(list)
for traj_id, n_steps in manifest.items():
    by_length[n_steps].append(traj_id)

print(f"Total trajectories: {len(manifest)}")
print(f"Unique lengths: {len(by_length)}")
print(f"\nLength distribution (top 10 most common):")
for length, ids in sorted(by_length.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  {length} steps: {len(ids)} trajectories")

# Pick a common length for detailed comparison
common_lengths = sorted(by_length.items(), key=lambda x: -len(x[1]))
target_length = common_lengths[0][0]
target_ids = common_lengths[0][1]
print(f"\n--- Analyzing {len(target_ids)} trajectories with {target_length} steps ---")

# Load all trajectories of this length
trajs = {}
for tid in target_ids[:200]:  # cap at 200 for speed
    data = np.load(DATA_DIR / f"{tid}.npz")
    trajs[tid] = {
        "inductance": data["inductance"],
        "positions": data["positions"],
    }

# Compute pairwise inductance differences for a sample
sample_ids = list(trajs.keys())[:50]
n = len(sample_ids)

# Stack inductance profiles
L_matrix = np.array([trajs[tid]["inductance"] for tid in sample_ids])  # (n, steps)
pos_matrix = np.array([trajs[tid]["positions"] for tid in sample_ids])  # (n, steps, 2)

# Compute trajectory-level stats
L_range = L_matrix.max(axis=1) - L_matrix.min(axis=1)
print(f"\nInductance range per trajectory: mean={L_range.mean():.4f}, std={L_range.std():.4f}")
print(f"  min range: {L_range.min():.4f}, max range: {L_range.max():.4f}")

# Pairwise: for each pair of trajectories, compute the max |L_i - L_j| across timesteps
# and the max position difference
print(f"\nPairwise comparison of {n} trajectories (same length={target_length}):")
l_diffs = []
pos_diffs = []
for i in range(n):
    for j in range(i + 1, n):
        l_diff = np.abs(L_matrix[i] - L_matrix[j])
        p_diff = np.linalg.norm(pos_matrix[i] - pos_matrix[j], axis=1)
        l_diffs.append(l_diff)
        pos_diffs.append(p_diff)

l_diffs = np.array(l_diffs)  # (n_pairs, steps)
pos_diffs = np.array(pos_diffs)

# Per-timestep: correlation between position difference and inductance difference
print("\nPer-timestep correlation (position distance vs inductance difference):")
correlations = []
for t in range(target_length):
    corr = np.corrcoef(pos_diffs[:, t], l_diffs[:, t])[0, 1]
    correlations.append(corr)
correlations = np.array(correlations)
print(f"  Mean correlation: {np.nanmean(correlations):.3f}")
print(f"  Min: {np.nanmin(correlations):.3f}, Max: {np.nanmax(correlations):.3f}")

# --- Plots ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Inductance Manifold Analysis — {target_length}-step trajectories", fontsize=14)

# 1. Overlay inductance profiles
ax = axes[0, 0]
for tid in sample_ids[:30]:
    ax.plot(trajs[tid]["inductance"], alpha=0.4, linewidth=0.8)
ax.set_xlabel("Timestep")
ax.set_ylabel("Inductance (L)")
ax.set_title(f"Inductance profiles (30 of {len(target_ids)} trajectories)")

# 2. Overlay trajectories in XY space, colored by mean inductance
ax = axes[0, 1]
mean_L = [trajs[tid]["inductance"].mean() for tid in sample_ids[:30]]
norm = plt.Normalize(min(mean_L), max(mean_L))
cmap = plt.cm.viridis
for tid, ml in zip(sample_ids[:30], mean_L):
    pos = trajs[tid]["positions"]
    ax.plot(pos[:, 0], pos[:, 1], color=cmap(norm(ml)), alpha=0.6, linewidth=1)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title("Trajectories colored by mean L")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(sm, ax=ax, label="Mean L")

# 3. Scatter: position distance vs inductance difference (all timesteps pooled)
ax = axes[0, 2]
# Subsample for plotting
idx = np.random.choice(l_diffs.size, min(5000, l_diffs.size), replace=False)
ax.scatter(pos_diffs.flatten()[idx], l_diffs.flatten()[idx], alpha=0.15, s=3)
ax.set_xlabel("Position distance (mm)")
ax.set_ylabel("|ΔL|")
ax.set_title("Position distance vs inductance difference")

# 4. Correlation over timesteps
ax = axes[1, 0]
ax.plot(correlations, 'o-', markersize=3)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("Timestep")
ax.set_ylabel("Pearson correlation")
ax.set_title("Corr(position dist, |ΔL|) per timestep")

# 5. Pick two trajectories that are spatially close vs far and overlay their L profiles
# Find closest and farthest pairs
mean_pos = pos_matrix.mean(axis=1)  # (n, 2)
pair_dists = []
for i in range(n):
    for j in range(i + 1, n):
        d = np.linalg.norm(mean_pos[i] - mean_pos[j])
        pair_dists.append((d, i, j))
pair_dists.sort()

ax = axes[1, 1]
# Close pair
_, ci, cj = pair_dists[0]
ax.plot(L_matrix[ci], label=f"{sample_ids[ci]} (close)", linewidth=1.5)
ax.plot(L_matrix[cj], label=f"{sample_ids[cj]} (close)", linewidth=1.5, linestyle='--')
# Far pair
_, fi, fj = pair_dists[-1]
ax.plot(L_matrix[fi], label=f"{sample_ids[fi]} (far)", linewidth=1.5)
ax.plot(L_matrix[fj], label=f"{sample_ids[fj]} (far)", linewidth=1.5, linestyle='--')
ax.set_xlabel("Timestep")
ax.set_ylabel("Inductance")
ax.set_title("Close vs far trajectory pairs")
ax.legend(fontsize=7)

# 6. Inductance at same timestep vs position — show L is position-dependent
ax = axes[1, 2]
mid_t = target_length // 2
sc = ax.scatter(pos_matrix[:, mid_t, 0], pos_matrix[:, mid_t, 1],
                c=L_matrix[:, mid_t], cmap='viridis', s=15, alpha=0.7)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title(f"Position at t={mid_t}, colored by L")
plt.colorbar(sc, ax=ax, label="L")

plt.tight_layout()
plt.savefig("output/inductance_analysis.png", dpi=150)
print(f"\nPlot saved to output/inductance_analysis.png")
plt.show()
