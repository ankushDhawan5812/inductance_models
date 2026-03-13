"""
Preprocess the large trajectory CSV into per-trajectory .npz files for fast loading.

Usage:
    python preprocess.py --csv_path ../gp_fit/trajectories/all_trajectories.csv
    python preprocess.py --csv_path PATH --output_dir data/trajectories
"""

import argparse
import json
import os
import numpy as np
import pandas as pd


def preprocess(csv_path: str, output_dir: str, chunk_size: int = 2_000_000):
    os.makedirs(output_dir, exist_ok=True)

    manifest = {}
    buffer = {}  # trajectory_id -> list of DataFrames
    n_saved = 0

    # Detect dimensionality from CSV header
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    pos_cols = ["x", "y", "z"] if "z" in header else ["x", "y"]
    print(f"  Position columns: {pos_cols}")

    def flush_trajectory(traj_id: str):
        nonlocal n_saved
        df = pd.concat(buffer.pop(traj_id), ignore_index=True)
        df = df.sort_values("timestep")
        inductance = df["predicted_inductance"].values.astype(np.float32)
        positions = df[pos_cols].values.astype(np.float32)
        np.savez_compressed(
            os.path.join(output_dir, f"{traj_id}.npz"),
            inductance=inductance,
            positions=positions,
        )
        manifest[traj_id] = len(df)
        n_saved += 1
        if n_saved % 5000 == 0:
            print(f"  Saved {n_saved} trajectories...")

    print(f"Reading {csv_path} in chunks of {chunk_size} rows...")
    active_ids = set()

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        chunk_ids = set(chunk["trajectory_id"].unique())

        # Group this chunk by trajectory_id
        for traj_id, group in chunk.groupby("trajectory_id"):
            if traj_id not in buffer:
                buffer[traj_id] = []
            buffer[traj_id].append(group)

        # Flush any trajectories that were in the previous chunk but not in this one
        # (meaning they are complete)
        finished = active_ids - chunk_ids
        for traj_id in finished:
            if traj_id in buffer:
                flush_trajectory(traj_id)

        active_ids = chunk_ids
        print(f"  Processed chunk {chunk_idx + 1} ({(chunk_idx + 1) * chunk_size / 1e6:.0f}M rows), "
              f"{len(buffer)} trajectories buffered")

    # Flush remaining
    for traj_id in list(buffer.keys()):
        flush_trajectory(traj_id)

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    print(f"\nDone! Saved {n_saved} trajectories to {output_dir}/")
    print(f"  Manifest: {manifest_path}")
    total_points = sum(manifest.values())
    print(f"  Total points: {total_points:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="../gp_fit/trajectories/all_trajectories.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Defaults to data/<csv_stem> (e.g. data/constant_v_traj)")
    parser.add_argument("--chunk_size", type=int, default=2_000_000)
    args = parser.parse_args()

    if args.output_dir is None:
        csv_stem = os.path.splitext(os.path.basename(args.csv_path))[0]
        args.output_dir = os.path.join("data", csv_stem)

    preprocess(args.csv_path, args.output_dir, args.chunk_size)
