"""
Sliding-window dataset over preprocessed trajectory .npz files.
All trajectories are pre-loaded into RAM at init (~160 MB for 50K x 200-point trajectories).
"""

import json
import os
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def _load_all_trajectories(trajectory_ids: list, trajectories_dir: str,
                           startup_pad: int = 0):
    """Load all trajectories into memory. Returns dict of traj_id -> (inductance, positions).

    Args:
        startup_pad: Number of origin-rest samples to prepend. Set to window_size - 1
            so the model has a full window from the very first real timestep.
            The pad uses position=(0,0,...) and the inductance measured at the origin
            (first point of each trajectory), simulating a sensor at rest before motion.
    """
    data = {}
    for i, traj_id in enumerate(trajectory_ids):
        path = os.path.join(trajectories_dir, f"{traj_id}.npz")
        npz = np.load(path)
        inductance = npz["inductance"]
        positions = npz["positions"]

        if startup_pad > 0:
            # Pad with origin: position=0, inductance=value at origin (first point)
            origin_ind = np.full(startup_pad, inductance[0], dtype=inductance.dtype)
            origin_pos = np.zeros((startup_pad, positions.shape[1]), dtype=positions.dtype)
            inductance = np.concatenate([origin_ind, inductance])
            positions = np.concatenate([origin_pos, positions])

        data[traj_id] = (inductance, positions)
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{len(trajectory_ids)} trajectories")
    return data


class TrajectoryWindowDataset(Dataset):
    """
    Sliding-window dataset with all data pre-loaded in RAM.

    Input:  (window_size, 1) — normalized inductance
    Target: (3,)             — normalized (x, y, z) at end of window
    """

    def __init__(
        self,
        trajectory_ids: list,
        traj_data: dict,
        window_size: int,
        stride: int,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
        subsample_factor: int = 1,
    ):
        self.trajectory_ids = trajectory_ids
        self.traj_data = traj_data
        self.window_size = window_size
        self.stride = stride
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.subsample_factor = subsample_factor

        self._build_index()

    def _build_index(self):
        """Build cumulative index: global_idx -> (trajectory_idx, window_start)."""
        self._cumulative = [0]

        for traj_id in self.trajectory_ids:
            inductance, _ = self.traj_data[traj_id]
            length = len(inductance)
            n_windows = max(0, (length - self.window_size) // self.stride + 1)
            n_windows = max(0, (n_windows + self.subsample_factor - 1) // self.subsample_factor)
            self._cumulative.append(self._cumulative[-1] + n_windows)

        self._total_windows = self._cumulative[-1]

    def __len__(self):
        return self._total_windows

    def __getitem__(self, idx):
        traj_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[traj_idx]

        traj_id = self.trajectory_ids[traj_idx]
        inductance, positions = self.traj_data[traj_id]

        window_start = local_idx * self.subsample_factor * self.stride
        window_end = window_start + self.window_size

        # Extract and normalize inductance
        inp = inductance[window_start:window_end].reshape(-1, 1)
        inp = self.input_scaler.transform(inp).astype(np.float32)

        target_pos = positions[window_end - 1].reshape(1, -1)
        target = self.target_scaler.transform(target_pos).flatten().astype(np.float32)

        return torch.from_numpy(inp), torch.from_numpy(target)


class ARTrajectoryWindowDataset(Dataset):
    """
    Autoregressive sliding-window dataset.

    Returns (input, target) where:
        input:  (window_size, 4) — [normalized_inductance, normalized_prev_x, prev_y, prev_z]
        target: (3,)             — normalized (x, y, z) at end of window

    The position channels are shifted by 1 timestep relative to inductance:
        inductance: [t-W+1, ..., t]
        positions:  [t-W,   ..., t-1]   (previous positions)

    Windows start at index 1 (not 0) so there's always a previous position available.
    """

    def __init__(
        self,
        trajectory_ids: list,
        traj_data: dict,
        window_size: int,
        stride: int,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
        subsample_factor: int = 1,
    ):
        self.trajectory_ids = trajectory_ids
        self.traj_data = traj_data
        self.window_size = window_size
        self.stride = stride
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.subsample_factor = subsample_factor

        self._build_index()

    def _build_index(self):
        self._cumulative = [0]

        for traj_id in self.trajectory_ids:
            inductance, _ = self.traj_data[traj_id]
            length = len(inductance)
            # Need 1 extra point before the window for the first previous position
            n_windows = max(0, (length - self.window_size - 1) // self.stride + 1)
            n_windows = max(0, (n_windows + self.subsample_factor - 1) // self.subsample_factor)
            self._cumulative.append(self._cumulative[-1] + n_windows)

        self._total_windows = self._cumulative[-1]

    def __len__(self):
        return self._total_windows

    def __getitem__(self, idx):
        traj_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[traj_idx]

        traj_id = self.trajectory_ids[traj_idx]
        inductance, positions = self.traj_data[traj_id]

        # Window starts at index 1+ so we can look back 1 step for positions
        window_start = 1 + local_idx * self.subsample_factor * self.stride
        window_end = window_start + self.window_size

        # Inductance: [window_start, ..., window_end-1]
        ind = inductance[window_start:window_end].reshape(-1, 1)
        ind_norm = self.input_scaler.transform(ind).astype(np.float32)

        # Previous positions: [window_start-1, ..., window_end-2] (shifted by 1)
        prev_pos = positions[window_start - 1:window_end - 1]
        prev_pos_norm = self.target_scaler.transform(prev_pos).astype(np.float32)

        # Concatenate: (window_size, 4)
        inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)

        # Target: position at window_end-1
        target_pos = positions[window_end - 1].reshape(1, -1)
        target = self.target_scaler.transform(target_pos).flatten().astype(np.float32)

        return torch.from_numpy(inp), torch.from_numpy(target)


class ARMultistepDataset(Dataset):
    """
    Multi-step autoregressive dataset for unrolled training.

    Returns (input, future_ind, targets) where:
        input:      (window_size, 1+D) — same initial window as ARTrajectoryWindowDataset
        future_ind: (K-1, 1)           — normalized inductance for shifting the window
        targets:    (K, D)             — normalized position targets for each unroll step

    K = unroll_steps. Step 0 target is the same as ARTrajectoryWindowDataset's target.
    Steps 1..K-1 shift the window forward, requiring future inductance values.
    """

    def __init__(
        self,
        trajectory_ids: list,
        traj_data: dict,
        window_size: int,
        stride: int,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
        subsample_factor: int = 1,
        unroll_steps: int = 10,
    ):
        self.trajectory_ids = trajectory_ids
        self.traj_data = traj_data
        self.window_size = window_size
        self.stride = stride
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.subsample_factor = subsample_factor
        self.unroll_steps = unroll_steps

        self._build_index()

    def _build_index(self):
        self._cumulative = [0]

        for traj_id in self.trajectory_ids:
            inductance, _ = self.traj_data[traj_id]
            length = len(inductance)
            # Need: 1 (position lag) + window_size + (unroll_steps - 1) <= length
            usable = length - self.window_size - 1 - (self.unroll_steps - 1)
            n_windows = max(0, usable // self.stride + 1)
            n_windows = max(0, (n_windows + self.subsample_factor - 1) // self.subsample_factor)
            self._cumulative.append(self._cumulative[-1] + n_windows)

        self._total_windows = self._cumulative[-1]

    def __len__(self):
        return self._total_windows

    def __getitem__(self, idx):
        traj_idx = bisect.bisect_right(self._cumulative, idx) - 1
        local_idx = idx - self._cumulative[traj_idx]

        traj_id = self.trajectory_ids[traj_idx]
        inductance, positions = self.traj_data[traj_id]
        K = self.unroll_steps
        W = self.window_size

        # Same starting logic as ARTrajectoryWindowDataset
        window_start = 1 + local_idx * self.subsample_factor * self.stride
        window_end = window_start + W

        # Initial window (same as single-step)
        ind = inductance[window_start:window_end].reshape(-1, 1)
        ind_norm = self.input_scaler.transform(ind).astype(np.float32)

        prev_pos = positions[window_start - 1:window_end - 1]
        prev_pos_norm = self.target_scaler.transform(prev_pos).astype(np.float32)

        inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)  # (W, 1+D)

        # Future inductance for shifting the window (K-1 values)
        if K > 1:
            fut_ind = inductance[window_end:window_end + K - 1].reshape(-1, 1)
            fut_ind_norm = self.input_scaler.transform(fut_ind).astype(np.float32)
        else:
            fut_ind_norm = np.empty((0, 1), dtype=np.float32)

        # K targets: positions at window_end-1, window_end, ..., window_end+K-2
        target_positions = positions[window_end - 1:window_end - 1 + K]
        targets = self.target_scaler.transform(target_positions).astype(np.float32)  # (K, D)

        return (
            torch.from_numpy(inp),
            torch.from_numpy(fut_ind_norm),
            torch.from_numpy(targets),
        )


def fit_scalers(trajectory_ids: list, traj_data: dict):
    """Fit StandardScalers on the given trajectories from pre-loaded data."""
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()

    for traj_id in trajectory_ids:
        inductance, positions = traj_data[traj_id]
        input_scaler.partial_fit(inductance.reshape(-1, 1))
        target_scaler.partial_fit(positions)

    return input_scaler, target_scaler


def create_data_splits(
    trajectories_dir: str,
    window_size: int = 20,
    stride: int = 1,
    subsample_factor: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 512,
    num_workers: int = 4,
    seed: int = 42,
    startup_pad: bool = True,
) -> dict:
    """
    Load all data into RAM, split into train/val/test, fit scalers, create DataLoaders.

    Args:
        startup_pad: If True, prepend window_size-1 origin-rest samples to each trajectory
            so the model has a full window from the very first real timestep.

    Returns dict with keys:
        train_loader, val_loader, test_loader,
        input_scaler, target_scaler,
        train_ids, val_ids, test_ids
    """
    manifest_path = os.path.join(trajectories_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ids = sorted(manifest.keys())
    print(f"Total trajectories: {len(all_ids)}")

    # Load everything into RAM
    pad_n = window_size - 1 if startup_pad else 0
    if pad_n > 0:
        print(f"  Startup padding: {pad_n} origin-rest samples per trajectory")
    print("Loading all trajectories into memory...")
    traj_data = _load_all_trajectories(all_ids, trajectories_dir, startup_pad=pad_n)
    total_points = sum(len(v[0]) for v in traj_data.values())
    mem_mb = total_points * (1 + 3) * 4 / (1024 * 1024)  # float32
    print(f"  {total_points:,} points loaded ({mem_mb:.0f} MB)")

    # Split: first separate test, then split remainder into train/val
    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=seed)
    relative_val = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=relative_val, random_state=seed)

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Fit scalers on training data only
    print("Fitting scalers on training data...")
    input_scaler, target_scaler = fit_scalers(train_ids, traj_data)

    # Create datasets (all share the same traj_data dict — no duplication)
    train_ds = TrajectoryWindowDataset(
        train_ids, traj_data, window_size, stride,
        input_scaler, target_scaler, subsample_factor,
    )
    val_ds = TrajectoryWindowDataset(
        val_ids, traj_data, window_size, stride,
        input_scaler, target_scaler, subsample_factor,
    )
    test_ds = TrajectoryWindowDataset(
        test_ids, traj_data, window_size, stride,
        input_scaler, target_scaler, subsample_factor,
    )

    print(f"  Train windows: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    # num_workers=0 since data is already in RAM — no I/O bottleneck
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "traj_data": traj_data,
    }


def create_ar_data_splits(
    trajectories_dir: str,
    window_size: int = 20,
    stride: int = 1,
    subsample_factor: int = 1,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 512,
    num_workers: int = 4,
    seed: int = 42,
    unroll_steps: int = 1,
    startup_pad: bool = True,
) -> dict:
    """
    Same as create_data_splits but uses ARTrajectoryWindowDataset.
    Training data includes previous ground-truth positions (teacher forcing).

    When unroll_steps > 1, training set uses ARMultistepDataset for multi-step
    unrolled training. Val/test still use ARTrajectoryWindowDataset (validation
    uses AR rollout via validate_ar, not the dataset).

    Args:
        startup_pad: If True, prepend window_size origin-rest samples to each trajectory
            (window_size, not window_size-1, because AR needs 1 extra for prev position).
    """
    manifest_path = os.path.join(trajectories_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_ids = sorted(manifest.keys())
    print(f"Total trajectories: {len(all_ids)}")

    # AR needs 1 extra point before the window for the previous position
    pad_n = window_size if startup_pad else 0
    if pad_n > 0:
        print(f"  Startup padding: {pad_n} origin-rest samples per trajectory")
    print("Loading all trajectories into memory...")
    traj_data = _load_all_trajectories(all_ids, trajectories_dir, startup_pad=pad_n)
    total_points = sum(len(v[0]) for v in traj_data.values())
    mem_mb = total_points * (1 + 3) * 4 / (1024 * 1024)
    print(f"  {total_points:,} points loaded ({mem_mb:.0f} MB)")

    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_size, random_state=seed)
    relative_val = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=relative_val, random_state=seed)

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    print("Fitting scalers on training data...")
    input_scaler, target_scaler = fit_scalers(train_ids, traj_data)

    if unroll_steps > 1:
        print(f"  Using multi-step dataset (unroll_steps={unroll_steps})")
        train_ds = ARMultistepDataset(
            train_ids, traj_data, window_size, stride,
            input_scaler, target_scaler, subsample_factor,
            unroll_steps=unroll_steps,
        )
    else:
        train_ds = ARTrajectoryWindowDataset(
            train_ids, traj_data, window_size, stride,
            input_scaler, target_scaler, subsample_factor,
        )
    val_ds = ARTrajectoryWindowDataset(
        val_ids, traj_data, window_size, stride,
        input_scaler, target_scaler, subsample_factor,
    )
    test_ds = ARTrajectoryWindowDataset(
        test_ids, traj_data, window_size, stride,
        input_scaler, target_scaler, subsample_factor,
    )

    print(f"  Train windows: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_scaler": input_scaler,
        "target_scaler": target_scaler,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "traj_data": traj_data,
    }
