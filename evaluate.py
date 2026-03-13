"""
Test-set evaluation: euclidean error, per-component MAE, accuracy at thresholds.
"""

import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute position prediction metrics. All values in mm (denormalized).

    Args:
        predictions: (N, D) — predicted positions (D=2 or D=3)
        targets: (N, D) — actual positions

    Returns:
        dict of metric name -> value
    """
    errors = predictions - targets
    euclidean = np.sqrt(np.sum(errors ** 2, axis=1))
    output_dim = predictions.shape[1]

    metrics = {
        "output_dim": output_dim,
        "euclidean_mean": float(np.mean(euclidean)),
        "euclidean_median": float(np.median(euclidean)),
        "euclidean_std": float(np.std(euclidean)),
        "euclidean_p90": float(np.percentile(euclidean, 90)),
        "euclidean_p95": float(np.percentile(euclidean, 95)),
        "euclidean_max": float(np.max(euclidean)),
        "x_mae": float(np.mean(np.abs(errors[:, 0]))),
        "y_mae": float(np.mean(np.abs(errors[:, 1]))),
        "acc_0p1mm": float(np.mean(euclidean < 0.1) * 100),
        "acc_0p25mm": float(np.mean(euclidean < 0.25) * 100),
        "acc_0p5mm": float(np.mean(euclidean < 0.5) * 100),
        "acc_1mm": float(np.mean(euclidean < 1.0) * 100),
    }
    if output_dim >= 3:
        metrics["z_mae"] = float(np.mean(np.abs(errors[:, 2])))
    return metrics


@torch.no_grad()
def evaluate_model(model, test_loader, target_scaler: StandardScaler, device) -> dict:
    """
    Run model on test set, inverse-transform predictions, compute metrics.

    Returns dict with 'metrics', 'predictions', 'targets'.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.numpy())

    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)

    # Denormalize to mm
    predictions = target_scaler.inverse_transform(preds_norm)
    targets = target_scaler.inverse_transform(targets_norm)

    metrics = compute_metrics(predictions, targets)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
    }


@torch.no_grad()
def evaluate_model_ar(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, predict_displacement: bool = False,
) -> dict:
    """
    Evaluate AR model with true autoregressive rollout per trajectory.

    For each test trajectory:
      1. Initialize position history with ground truth for the first window
      2. Predict autoregressively, feeding back own predictions
      3. Collect predictions and ground truth for metrics

    Args:
        predict_displacement: if True, model outputs delta (normalized) instead
            of absolute position. Reconstruction: pos = prev_pos + delta.

    Returns dict with 'metrics', 'predictions', 'targets'.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for traj_id in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        # Initialize with ground truth positions for the first window
        pos_history = positions[:window_size].copy()

        for start in range(1, len(inductance) - window_size + 1):
            # Inductance window
            ind = inductance[start:start + window_size].reshape(-1, 1)
            ind_norm = input_scaler.transform(ind).astype(np.float32)

            # Previous positions (autoregressive — uses own predictions)
            prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)

            inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)

            out = model(inp_t).cpu().numpy().flatten()

            if predict_displacement:
                # out is normalized delta; reconstruct absolute position
                prev_norm = prev_pos_norm[-1]  # last position, already normalized
                pred_pos_norm = prev_norm + out
                pred_pos = target_scaler.inverse_transform(
                    pred_pos_norm.reshape(1, -1)
                ).flatten()
            else:
                pred_pos = target_scaler.inverse_transform(
                    out.reshape(1, -1)
                ).flatten()

            all_preds.append(pred_pos)
            all_targets.append(positions[start + window_size - 1])

            # Shift history: drop oldest, append new prediction
            pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

    predictions = np.array(all_preds)
    targets = np.array(all_targets)
    metrics = compute_metrics(predictions, targets)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
    }


def print_metrics(metrics: dict):
    output_dim = metrics.get("output_dim", 3)
    print("\n=== Test Set Metrics ===")
    print(f"  Euclidean error:")
    print(f"    Mean:   {metrics['euclidean_mean']:.4f} mm")
    print(f"    Median: {metrics['euclidean_median']:.4f} mm")
    print(f"    Std:    {metrics['euclidean_std']:.4f} mm")
    print(f"    P90:    {metrics['euclidean_p90']:.4f} mm")
    print(f"    P95:    {metrics['euclidean_p95']:.4f} mm")
    print(f"    Max:    {metrics['euclidean_max']:.4f} mm")
    print(f"  Per-component MAE:")
    print(f"    X: {metrics['x_mae']:.4f} mm")
    print(f"    Y: {metrics['y_mae']:.4f} mm")
    if output_dim >= 3:
        print(f"    Z: {metrics['z_mae']:.4f} mm")
    print(f"  Accuracy:")
    print(f"    < 0.10 mm: {metrics['acc_0p1mm']:.1f}%")
    print(f"    < 0.25 mm: {metrics['acc_0p25mm']:.1f}%")
    print(f"    < 0.50 mm: {metrics['acc_0p5mm']:.1f}%")
    print(f"    < 1.00 mm: {metrics['acc_1mm']:.1f}%")
