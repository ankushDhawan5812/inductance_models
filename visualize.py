"""
Visualization: training curves, predicted vs actual, error distribution, trajectory samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler


def plot_training_history(history: dict, save_path: str):
    """Training/val loss curves + LR schedule."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(len(history["train_loss"]))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val (short)")
    if "val_loss_full" in history and history["val_loss_full"]:
        axes[0].plot(epochs, history["val_loss_full"], label="Val (full rollout)", alpha=0.6)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(epochs, history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(predictions: np.ndarray, targets: np.ndarray, metrics: dict, save_path: str):
    """Scatter: pred vs actual for each component (2 or 3 panels)."""
    output_dim = predictions.shape[1]
    labels = ["X (mm)", "Y (mm)", "Z (mm)"][:output_dim]
    fig, axes = plt.subplots(1, output_dim, figsize=(5 * output_dim, 5))
    if output_dim == 1:
        axes = [axes]

    n = min(20000, len(predictions))
    idx = np.random.default_rng(42).choice(len(predictions), n, replace=False)

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.scatter(targets[idx, i], predictions[idx, i], s=1, alpha=0.3)
        ax.plot([-8, 8], [-8, 8], "r--", linewidth=1)
        ax.set_xlabel(f"Actual {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect("equal")

    fig.suptitle(
        f"Predicted vs Actual (mean err={metrics['euclidean_mean']:.3f} mm, "
        f"median={metrics['euclidean_median']:.3f} mm)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(predictions: np.ndarray, targets: np.ndarray, save_path: str):
    """Euclidean error histogram + per-component histograms."""
    output_dim = predictions.shape[1]
    errors = predictions - targets
    euclidean = np.sqrt(np.sum(errors ** 2, axis=1))
    component_labels = ["X", "Y", "Z"][:output_dim]

    fig, axes = plt.subplots(1, 1 + output_dim, figsize=(4.5 * (1 + output_dim), 4))

    axes[0].hist(euclidean, bins=100, alpha=0.7)
    axes[0].axvline(np.median(euclidean), color="r", linestyle="--", label=f"median={np.median(euclidean):.3f}")
    axes[0].set_xlabel("Euclidean Error (mm)")
    axes[0].set_title("Euclidean Error")
    axes[0].legend()

    for i, (ax, label) in enumerate(zip(axes[1:], component_labels)):
        ax.hist(errors[:, i], bins=100, alpha=0.7)
        ax.axvline(0, color="r", linestyle="--")
        ax.set_xlabel(f"{label} Error (mm)")
        ax.set_title(f"{label} Component Error")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(predictions: np.ndarray, targets: np.ndarray, save_path: str):
    """Target positions colored by error magnitude."""
    output_dim = predictions.shape[1]
    euclidean = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))

    n = min(20000, len(predictions))
    idx = np.random.default_rng(42).choice(len(predictions), n, replace=False)

    n_panels = 1 if output_dim == 2 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # XY view
    sc = axes[0].scatter(targets[idx, 0], targets[idx, 1], c=euclidean[idx], s=1, cmap="hot", alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="Error (mm)")
    axes[0].set_xlabel("X (mm)")
    axes[0].set_ylabel("Y (mm)")
    axes[0].set_title("Error vs Position (XY)")
    axes[0].set_xlim(-8, 8)
    axes[0].set_ylim(-8, 8)
    axes[0].set_aspect("equal")

    # XZ view (only for 3D)
    if output_dim >= 3:
        sc = axes[1].scatter(targets[idx, 0], targets[idx, 2], c=euclidean[idx], s=1, cmap="hot", alpha=0.5)
        plt.colorbar(sc, ax=axes[1], label="Error (mm)")
        axes[1].set_xlabel("X (mm)")
        axes[1].set_ylabel("Z (mm)")
        axes[1].set_title("Error vs Position (XZ)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_sample_trajectories(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, save_path: str, n_trajectories: int = 6,
    gp_predictor=None,
):
    """Sliding-window predictions on full test trajectories, actual vs predicted."""
    model.eval()
    rng = np.random.default_rng(42)
    selected = rng.choice(test_ids, size=min(n_trajectories, len(test_ids)), replace=False)

    # Detect dimensionality from first trajectory
    sample_data = np.load(os.path.join(trajectories_dir, f"{selected[0]}.npz"))
    output_dim = sample_data["positions"].shape[1]

    # Build manifold surface once if GP available
    manifold = None
    if gp_predictor is not None:
        manifold = _build_manifold_surface(gp_predictor)

    has_3d_col = gp_predictor is not None
    n_cols = (1 if output_dim == 2 else 2) + 1 + (1 if has_3d_col else 0)
    n_rows = len(selected)
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for row, traj_id in enumerate(selected):
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        # Run sliding window predictions
        preds = []
        for start in range(len(inductance) - window_size + 1):
            window = inductance[start:start + window_size].reshape(-1, 1)
            window_norm = input_scaler.transform(window).astype(np.float32)
            inp = torch.from_numpy(window_norm).unsqueeze(0).to(device)
            out = model(inp).cpu().numpy()
            pred_pos = target_scaler.inverse_transform(out).flatten()
            preds.append(pred_pos)

        preds = np.array(preds)
        actual = positions[window_size - 1:]

        col_idx = 0

        # XY view
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        ax.plot(actual[:, 0], actual[:, 1], "b-", alpha=0.5, label="Actual")
        ax.plot(preds[:, 0], preds[:, 1], "r-", alpha=0.5, label="Predicted")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{traj_id} (XY)")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        col_idx += 1

        # XZ view (only for 3D)
        if output_dim >= 3:
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
            ax.plot(actual[:, 0], actual[:, 2], "b-", alpha=0.5, label="Actual")
            ax.plot(preds[:, 0], preds[:, 2], "r-", alpha=0.5, label="Predicted")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Z (mm)")
            ax.set_title(f"{traj_id} (XZ)")
            ax.legend(fontsize=7)
            col_idx += 1

        # Inductance profile: ground truth + predicted (via GP)
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        actual_L = inductance[window_size - 1:]
        ax.plot(actual_L, "b-", alpha=0.7, label="Actual L")
        pred_L = None
        if gp_predictor is not None:
            pred_L, _ = gp_predictor.predict(preds[:, :2])
            ax.plot(pred_L, "r-", alpha=0.7, label="Predicted L")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Inductance (L)")
        ax.set_title(f"{traj_id} — Inductance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        col_idx += 1

        # 3D manifold view
        if has_3d_col and manifold is not None:
            Xg, Yg, Zg = manifold
            ax3d = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1,
                                   projection='3d')
            ax3d.plot_surface(Xg, Yg, Zg, alpha=0.25, cmap='viridis',
                              edgecolor='none', rcount=30, ccount=30)
            ax3d.plot(actual[:, 0], actual[:, 1], actual_L,
                      'b-', linewidth=2, label="Actual", alpha=0.8)
            if pred_L is not None:
                ax3d.plot(preds[:, 0], preds[:, 1], pred_L,
                          'r-', linewidth=2, label="Predicted", alpha=0.8)
            ax3d.set_xlabel("X", fontsize=7)
            ax3d.set_ylabel("Y", fontsize=7)
            ax3d.set_zlabel("L", fontsize=7)
            ax3d.set_title(f"{traj_id} — Manifold", fontsize=9)
            ax3d.legend(fontsize=6)
            ax3d.tick_params(labelsize=6)

    fig.suptitle("Sample Trajectory Predictions", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_manifold_surface(gp_predictor, grid_n=40, bound=8.0):
    """Build a grid of inductance values for the 3D manifold surface."""
    xs = np.linspace(-bound, bound, grid_n)
    ys = np.linspace(-bound, bound, grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    query = np.column_stack([Xg.ravel(), Yg.ravel()])
    Zg, _ = gp_predictor.predict(query)
    return Xg, Yg, Zg.reshape(Xg.shape)


@torch.no_grad()
def plot_sample_trajectories_ar(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, save_path: str, n_trajectories: int = 6,
    predict_displacement: bool = False, gp_predictor=None,
):
    """AR sliding-window predictions: feed back predicted positions autoregressively."""
    model.eval()
    rng = np.random.default_rng(42)
    selected = rng.choice(test_ids, size=min(n_trajectories, len(test_ids)), replace=False)

    # Detect dimensionality
    sample_data = np.load(os.path.join(trajectories_dir, f"{selected[0]}.npz"))
    output_dim = sample_data["positions"].shape[1]

    # Build manifold surface once if GP available
    manifold = None
    if gp_predictor is not None:
        manifold = _build_manifold_surface(gp_predictor)

    has_3d_col = gp_predictor is not None
    n_cols = (1 if output_dim == 2 else 2) + 1 + (1 if has_3d_col else 0)
    n_rows = len(selected)
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for row, traj_id in enumerate(selected):
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        preds = []
        pos_history = positions[:window_size].copy()

        for start in range(1, len(inductance) - window_size + 1):
            ind = inductance[start:start + window_size].reshape(-1, 1)
            ind_norm = input_scaler.transform(ind).astype(np.float32)
            prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)
            inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t).cpu().numpy().flatten()

            if predict_displacement:
                prev_norm = prev_pos_norm[-1]
                pred_pos_norm = prev_norm + out
                pred_pos = target_scaler.inverse_transform(
                    pred_pos_norm.reshape(1, -1)
                ).flatten()
            else:
                pred_pos = target_scaler.inverse_transform(
                    out.reshape(1, -1)
                ).flatten()

            preds.append(pred_pos)
            pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

        preds = np.array(preds)
        actual = positions[window_size:]

        col_idx = 0

        # XY view
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        ax.plot(actual[:, 0], actual[:, 1], "b-", alpha=0.5, label="Actual")
        ax.plot(preds[:, 0], preds[:, 1], "r-", alpha=0.5, label="Predicted")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{traj_id} (XY) — AR")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        col_idx += 1

        # XZ view (only for 3D)
        if output_dim >= 3:
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
            ax.plot(actual[:, 0], actual[:, 2], "b-", alpha=0.5, label="Actual")
            ax.plot(preds[:, 0], preds[:, 2], "r-", alpha=0.5, label="Predicted")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Z (mm)")
            ax.set_title(f"{traj_id} (XZ) — AR")
            ax.legend(fontsize=7)
            col_idx += 1

        # Inductance profile: ground truth + predicted (via GP)
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        actual_L = inductance[window_size:]
        ax.plot(actual_L, "b-", alpha=0.7, label="Actual L")
        pred_L = None
        if gp_predictor is not None:
            pred_L, _ = gp_predictor.predict(preds[:, :2])
            ax.plot(pred_L, "r-", alpha=0.7, label="Predicted L")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Inductance (L)")
        ax.set_title(f"{traj_id} — Inductance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        col_idx += 1

        # 3D manifold view
        if has_3d_col and manifold is not None:
            Xg, Yg, Zg = manifold
            ax3d = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1,
                                   projection='3d')
            ax3d.plot_surface(Xg, Yg, Zg, alpha=0.25, cmap='viridis',
                              edgecolor='none', rcount=30, ccount=30)
            ax3d.plot(actual[:, 0], actual[:, 1], actual_L,
                      'b-', linewidth=2, label="Actual", alpha=0.8)
            if pred_L is not None:
                ax3d.plot(preds[:, 0], preds[:, 1], pred_L,
                          'r-', linewidth=2, label="Predicted", alpha=0.8)
            ax3d.set_xlabel("X", fontsize=7)
            ax3d.set_ylabel("Y", fontsize=7)
            ax3d.set_zlabel("L", fontsize=7)
            ax3d.set_title(f"{traj_id} — Manifold", fontsize=9)
            ax3d.legend(fontsize=6)
            ax3d.tick_params(labelsize=6)

    fig.suptitle("Sample AR Trajectory Predictions", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_worst_trajectories_ar(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, save_path: str, n_worst: int = 5,
    predict_displacement: bool = False, gp_predictor=None,
):
    """Plot the worst-performing test trajectories by mean euclidean error."""
    model.eval()

    # Score every test trajectory
    traj_errors = []
    for traj_id in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        preds = []
        pos_history = positions[:window_size].copy()

        for start in range(1, len(inductance) - window_size + 1):
            ind = inductance[start:start + window_size].reshape(-1, 1)
            ind_norm = input_scaler.transform(ind).astype(np.float32)
            prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)
            inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t).cpu().numpy().flatten()

            if predict_displacement:
                prev_norm = prev_pos_norm[-1]
                pred_pos_norm = prev_norm + out
                pred_pos = target_scaler.inverse_transform(
                    pred_pos_norm.reshape(1, -1)
                ).flatten()
            else:
                pred_pos = target_scaler.inverse_transform(
                    out.reshape(1, -1)
                ).flatten()

            preds.append(pred_pos)
            pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

        preds = np.array(preds)
        actual = positions[window_size:]
        if len(preds) == 0:
            continue
        mean_err = np.mean(np.sqrt(np.sum((preds - actual) ** 2, axis=1)))
        max_err = np.max(np.sqrt(np.sum((preds - actual) ** 2, axis=1)))
        traj_errors.append((traj_id, mean_err, max_err, preds, actual, inductance))

    # Sort by mean error descending, pick worst N
    traj_errors.sort(key=lambda x: x[1], reverse=True)
    worst = traj_errors[:n_worst]

    output_dim = worst[0][3].shape[1]

    # Build manifold surface once if GP available
    manifold = None
    if gp_predictor is not None:
        manifold = _build_manifold_surface(gp_predictor)

    has_3d_col = gp_predictor is not None
    n_cols = (1 if output_dim == 2 else 2) + 1 + (1 if has_3d_col else 0)
    n_rows = len(worst)
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for row, (traj_id, mean_err, max_err, preds, actual, inductance) in enumerate(worst):
        col_idx = 0

        # XY view
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        ax.plot(actual[:, 0], actual[:, 1], "b-", alpha=0.5, label="Actual")
        ax.plot(preds[:, 0], preds[:, 1], "r-", alpha=0.5, label="Predicted")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(
            f"#{row+1} {traj_id} — mean={mean_err:.3f} mm, max={max_err:.3f} mm"
        )
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        col_idx += 1

        if output_dim >= 3:
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
            ax.plot(actual[:, 0], actual[:, 2], "b-", alpha=0.5, label="Actual")
            ax.plot(preds[:, 0], preds[:, 2], "r-", alpha=0.5, label="Predicted")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Z (mm)")
            ax.set_title(
                f"#{row+1} {traj_id} (XZ) — mean={mean_err:.3f} mm"
            )
            ax.legend(fontsize=7)
            col_idx += 1

        # Inductance profile: ground truth + predicted (via GP)
        ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1)
        actual_L = inductance[window_size:]
        ax.plot(actual_L, "b-", alpha=0.7, label="Actual L")
        pred_L = None
        if gp_predictor is not None:
            pred_L, _ = gp_predictor.predict(preds[:, :2])
            ax.plot(pred_L, "r-", alpha=0.7, label="Predicted L")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Inductance (L)")
        ax.set_title(f"{traj_id} — Inductance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        col_idx += 1

        # 3D manifold view
        if has_3d_col and manifold is not None:
            Xg, Yg, Zg = manifold
            ax3d = fig.add_subplot(n_rows, n_cols, row * n_cols + col_idx + 1,
                                   projection='3d')
            ax3d.plot_surface(Xg, Yg, Zg, alpha=0.25, cmap='viridis',
                              edgecolor='none', rcount=30, ccount=30)
            ax3d.plot(actual[:, 0], actual[:, 1], actual_L,
                      'b-', linewidth=2, label="Actual", alpha=0.8)
            if pred_L is not None:
                ax3d.plot(preds[:, 0], preds[:, 1], pred_L,
                          'r-', linewidth=2, label="Predicted", alpha=0.8)
            ax3d.set_xlabel("X", fontsize=7)
            ax3d.set_ylabel("Y", fontsize=7)
            ax3d.set_zlabel("L", fontsize=7)
            ax3d.set_title(f"{traj_id} — Manifold", fontsize=9)
            ax3d.legend(fontsize=6)
            ax3d.tick_params(labelsize=6)

    fig.suptitle("Worst AR Trajectory Predictions", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_worst_trajectories_ar_interactive(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, save_path: str, n_worst: int = 6,
    predict_displacement: bool = False, gp_predictor=None,
):
    """Interactive HTML version of worst trajectory plots using Plotly."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    model.eval()

    # Score every test trajectory
    traj_errors = []
    for traj_id in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        preds = []
        pos_history = positions[:window_size].copy()

        for start in range(1, len(inductance) - window_size + 1):
            ind = inductance[start:start + window_size].reshape(-1, 1)
            ind_norm = input_scaler.transform(ind).astype(np.float32)
            prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)
            inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t).cpu().numpy().flatten()

            if predict_displacement:
                prev_norm = prev_pos_norm[-1]
                pred_pos_norm = prev_norm + out
                pred_pos = target_scaler.inverse_transform(
                    pred_pos_norm.reshape(1, -1)
                ).flatten()
            else:
                pred_pos = target_scaler.inverse_transform(
                    out.reshape(1, -1)
                ).flatten()

            preds.append(pred_pos)
            pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

        preds = np.array(preds)
        actual = positions[window_size:]
        mean_err = np.mean(np.sqrt(np.sum((preds - actual) ** 2, axis=1)))
        max_err = np.max(np.sqrt(np.sum((preds - actual) ** 2, axis=1)))
        traj_errors.append((traj_id, mean_err, max_err, preds, actual, inductance))

    traj_errors.sort(key=lambda x: x[1], reverse=True)
    worst = traj_errors[:n_worst]

    # Build manifold surface
    manifold = None
    if gp_predictor is not None:
        manifold = _build_manifold_surface(gp_predictor, grid_n=60)

    n_rows = len(worst)
    has_3d = gp_predictor is not None

    # Build subplot specs: XY (xy), Inductance (xy), 3D manifold (scene)
    col_specs = [{"type": "xy"}, {"type": "xy"}]
    col_titles_base = ["XY Path", "Inductance"]
    if has_3d:
        col_specs.append({"type": "scene"})
        col_titles_base.append("3D Manifold")
    n_cols = len(col_specs)

    subplot_titles = []
    for row_i, (traj_id, mean_err, max_err, _, _, _) in enumerate(worst):
        subplot_titles.append(f"#{row_i+1} {traj_id} — mean={mean_err:.3f}mm, max={max_err:.3f}mm")
        subplot_titles.append(f"{traj_id} — Inductance")
        if has_3d:
            subplot_titles.append(f"{traj_id} — Manifold")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=[col_specs] * n_rows,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.06,
    )

    for row_i, (traj_id, mean_err, max_err, preds, actual, inductance) in enumerate(worst):
        row = row_i + 1
        show_legend = (row_i == 0)

        # Col 1: XY paths
        fig.add_trace(go.Scatter(
            x=actual[:, 0].tolist(), y=actual[:, 1].tolist(), mode='lines',
            line=dict(color='blue', width=2), name='Actual',
            legendgroup='actual', showlegend=show_legend,
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra>Actual</extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=preds[:, 0].tolist(), y=preds[:, 1].tolist(), mode='lines',
            line=dict(color='red', width=2), name='Predicted',
            legendgroup='predicted', showlegend=show_legend,
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra>Predicted</extra>",
        ), row=row, col=1)

        # Col 2: Inductance
        actual_L = inductance[window_size:]
        timesteps = list(range(len(actual_L)))
        fig.add_trace(go.Scatter(
            x=timesteps, y=actual_L.tolist(), mode='lines',
            line=dict(color='blue', width=2), name='Actual L',
            legendgroup='actual_L', showlegend=show_legend,
            hovertemplate="t: %{x}<br>L: %{y:.6f}<extra>Actual L</extra>",
        ), row=row, col=2)
        pred_L = None
        if gp_predictor is not None:
            pred_L, _ = gp_predictor.predict(preds[:, :2])
            fig.add_trace(go.Scatter(
                x=timesteps[:len(pred_L)], y=pred_L.tolist(), mode='lines',
                line=dict(color='red', width=2), name='Predicted L',
                legendgroup='predicted_L', showlegend=show_legend,
                hovertemplate="t: %{x}<br>L: %{y:.6f}<extra>Predicted L</extra>",
            ), row=row, col=2)

        # Col 3: 3D manifold
        if has_3d and manifold is not None:
            Xg, Yg, Zg = manifold
            fig.add_trace(go.Surface(
                x=Xg.tolist(), y=Yg.tolist(), z=Zg.tolist(),
                colorscale='Viridis', opacity=0.35,
                showscale=False, name="Manifold",
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>L: %{z:.6f}<extra></extra>",
                showlegend=False,
            ), row=row, col=3)
            fig.add_trace(go.Scatter3d(
                x=actual[:, 0].tolist(), y=actual[:, 1].tolist(), z=actual_L.tolist(),
                mode='lines', line=dict(color='blue', width=4),
                name='Actual', legendgroup='actual', showlegend=False,
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>L: %{z:.6f}<extra>Actual</extra>",
            ), row=row, col=3)
            if pred_L is not None:
                fig.add_trace(go.Scatter3d(
                    x=preds[:, 0].tolist(), y=preds[:, 1].tolist(), z=pred_L.tolist(),
                    mode='lines', line=dict(color='red', width=4),
                    name='Predicted', legendgroup='predicted', showlegend=False,
                    hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>L: %{z:.6f}<extra>Predicted</extra>",
                ), row=row, col=3)

    # Style axes using subplot references
    for row_i in range(n_rows):
        row = row_i + 1
        # Get the actual axis references from the subplot grid
        xy_xref = fig.get_subplot(row, 1).xaxis.plotly_name  # e.g. "xaxis", "xaxis3"
        xy_yref = fig.get_subplot(row, 1).yaxis.plotly_name
        fig.update_layout(**{
            xy_xref: dict(range=[-8, 8], scaleanchor=xy_yref.replace("axis", ""), scaleratio=1),
            xy_yref: dict(range=[-8, 8]),
        })

    # Style 3D scenes
    if has_3d:
        for row_i in range(n_rows):
            scene_key = f"scene{row_i + 1}" if row_i > 0 else "scene"
            fig.update_layout(**{
                scene_key: dict(
                    xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="L",
                    aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5),
                ),
            })

    fig.update_layout(
        title_text="Worst AR Trajectory Predictions (Interactive)",
        height=400 * n_rows,
        width=500 * n_cols,
        legend=dict(x=1.0, y=1.0),
    )

    fig.write_html(save_path, include_plotlyjs=True)
    print(f"Saved interactive plot: {save_path}")


@torch.no_grad()
def plot_error_vs_step_ar(
    model, test_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, save_path: str,
    predict_displacement: bool = False,
    unroll_steps: int = 0,
):
    """Plot mean euclidean error vs rollout step, averaged across test trajectories.

    Shows where errors accumulate and whether there's a jump at the training horizon.
    """
    model.eval()
    # Collect per-step errors across all test trajectories
    step_errors = {}  # step_index -> list of euclidean errors

    for traj_id in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        pos_history = positions[:window_size].copy()

        for step, start in enumerate(range(1, len(inductance) - window_size + 1)):
            ind = inductance[start:start + window_size].reshape(-1, 1)
            ind_norm = input_scaler.transform(ind).astype(np.float32)
            prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)
            inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
            out = model(inp_t).cpu().numpy().flatten()

            if predict_displacement:
                prev_norm = prev_pos_norm[-1]
                pred_pos_norm = prev_norm + out
                pred_pos = target_scaler.inverse_transform(
                    pred_pos_norm.reshape(1, -1)
                ).flatten()
            else:
                pred_pos = target_scaler.inverse_transform(
                    out.reshape(1, -1)
                ).flatten()

            actual_pos = positions[start + window_size - 1]
            err = np.sqrt(np.sum((pred_pos - actual_pos) ** 2))

            if step not in step_errors:
                step_errors[step] = []
            step_errors[step].append(err)

            pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

    # Compute statistics per step
    max_step = max(step_errors.keys())
    steps = np.arange(max_step + 1)
    mean_errs = np.array([np.mean(step_errors[s]) for s in steps])
    p25 = np.array([np.percentile(step_errors[s], 25) for s in steps])
    p75 = np.array([np.percentile(step_errors[s], 75) for s in steps])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, mean_errs, "b-", linewidth=1.5, label="Mean error")
    ax.fill_between(steps, p25, p75, alpha=0.2, color="blue", label="25th-75th percentile")

    if unroll_steps > 0:
        ax.axvline(unroll_steps, color="red", linestyle="--", alpha=0.7,
                   label=f"Training horizon ({unroll_steps} steps)")
    if window_size > 0:
        ax.axvline(window_size, color="orange", linestyle=":", alpha=0.7,
                   label=f"GT fully flushed ({window_size} steps)")

    ax.set_xlabel("Rollout Step")
    ax.set_ylabel("Euclidean Error (mm)")
    ax.set_title("Error vs Rollout Step (AR)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def generate_all_plots(history, eval_results, model, test_ids,
                       trajectories_dir, input_scaler, target_scaler,
                       window_size, device, output_dir, is_ar=False,
                       predict_displacement=False, unroll_steps=0,
                       gp_predictor=None):
    """Generate all visualization plots."""
    predictions = eval_results["predictions"]
    targets = eval_results["targets"]
    metrics = eval_results["metrics"]

    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    print("  Saved training_history.png")

    plot_predicted_vs_actual(predictions, targets, metrics, os.path.join(output_dir, "pred_vs_actual.png"))
    print("  Saved pred_vs_actual.png")

    plot_error_distribution(predictions, targets, os.path.join(output_dir, "error_distribution.png"))
    print("  Saved error_distribution.png")

    plot_error_heatmap(predictions, targets, os.path.join(output_dir, "error_heatmap.png"))
    print("  Saved error_heatmap.png")

    if is_ar:
        plot_sample_trajectories_ar(
            model, test_ids, trajectories_dir,
            input_scaler, target_scaler, window_size, device,
            os.path.join(output_dir, "trajectory_samples.png"),
            predict_displacement=predict_displacement,
            gp_predictor=gp_predictor,
        )
    else:
        plot_sample_trajectories(
            model, test_ids, trajectories_dir,
            input_scaler, target_scaler, window_size, device,
            os.path.join(output_dir, "trajectory_samples.png"),
            gp_predictor=gp_predictor,
        )
    print("  Saved trajectory_samples.png")

    if is_ar:
        plot_worst_trajectories_ar(
            model, test_ids, trajectories_dir,
            input_scaler, target_scaler, window_size, device,
            os.path.join(output_dir, "worst_trajectories.png"),
            predict_displacement=predict_displacement,
            gp_predictor=gp_predictor,
        )
        print("  Saved worst_trajectories.png")

        plot_error_vs_step_ar(
            model, test_ids, trajectories_dir,
            input_scaler, target_scaler, window_size, device,
            os.path.join(output_dir, "error_vs_step.png"),
            predict_displacement=predict_displacement,
            unroll_steps=unroll_steps,
        )
        print("  Saved error_vs_step.png")
