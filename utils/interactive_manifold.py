"""
Interactive 3D manifold visualization with trajectory overlays using Plotly.

Opens in browser — rotate, zoom, pan freely.

Usage:
    python interactive_manifold.py output/2026-03-11_185701_transformer_ar_constant_v_traj \
        --checkpoint output/2026-03-10_144143_transformer_ar_constant_v_traj/checkpoints/best_model.pth \
        --gp_models_dir ../gp_fit_2d/models \
        --n_trajectories 4
"""

import argparse
import os
import sys
import joblib
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from model import PositionTransformer, CausalPositionTransformer


def build_manifold_surface(gp_predictor, grid_n=60, bound=8.0):
    xs = np.linspace(-bound, bound, grid_n)
    ys = np.linspace(-bound, bound, grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    query = np.column_stack([Xg.ravel(), Yg.ravel()])
    Zg, _ = gp_predictor.predict(query)
    return Xg, Yg, Zg.reshape(Xg.shape)


def run_ar_inference(model, inductance, positions, window_size, input_scaler, target_scaler, device):
    preds = []
    pos_history = positions[:window_size].copy()

    for start in range(1, len(inductance) - window_size + 1):
        ind = inductance[start:start + window_size].reshape(-1, 1)
        ind_norm = input_scaler.transform(ind).astype(np.float32)
        prev_pos_norm = target_scaler.transform(pos_history).astype(np.float32)
        inp = np.concatenate([ind_norm, prev_pos_norm], axis=1)
        inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        out = model(inp_t).cpu().numpy().flatten()
        pred_pos = target_scaler.inverse_transform(out.reshape(1, -1)).flatten()
        preds.append(pred_pos)
        pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

    return np.array(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory with config/scalers")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gp_models_dir", required=True)
    parser.add_argument("--n_trajectories", type=int, default=4)
    parser.add_argument("--grid_n", type=int, default=60)
    parser.add_argument("--output", type=str, default=None, help="Output HTML path (default: run_dir/manifold_interactive.html)")
    args = parser.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Load config
    config_path = os.path.join(args.run_dir, [f for f in os.listdir(args.run_dir) if f.endswith('.yaml')][0])
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_type = cfg["model"]["model_type"]
    is_ar = model_type.endswith("_ar")
    output_dim = cfg["model"].get("output_dim", 2)
    window_size = cfg["data"]["window_size"]
    trajectories_dir = cfg["data"]["trajectories_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    input_size = (1 + output_dim) if is_ar else 1
    if "causal" in model_type:
        model = CausalPositionTransformer(
            input_size=input_size, d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            encoder_dims=cfg["model"]["encoder_dims"],
            decoder_dims=cfg["model"]["decoder_dims"],
            dropout=cfg["model"]["dropout"], max_len=window_size + 10,
            output_dim=output_dim,
        ).to(device)
    else:
        model = PositionTransformer(
            input_size=input_size, d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"], num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            fc_dims=cfg["model"]["fc_dims"], dropout=cfg["model"]["dropout"],
            max_len=window_size + 10, output_dim=output_dim,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    input_scaler = joblib.load(os.path.join(args.run_dir, "input_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(args.run_dir, "target_scaler.pkl"))

    # Load GP predictor
    gp_parent = os.path.dirname(os.path.abspath(args.gp_models_dir))
    sys.path.insert(0, gp_parent)
    from patched_predictor import PatchedGPPredictor
    from patch_config import PatchConfig
    from patch_manager import PatchManager
    gp_config = PatchConfig()
    gp_manager = PatchManager(gp_config, models_dir=args.gp_models_dir)
    gp_predictor = PatchedGPPredictor(gp_config, gp_manager)
    gp_predictor.load_all_models()

    # Build manifold surface
    print(f"Building {args.grid_n}x{args.grid_n} manifold surface...")
    Xg, Yg, Zg = build_manifold_surface(gp_predictor, grid_n=args.grid_n)

    # Load test trajectories
    from dataset import create_ar_data_splits, create_data_splits
    if is_ar:
        splits = create_ar_data_splits(
            trajectories_dir, window_size=window_size,
            test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"],
            seed=cfg["system"]["seed"],
        )
    else:
        splits = create_data_splits(
            trajectories_dir, window_size=window_size,
            test_size=cfg["data"]["test_size"], val_size=cfg["data"]["val_size"],
            seed=cfg["system"]["seed"],
        )
    test_ids = splits["test_ids"]

    valid_ids = []
    for tid in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{tid}.npz"))
        if len(data["inductance"]) > window_size + 1:
            valid_ids.append(tid)

    rng = np.random.default_rng(42)
    selected = rng.choice(valid_ids, size=min(args.n_trajectories, len(valid_ids)), replace=False)
    print(f"Running inference on {len(selected)} trajectories...")

    # Create plotly figure
    fig = go.Figure()

    # Add manifold surface
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=Zg,
        colorscale='Viridis',
        opacity=0.4,
        showscale=True,
        colorbar=dict(title="Inductance", x=1.05),
        name="Manifold",
        hovertemplate="X: %{x:.2f}mm<br>Y: %{y:.2f}mm<br>L: %{z:.6f}<extra>Manifold</extra>",
    ))

    colors_actual = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#e377c2', '#7f7f7f']
    colors_pred = ['#ff7f0e', '#d62728', '#8c564b', '#bcbd22', '#ff9896', '#c49c94']

    for i, traj_id in enumerate(selected):
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        if is_ar:
            preds = run_ar_inference(model, inductance, positions, window_size,
                                     input_scaler, target_scaler, device)
            actual = positions[window_size:]
            actual_L = inductance[window_size:]
        else:
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
            actual_L = inductance[window_size - 1:]

        pred_L, _ = gp_predictor.predict(preds[:, :2])

        c_act = colors_actual[i % len(colors_actual)]
        c_pred = colors_pred[i % len(colors_pred)]

        # Actual trajectory on manifold
        fig.add_trace(go.Scatter3d(
            x=actual[:, 0], y=actual[:, 1], z=actual_L,
            mode='lines', line=dict(color=c_act, width=4),
            name=f"{traj_id} actual",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>L: %{z:.6f}<extra>Actual</extra>",
        ))

        # Predicted trajectory on manifold
        fig.add_trace(go.Scatter3d(
            x=preds[:, 0], y=preds[:, 1], z=pred_L,
            mode='lines', line=dict(color=c_pred, width=4, dash='dash'),
            name=f"{traj_id} predicted",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>L: %{z:.6f}<extra>Predicted</extra>",
        ))

    fig.update_layout(
        title="Interactive 3D Inductance Manifold with Trajectory Predictions",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Inductance (L)",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        width=1200, height=800,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
    )

    out_path = args.output or os.path.join(args.run_dir, "manifold_interactive.html")
    fig.write_html(out_path, include_plotlyjs=True)
    print(f"\nSaved interactive plot: {out_path}")
    print("Open in browser to rotate, zoom, and pan the 3D view.")


if __name__ == "__main__":
    main()
