"""
Generate additional trajectory sample plots from a saved model run.

Usage:
    python generate_more_samples.py output/2026-03-11_185701_transformer_ar_constant_v_traj \
        --checkpoint output/2026-03-10_144143_transformer_ar_constant_v_traj/checkpoints/best_model.pth \
        --n_batches 4 --n_per_batch 6
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
from visualize import plot_sample_trajectories_ar, plot_sample_trajectories, plot_worst_trajectories_ar, plot_worst_trajectories_ar_interactive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory with config/scalers")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--n_batches", type=int, default=4, help="Number of sample pages")
    parser.add_argument("--n_per_batch", type=int, default=6, help="Trajectories per page")
    parser.add_argument("--gp_models_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.run_dir, os.path.basename(
        [f for f in os.listdir(args.run_dir) if f.endswith('.yaml')][0]
    ))
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
            input_size=input_size,
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            encoder_dims=cfg["model"]["encoder_dims"],
            decoder_dims=cfg["model"]["decoder_dims"],
            dropout=cfg["model"]["dropout"],
            max_len=window_size + 10,
            output_dim=output_dim,
        ).to(device)
    else:
        model = PositionTransformer(
            input_size=input_size,
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_layers=cfg["model"]["num_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            fc_dims=cfg["model"]["fc_dims"],
            dropout=cfg["model"]["dropout"],
            max_len=window_size + 10,
            output_dim=output_dim,
        ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Load scalers
    input_scaler = joblib.load(os.path.join(args.run_dir, "input_scaler.pkl"))
    target_scaler = joblib.load(os.path.join(args.run_dir, "target_scaler.pkl"))

    # Load test IDs from data split
    from dataset import create_ar_data_splits, create_data_splits
    if is_ar:
        splits = create_ar_data_splits(
            trajectories_dir, window_size=window_size,
            test_size=cfg["data"]["test_size"],
            val_size=cfg["data"]["val_size"],
            seed=cfg["system"]["seed"],
        )
    else:
        splits = create_data_splits(
            trajectories_dir, window_size=window_size,
            test_size=cfg["data"]["test_size"],
            val_size=cfg["data"]["val_size"],
            seed=cfg["system"]["seed"],
        )
    test_ids = splits["test_ids"]
    print(f"Test set: {len(test_ids)} trajectories")

    # Optionally load GP predictor
    gp_predictor = None
    if args.gp_models_dir:
        gp_parent = os.path.dirname(os.path.abspath(args.gp_models_dir))
        sys.path.insert(0, gp_parent)
        from patched_predictor import PatchedGPPredictor
        from patch_config import PatchConfig
        from patch_manager import PatchManager
        gp_config = PatchConfig()
        gp_manager = PatchManager(gp_config, models_dir=args.gp_models_dir)
        gp_predictor = PatchedGPPredictor(gp_config, gp_manager)
        gp_predictor.load_all_models()

    # Filter to trajectories long enough for at least one prediction window
    valid_test_ids = []
    for tid in test_ids:
        data = np.load(os.path.join(trajectories_dir, f"{tid}.npz"))
        if len(data["inductance"]) > window_size + 1:
            valid_test_ids.append(tid)
    print(f"Valid test trajectories (len > {window_size + 1}): {len(valid_test_ids)}")

    # Sample different subsets for each batch
    rng = np.random.default_rng(42)
    total_needed = args.n_batches * args.n_per_batch
    sampled = rng.choice(valid_test_ids, size=min(total_needed, len(valid_test_ids)), replace=False)

    plot_fn = plot_sample_trajectories_ar if is_ar else plot_sample_trajectories
    for batch in range(args.n_batches):
        batch_ids = sampled[batch * args.n_per_batch:(batch + 1) * args.n_per_batch]
        if len(batch_ids) == 0:
            break
        save_path = os.path.join(args.run_dir, f"trajectory_samples_{batch + 1}.png")

        # Call plot function directly with pre-selected IDs
        kwargs = dict(
            model=model, test_ids=list(batch_ids), trajectories_dir=trajectories_dir,
            input_scaler=input_scaler, target_scaler=target_scaler,
            window_size=window_size, device=device, save_path=save_path,
            n_trajectories=len(batch_ids),
        )
        if is_ar:
            kwargs["predict_displacement"] = False
            kwargs["gp_predictor"] = gp_predictor
        else:
            kwargs["gp_predictor"] = gp_predictor

        plot_fn(**kwargs)
        print(f"Saved: {save_path}")

    # Generate worst trajectories plot (static PNG + interactive HTML)
    if is_ar:
        worst_path = os.path.join(args.run_dir, "worst_trajectories.png")
        plot_worst_trajectories_ar(
            model=model, test_ids=valid_test_ids, trajectories_dir=trajectories_dir,
            input_scaler=input_scaler, target_scaler=target_scaler,
            window_size=window_size, device=device, save_path=worst_path,
            n_worst=6, predict_displacement=False, gp_predictor=gp_predictor,
        )
        print(f"Saved: {worst_path}")

        worst_html = os.path.join(args.run_dir, "worst_trajectories.html")
        plot_worst_trajectories_ar_interactive(
            model=model, test_ids=valid_test_ids, trajectories_dir=trajectories_dir,
            input_scaler=input_scaler, target_scaler=target_scaler,
            window_size=window_size, device=device, save_path=worst_html,
            n_worst=6, predict_displacement=False, gp_predictor=gp_predictor,
        )
        print(f"Saved: {worst_html}")


if __name__ == "__main__":
    main()
