"""
CLI entry point: preprocess → train → evaluate → visualize.

Usage:
    python main.py config.yaml
    python main.py config.yaml --preprocess_only
    python main.py config.yaml --eval_only --checkpoint output/.../checkpoints/best_model.pth
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from types import SimpleNamespace

import torch
import yaml
import joblib
import numpy as np

from preprocess import preprocess
from dataset import create_data_splits, create_ar_data_splits
from model import PositionCNN, PositionCNN_AR, PositionLSTM, PositionCNN_LSTM, PositionTransformer, CausalPositionTransformer
from train import train, train_ar
from evaluate import evaluate_model, evaluate_model_ar, print_metrics
from visualize import generate_all_plots
from physics_manifold import PhysicsManifold


def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Flatten into a single namespace
    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
        else:
            flat.update(raw)
            break
    return SimpleNamespace(**flat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml", nargs="?")
    parser.add_argument("--preprocess_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--gp_models_dir", type=str, default=None,
                        help="Path to GP patch models dir for inductance visualization "
                             "(e.g. ../gp_fit_2d/models)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 0: Preprocess if needed
    if not os.path.exists(os.path.join(config.trajectories_dir, "manifest.json")):
        print("\n=== Step 0: Preprocessing ===")
        preprocess(config.csv_path, config.trajectories_dir)

    if args.preprocess_only:
        print("Done (preprocess only).")
        return

    # Create timestamped output directory with model type and data name
    model_type = getattr(config, "model_type", "feedforward")
    data_name = os.path.basename(config.trajectories_dir.rstrip("/"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"{timestamp}_{model_type}_{data_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config_filename = os.path.basename(args.config)
    shutil.copy2(args.config, os.path.join(run_dir, config_filename))

    is_ar = model_type in ("ar", "lstm_ar", "cnn_lstm_ar", "transformer_ar", "causal_transformer_ar")

    # Step 1: Data splits
    print("\n=== Step 1: Creating data splits ===")
    if is_ar:
        warmup_steps = getattr(config, "warmup_steps", 0)
        unroll_steps = getattr(config, "unroll_steps", 1)
        total_unroll = warmup_steps + unroll_steps
        splits = create_ar_data_splits(
            trajectories_dir=config.trajectories_dir,
            window_size=config.window_size,
            stride=config.stride,
            subsample_factor=config.subsample_factor,
            test_size=config.test_size,
            val_size=config.val_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            unroll_steps=total_unroll,
        )
    else:
        splits = create_data_splits(
            trajectories_dir=config.trajectories_dir,
            window_size=config.window_size,
            stride=config.stride,
            subsample_factor=config.subsample_factor,
            test_size=config.test_size,
            val_size=config.val_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
        )

    # Save scalers
    joblib.dump(splits["input_scaler"], os.path.join(run_dir, "input_scaler.pkl"))
    joblib.dump(splits["target_scaler"], os.path.join(run_dir, "target_scaler.pkl"))

    # Step 2: Build model
    output_dim = getattr(config, "output_dim", 3)
    print("\n=== Step 2: Building model ===")
    print(f"  Model type: {model_type}, output_dim: {output_dim}")
    if model_type == "ar":
        model = PositionCNN_AR(
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            output_dim=output_dim,
        ).to(device)
    elif model_type in ("lstm", "lstm_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionLSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            bidirectional=getattr(config, "bidirectional", False),
            output_dim=output_dim,
        ).to(device)
    elif model_type in ("cnn_lstm", "cnn_lstm_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionCNN_LSTM(
            input_size=input_size,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            bidirectional=getattr(config, "bidirectional", False),
            output_dim=output_dim,
        ).to(device)
    elif model_type in ("transformer", "transformer_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = PositionTransformer(
            input_size=input_size,
            d_model=getattr(config, "d_model", 64),
            nhead=getattr(config, "nhead", 4),
            num_layers=getattr(config, "num_layers", 3),
            dim_feedforward=getattr(config, "dim_feedforward", 256),
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            max_len=config.window_size + 10,
            output_dim=output_dim,
        ).to(device)
    elif model_type in ("causal_transformer", "causal_transformer_ar"):
        input_size = (1 + output_dim) if is_ar else 1
        model = CausalPositionTransformer(
            input_size=input_size,
            d_model=getattr(config, "d_model", 64),
            nhead=getattr(config, "nhead", 4),
            num_layers=getattr(config, "num_layers", 3),
            dim_feedforward=getattr(config, "dim_feedforward", 256),
            encoder_dims=getattr(config, "encoder_dims", [64]),
            decoder_dims=getattr(config, "decoder_dims", [128, 64]),
            dropout=config.dropout,
            max_len=config.window_size + 10,
            output_dim=output_dim,
        ).to(device)
    else:
        model = PositionCNN(
            in_channels=1,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            fc_dims=config.fc_dims,
            dropout=config.dropout,
            output_dim=output_dim,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    if args.eval_only and args.checkpoint:
        # Load existing checkpoint
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        history = None
    else:
        # Step 3: Train
        print("\n=== Step 3: Training ===")
        # Load physics manifold if configured
        manifold_path = getattr(config, "physics_manifold", None)
        physics_manifold = None
        if manifold_path and getattr(config, "physics_weight", 0) > 0:
            print(f"  Loading physics manifold: {manifold_path}")
            physics_manifold = PhysicsManifold(manifold_path)

        if is_ar:
            result = train_ar(
                model, splits["train_loader"], splits["val_loader"],
                config, device, run_dir,
                val_ids=splits["val_ids"],
                trajectories_dir=config.trajectories_dir,
                input_scaler=splits["input_scaler"],
                target_scaler=splits["target_scaler"],
                physics_manifold=physics_manifold,
            )
        else:
            result = train(
                model, splits["train_loader"], splits["val_loader"],
                config, device, run_dir,
            )
        model = result["model"]
        history = result["history"]

        # Save history
        with open(os.path.join(run_dir, "history.json"), "w") as f:
            json.dump(history, f)

    # Step 4: Evaluate
    predict_displacement = getattr(config, "predict_displacement", False)
    print("\n=== Step 4: Evaluation ===")
    if is_ar:
        print("  Running autoregressive rollout evaluation...")
        eval_results = evaluate_model_ar(
            model, splits["test_ids"], config.trajectories_dir,
            splits["input_scaler"], splits["target_scaler"],
            config.window_size, device,
            predict_displacement=predict_displacement,
        )
    else:
        eval_results = evaluate_model(
            model, splits["test_loader"], splits["target_scaler"], device
        )
    print_metrics(eval_results["metrics"])

    # Save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(eval_results["metrics"], f, indent=2)

    # Step 5: Visualize
    print("\n=== Step 5: Generating plots ===")

    # Optionally load GP predictor for inductance visualization
    gp_predictor = None
    if args.gp_models_dir:
        import sys
        gp_parent = os.path.dirname(os.path.abspath(args.gp_models_dir))
        sys.path.insert(0, gp_parent)
        from patched_predictor import PatchedGPPredictor
        from patch_config import PatchConfig
        from patch_manager import PatchManager
        gp_config = PatchConfig()
        gp_manager = PatchManager(gp_config, models_dir=args.gp_models_dir)
        gp_predictor = PatchedGPPredictor(gp_config, gp_manager)
        gp_predictor.load_all_models()
        print(f"  Loaded GP predictor from {args.gp_models_dir}")

    if history is not None:
        generate_all_plots(
            history, eval_results, model, splits["test_ids"],
            config.trajectories_dir, splits["input_scaler"], splits["target_scaler"],
            config.window_size, device, run_dir, is_ar=is_ar,
            predict_displacement=predict_displacement,
            unroll_steps=getattr(config, "unroll_steps", 0),
            gp_predictor=gp_predictor,
        )
    else:
        # eval_only mode — skip training history plot
        from visualize import (
            plot_predicted_vs_actual, plot_error_distribution,
            plot_error_heatmap, plot_sample_trajectories,
            plot_sample_trajectories_ar,
        )
        plot_predicted_vs_actual(
            eval_results["predictions"], eval_results["targets"],
            eval_results["metrics"], os.path.join(run_dir, "pred_vs_actual.png"),
        )
        plot_error_distribution(
            eval_results["predictions"], eval_results["targets"],
            os.path.join(run_dir, "error_distribution.png"),
        )
        plot_error_heatmap(
            eval_results["predictions"], eval_results["targets"],
            os.path.join(run_dir, "error_heatmap.png"),
        )
        if is_ar:
            plot_sample_trajectories_ar(
                model, splits["test_ids"], config.trajectories_dir,
                splits["input_scaler"], splits["target_scaler"],
                config.window_size, device,
                os.path.join(run_dir, "trajectory_samples.png"),
                predict_displacement=predict_displacement,
                gp_predictor=gp_predictor,
            )
        else:
            plot_sample_trajectories(
                model, splits["test_ids"], config.trajectories_dir,
                splits["input_scaler"], splits["target_scaler"],
                config.window_size, device,
                os.path.join(run_dir, "trajectory_samples.png"),
                gp_predictor=gp_predictor,
            )

    print(f"\nAll outputs saved to: {run_dir}/")


if __name__ == "__main__":
    main()
