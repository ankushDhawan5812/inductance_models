"""
Training loop with early stopping, LR scheduling, and checkpointing.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0, epoch=0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} train", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(loader, desc="         val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / n_batches
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets


@torch.no_grad()
def validate_ar(
    model, val_ids: list, trajectories_dir: str,
    input_scaler: StandardScaler, target_scaler: StandardScaler,
    window_size: int, device, max_trajectories: int = 50,
    predict_displacement: bool = False,
    max_steps: int = 0,
) -> float:
    """
    AR validation with true autoregressive rollout.

    Runs AR inference on a subset of validation trajectories and returns
    mean squared error in normalized space.

    Args:
        max_steps: If > 0, only unroll this many steps per trajectory segment
                   (makes the metric comparable to K-step training loss).
                   If 0, rolls out the full trajectory.
    """
    model.eval()
    rng = np.random.default_rng(0)
    subset = rng.choice(val_ids, size=min(max_trajectories, len(val_ids)), replace=False)

    all_errors_sq = []

    for traj_id in subset:
        data = np.load(os.path.join(trajectories_dir, f"{traj_id}.npz"))
        inductance = data["inductance"]
        positions = data["positions"]

        total_steps = len(inductance) - window_size
        if total_steps <= 0:
            continue
        if max_steps > 0:
            # Sample multiple short segments from this trajectory
            n_segments = max(1, total_steps // max_steps)
            segment_starts = rng.choice(total_steps, size=min(n_segments, total_steps), replace=False)
        else:
            segment_starts = [0]

        for seg_start in segment_starts:
            # Reset position history from ground truth at segment start
            gt_start = seg_start
            pos_history = positions[gt_start:gt_start + window_size].copy()
            n_rollout = max_steps if max_steps > 0 else total_steps

            for step in range(n_rollout):
                abs_start = gt_start + step + 1
                if abs_start + window_size > len(inductance):
                    break

                ind = inductance[abs_start:abs_start + window_size].reshape(-1, 1)
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
                    pred_pos_norm = out
                    pred_pos = target_scaler.inverse_transform(
                        out.reshape(1, -1)
                    ).flatten()

                target_norm = target_scaler.transform(
                    positions[abs_start + window_size - 1].reshape(1, -1)
                ).flatten()
                # Mean squared error per dimension (comparable to MSELoss)
                err_sq = np.mean((pred_pos_norm - target_norm) ** 2)
                all_errors_sq.append(err_sq)

                pos_history = np.vstack([pos_history[1:], pred_pos.reshape(1, -1)])

    return float(np.mean(all_errors_sq))


def train(model, train_loader, val_loader, config, device, output_dir):
    """
    Full training loop.

    Args:
        config: namespace with training.*, system.* attributes

    Returns:
        dict with model, history, best_val_loss
    """
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    criterion = nn.MSELoss()
    if hasattr(config, "loss") and config.loss == "huber":
        criterion = nn.HuberLoss()

    loss_scale = getattr(config, "loss_scale", 1.0)
    if loss_scale != 1.0:
        base_criterion = criterion
        criterion = lambda pred, tgt: base_criterion(pred, tgt) * loss_scale
        print(f"  Loss scale: {loss_scale}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\nTraining for up to {config.epochs} epochs...")
    print(f"  Device: {device}")
    print(f"  Train batches/epoch: {len(train_loader):,}")
    print(f"  Val batches/epoch: {len(val_loader):,}")

    for epoch in range(config.epochs):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip, epoch
        )
        val_loss, _, _ = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Step scheduler
        if config.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(output_dir, "checkpoints", "best_model.pth"),
            )
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:3d} | "
            f"train={train_loss:.6f} val={val_loss:.6f} | "
            f"lr={current_lr:.1e} | "
            f"{elapsed:.1f}s"
            + (" *" if epoch == best_epoch else "")
        )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch}, best was epoch {best_epoch}")
            break

    # Restore best model
    checkpoint = torch.load(
        os.path.join(output_dir, "checkpoints", "best_model.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


# --- Autoregressive training ---


def train_epoch_ar(model, loader, criterion, optimizer, device,
                   grad_clip=1.0, epoch=0, teacher_forcing_ratio=1.0):
    """
    AR training epoch with scheduled sampling.

    teacher_forcing_ratio=1.0: always use ground truth previous positions (pure teacher forcing)
    teacher_forcing_ratio=0.0: always use model's own predictions for position channels

    For values in between, each sample in the batch independently uses GT or model
    predictions based on a Bernoulli draw.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} train (tf={teacher_forcing_ratio:.2f})", leave=False)
    for inputs, targets in pbar:
        # inputs: (batch, window_size, 4) — [inductance, gt_prev_x, gt_prev_y, gt_prev_z]
        # targets: (batch, 3)
        inputs = inputs.to(device)
        targets = targets.to(device)

        if teacher_forcing_ratio >= 1.0:
            # Pure teacher forcing — use input as-is
            outputs = model(inputs)
        else:
            # Scheduled sampling: replace position channels with model predictions
            # for some samples in the batch
            batch_size = inputs.size(0)
            use_model = torch.rand(batch_size, device=device) > teacher_forcing_ratio

            if use_model.any():
                # For samples using model predictions, run the model with GT first
                # to get a prediction, then corrupt the position channels
                with torch.no_grad():
                    model_pred = model(inputs)  # (batch, 3)

                # Replace the last position in the window with model's prediction
                # This simulates feeding back predictions at the most recent step
                modified = inputs.clone()
                # Position channels are indices 1,2,3 at the last timestep
                modified[use_model, -1, 1:] = model_pred[use_model]
                outputs = model(modified)
            else:
                outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def train_epoch_ar_multistep(model, loader, criterion, optimizer, device,
                             grad_clip=1.0, epoch=0, unroll_steps=10,
                             loss_weighting="uniform", smoothness_weight=0.0,
                             predict_displacement=False, warmup_steps=0,
                             physics_manifold=None, physics_weight=0.0,
                             target_mean=None, target_std=None,
                             input_mean=None, input_std=None):
    """
    Multi-step unrolled AR training epoch.

    Unrolls the model for K steps, feeding back its own predictions at each step.
    Backpropagates through the entire unrolled chain, directly optimizing for
    multi-step rollout performance.

    Optionally runs M warm-up steps (no gradients) before the K training steps
    to contaminate the position history with model predictions, matching the
    distribution seen during full-trajectory AR evaluation.

    Args:
        unroll_steps: K — number of AR steps with loss + backprop
        loss_weighting: "uniform" or "linear" (later steps weighted more)
        smoothness_weight: weight for velocity-matching auxiliary loss (0 = disabled)
        predict_displacement: if True, model predicts delta instead of absolute pos
        warmup_steps: M — number of AR steps to run without gradients before
                      the K training steps. Contaminates the position buffer with
                      model predictions so training sees realistic inputs.
        physics_manifold: optional PhysicsManifold module for consistency loss
        physics_weight: weight for the physics consistency loss (0 = disabled)
        target_mean/target_std: (D,) tensors for denormalizing position predictions
        input_mean/input_std: (1,) tensors for denormalizing inductance values
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    K = unroll_steps
    M = warmup_steps

    # Compute step weights (sum to 1) — only for the K training steps
    if loss_weighting == "linear":
        weights = torch.arange(1, K + 1, dtype=torch.float32, device=device)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(K, dtype=torch.float32, device=device) / K

    desc = f"Epoch {epoch:3d} train (warmup={M}, unroll={K})" if M > 0 else f"Epoch {epoch:3d} train (unroll={K})"
    pbar = tqdm(loader, desc=desc, leave=False)
    for inp, future_ind, targets in pbar:
        # inp:        (B, W, 1+D) — initial window [inductance, gt_prev_positions]
        # future_ind: (B, M+K-1, 1) — inductance values for shifting the window
        # targets:    (B, M+K, D)   — position targets for all steps
        inp = inp.to(device)
        future_ind = future_ind.to(device)
        targets = targets.to(device)

        # Split initial window into inductance and position channels
        ind_window = inp[:, :, 0:1]   # (B, W, 1)
        pos_history = inp[:, :, 1:]   # (B, W, D)

        # --- Phase 1: Warm-up (no gradients) ---
        # Run M steps to contaminate position buffer with model predictions.
        # After M >= W steps, the entire buffer is model predictions.
        if M > 0:
            with torch.no_grad():
                for m in range(M):
                    x = torch.cat([ind_window, pos_history], dim=-1)
                    out = model(x)

                    if predict_displacement:
                        pred_pos = pos_history[:, -1, :] + out
                    else:
                        pred_pos = out

                    # Shift window
                    new_ind = future_ind[:, m:m + 1, :]
                    ind_window = torch.cat([ind_window[:, 1:, :], new_ind], dim=1)
                    pos_history = torch.cat([pos_history[:, 1:, :], pred_pos.unsqueeze(1)], dim=1)

            # Detach to ensure no gradient leakage from warm-up
            ind_window = ind_window.detach()
            pos_history = pos_history.detach()

        # --- Phase 2: Training (with gradients) ---
        optimizer.zero_grad()

        position_loss = 0.0
        smooth_loss = 0.0
        phys_loss = 0.0
        prev_pred_pos = pos_history[:, -1, :]

        for k in range(K):
            x = torch.cat([ind_window, pos_history], dim=-1)
            out = model(x)

            # Target index is M + k (skip warm-up targets)
            t_idx = M + k

            if predict_displacement:
                if k == 0:
                    prev_target = pos_history[:, -1, :]
                else:
                    prev_target = targets[:, t_idx - 1]
                delta_target = targets[:, t_idx] - prev_target
                position_loss = position_loss + weights[k] * criterion(out, delta_target)
                pred_pos = prev_pred_pos + out
            else:
                position_loss = position_loss + weights[k] * criterion(out, targets[:, t_idx])
                pred_pos = out

            if smoothness_weight > 0:
                pred_velocity = pred_pos - prev_pred_pos
                if k == 0:
                    target_velocity = targets[:, t_idx] - pos_history[:, -1, :]
                else:
                    target_velocity = targets[:, t_idx] - targets[:, t_idx - 1]
                smooth_loss = smooth_loss + criterion(pred_velocity, target_velocity)

            # Physics consistency loss: does the predicted position's expected
            # inductance match the actually observed inductance?
            if physics_weight > 0 and physics_manifold is not None:
                # Denormalize predicted position to mm
                pred_mm = pred_pos * target_std + target_mean  # (B, D)
                # Look up expected inductance from manifold (only x, y)
                expected_ind = physics_manifold(pred_mm[:, :2])  # (B,)
                # Actual inductance at this timestep: last value in current window
                actual_ind_norm = ind_window[:, -1, 0]  # (B,)
                actual_ind = actual_ind_norm * input_std[0] + input_mean[0]  # denormalize
                # MSE between expected and actual inductance
                phys_loss = phys_loss + weights[k] * F.mse_loss(expected_ind, actual_ind)

            prev_pred_pos = pred_pos

            if k < K - 1:
                new_ind = future_ind[:, M + k:M + k + 1, :]
                ind_window = torch.cat([ind_window[:, 1:, :], new_ind], dim=1)
                pos_history = torch.cat([pos_history[:, 1:, :], pred_pos.unsqueeze(1)], dim=1)

        loss = position_loss
        if smoothness_weight > 0:
            loss = loss + smoothness_weight * (smooth_loss / K)
        if physics_weight > 0 and physics_manifold is not None:
            loss = loss + physics_weight * phys_loss
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.6f}")

    return total_loss / n_batches


def train_ar(model, train_loader, val_loader, config, device, output_dir,
             val_ids=None, trajectories_dir=None,
             input_scaler=None, target_scaler=None,
             physics_manifold=None):
    """
    AR training loop with scheduled sampling.

    Scheduled sampling linearly decays teacher_forcing_ratio from 1.0 to
    config.tf_ratio_end over config.tf_decay_epochs epochs.

    If val_ids/trajectories_dir/scalers are provided, validation uses true
    autoregressive rollout. Otherwise falls back to teacher-forced validation.
    """
    use_ar_val = val_ids is not None and trajectories_dir is not None
    unroll_steps = getattr(config, "unroll_steps", 1)
    warmup_steps = getattr(config, "warmup_steps", 0)
    use_multistep = unroll_steps > 1
    unroll_curriculum_epochs = getattr(config, "unroll_curriculum_epochs", 0)
    warmup_curriculum_epochs = getattr(config, "warmup_curriculum_epochs", 0)
    loss_weighting = getattr(config, "unroll_loss_weighting", "uniform")
    smoothness_weight = getattr(config, "smoothness_weight", 0.0)
    predict_displacement = getattr(config, "predict_displacement", False)
    physics_weight = getattr(config, "physics_weight", 0.0)

    # Prepare scaler tensors for physics loss (denormalization in the training loop)
    target_mean_t = target_std_t = input_mean_t = input_std_t = None
    if physics_manifold is not None and physics_weight > 0 and target_scaler is not None:
        target_mean_t = torch.tensor(target_scaler.mean_, dtype=torch.float32, device=device)
        target_std_t = torch.tensor(target_scaler.scale_, dtype=torch.float32, device=device)
        input_mean_t = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        input_std_t = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        physics_manifold = physics_manifold.to(device)
        print(f"  Physics manifold enabled: weight={physics_weight}")

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    criterion = nn.MSELoss()
    if hasattr(config, "loss") and config.loss == "huber":
        criterion = nn.HuberLoss()

    loss_scale = getattr(config, "loss_scale", 1.0)
    if loss_scale != 1.0:
        base_criterion = criterion
        criterion = lambda pred, tgt: base_criterion(pred, tgt) * loss_scale
        print(f"  Loss scale: {loss_scale}")

    # Collect parameters: model + physics manifold scale/offset (if present)
    params = list(model.parameters())
    if physics_manifold is not None and physics_weight > 0:
        params += list(physics_manifold.parameters())
    optimizer = torch.optim.Adam(
        params, lr=config.lr, weight_decay=config.weight_decay
    )

    if config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    elif config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    else:
        scheduler = None

    tf_start = getattr(config, "tf_ratio_start", 1.0)
    tf_end = getattr(config, "tf_ratio_end", 0.2)
    tf_decay_epochs = getattr(config, "tf_decay_epochs", 30)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_loss_full": [],
        "lr": [],
        "teacher_forcing_ratio": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\nAR Training for up to {config.epochs} epochs...")
    print(f"  Device: {device}")
    print(f"  Train batches/epoch: {len(train_loader):,}")
    print(f"  Validation: {'AR rollout (50 trajectories)' if use_ar_val else 'teacher-forced'}")
    if use_multistep:
        print(f"  Multi-step unrolled training: {unroll_steps} steps")
        if warmup_steps > 0:
            print(f"  Warm-up steps: {warmup_steps} (no-grad buffer contamination)")
            if warmup_curriculum_epochs > 0:
                print(f"  Warmup curriculum: ramp 0 → {warmup_steps} over {warmup_curriculum_epochs} epochs")
        if unroll_curriculum_epochs > 0:
            print(f"  Unroll curriculum: ramp 2 → {unroll_steps} over {unroll_curriculum_epochs} epochs")
        if loss_weighting != "uniform":
            print(f"  Loss weighting: {loss_weighting}")
        if smoothness_weight > 0:
            print(f"  Smoothness weight: {smoothness_weight}")
        if predict_displacement:
            print(f"  Predicting displacement (delta)")
    else:
        print(f"  Teacher forcing: {tf_start:.1f} → {tf_end:.1f} over {tf_decay_epochs} epochs")

    for epoch in range(config.epochs):
        t0 = time.time()

        if use_multistep:
            tf_ratio = 1.0  # Not used, but logged for consistency
            # Curriculum: ramp effective unroll steps from 2 to target
            if unroll_curriculum_epochs > 0 and epoch < unroll_curriculum_epochs:
                effective_steps = 2 + int(
                    (unroll_steps - 2) * epoch / unroll_curriculum_epochs
                )
                effective_steps = min(effective_steps, unroll_steps)
            else:
                effective_steps = unroll_steps
            # Warmup curriculum: ramp from 0 to warmup_steps
            if warmup_steps > 0 and warmup_curriculum_epochs > 0 and epoch < warmup_curriculum_epochs:
                effective_warmup = int(warmup_steps * epoch / warmup_curriculum_epochs)
            else:
                effective_warmup = warmup_steps
            train_loss = train_epoch_ar_multistep(
                model, train_loader, criterion, optimizer, device,
                config.grad_clip, epoch, effective_steps,
                loss_weighting, smoothness_weight, predict_displacement,
                warmup_steps=effective_warmup,
                physics_manifold=physics_manifold,
                physics_weight=physics_weight,
                target_mean=target_mean_t, target_std=target_std_t,
                input_mean=input_mean_t, input_std=input_std_t,
            )
        else:
            # Compute teacher forcing ratio for this epoch
            if epoch < tf_decay_epochs:
                tf_ratio = tf_start - (tf_start - tf_end) * epoch / tf_decay_epochs
            else:
                tf_ratio = tf_end

            train_loss = train_epoch_ar(
                model, train_loader, criterion, optimizer, device,
                config.grad_clip, epoch, tf_ratio,
            )

        if use_ar_val:
            # Short-horizon val (comparable to training loss)
            val_loss = validate_ar(
                model, val_ids, trajectories_dir,
                input_scaler, target_scaler,
                config.window_size, device,
                predict_displacement=predict_displacement,
                max_steps=effective_steps if use_multistep else 0,
            )
            # Full-rollout val (true AR performance)
            val_loss_full = validate_ar(
                model, val_ids, trajectories_dir,
                input_scaler, target_scaler,
                config.window_size, device,
                predict_displacement=predict_displacement,
                max_steps=0,
            )
        else:
            val_loss, _, _ = validate(model, val_loader, criterion, device)
            val_loss_full = val_loss

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_loss_full"].append(val_loss_full)
        history["lr"].append(current_lr)
        history["teacher_forcing_ratio"].append(tf_ratio)

        if config.scheduler == "plateau":
            scheduler.step(val_loss_full)
        elif scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        if val_loss_full < best_val_loss:
            best_val_loss = val_loss_full
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(output_dir, "checkpoints", "best_model.pth"),
            )
        else:
            patience_counter += 1

        print(
            f"  Epoch {epoch:3d} | "
            f"train={train_loss:.6f} val={val_loss:.6f} val_full={val_loss_full:.6f} | "
            f"lr={current_lr:.1e} tf={tf_ratio:.2f} | "
            f"{elapsed:.1f}s"
            + (" *" if epoch == best_epoch else "")
        )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch}, best was epoch {best_epoch}")
            break

    checkpoint = torch.load(
        os.path.join(output_dir, "checkpoints", "best_model.pth"),
        map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }
