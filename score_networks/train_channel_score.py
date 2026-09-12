"""
Train channel score network q_{theta_H}.

Supports two modes:

1. Independent:
       H_j = H_0 + sigma * epsilon

       epsilon_theta(H_j, sigma) ~= -epsilon

2. Temporal AR(1):
       H_t = alpha * H_{t-1} + sqrt(1 - alpha^2) * W_t

       H_j = H_t + sigma * epsilon

       epsilon_theta(H_j, H_{t-1}, sigma) ~= -epsilon

Config:

    channel_temporal: false

or:

    channel_temporal: true
    channel_ar1_alpha: 0.7
    channel_sequence_length: 20

Usage:
    python -m score_networks.train_channel_score \
        --config configs/runpod_minimal.yaml
"""

import argparse
import csv
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .ncsnpp import (
    ChannelScoreNet,
    ChannelTemporalScoreNet,
)

from channels.rayleigh import (
    generate_rayleigh_channel,
    generate_rayleigh_channel_sequence,
)


def setup_logging(log_dir: str, run_name: str):
    """Set up file + console logging and initialize CSV metrics."""
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"{run_name}.log",
    )

    csv_path = os.path.join(
        log_dir,
        f"{run_name}_metrics.csv",
    )

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(console_handler)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "avg_loss",
            "lr",
            "epoch_time_sec",
            "best_loss_so_far",
            "timestamp",
        ])

    return logger, csv_path, log_path


def sample_log_sigma(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
):
    """Sample sigma continuously and uniformly in log-space."""
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)

    log_sigma = (
        torch.rand(batch_size, device=device)
        * (log_sigma_max - log_sigma_min)
        + log_sigma_min
    )

    return log_sigma.exp()


def epsilon_dsm_loss(
    net: nn.Module,
    H0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
):
    """
    Epsilon-parameterized DSM for independent channels.

    H_j = H_0 + sigma * epsilon

    Target:
        epsilon_target = -epsilon

    Network:
        epsilon_pred = net(H_j, sigma)
    """
    batch_size = H0.shape[0]
    device = H0.device

    sigma = sample_log_sigma(
        batch_size,
        sigma_min,
        sigma_max,
        device,
    )

    eps_real = torch.randn_like(H0.real)
    eps_imag = torch.randn_like(H0.imag)

    sigma_3d = sigma[:, None, None]

    H_j_real = H0.real + sigma_3d * eps_real
    H_j_imag = H0.imag + sigma_3d * eps_imag

    H_j = torch.stack(
        [H_j_real, H_j_imag],
        dim=1,
    )

    H_j_norm = (
        H_j
        / sigma[:, None, None, None]
    )

    epsilon_target = torch.stack(
        [-eps_real, -eps_imag],
        dim=1,
    )

    epsilon_pred = net(
        H_j_norm,
        sigma,
    )

    loss = (
        epsilon_pred - epsilon_target
    ).pow(2).mean()

    return loss


def temporal_epsilon_dsm_loss(
    net: nn.Module,
    H_curr: torch.Tensor,
    H_prev: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
):
    """
    Epsilon-parameterized DSM for temporal channels.

    H_j = H_t + sigma * epsilon

    The network receives:

        H_j
        H_{t-1}
        sigma

    and learns:

        epsilon_theta(H_j, H_{t-1}, sigma) ~= -epsilon
    """
    batch_size = H_curr.shape[0]
    device = H_curr.device

    sigma = sample_log_sigma(
        batch_size,
        sigma_min,
        sigma_max,
        device,
    )

    eps_real = torch.randn_like(H_curr.real)
    eps_imag = torch.randn_like(H_curr.imag)

    sigma_3d = sigma[:, None, None]

    # Perturb only H_t.
    H_j_real = (
        H_curr.real
        + sigma_3d * eps_real
    )

    H_j_imag = (
        H_curr.imag
        + sigma_3d * eps_imag
    )

    H_j = torch.stack(
        [H_j_real, H_j_imag],
        dim=1,
    )

    # Same normalization used by the existing channel model.
    H_j_norm = (
        H_j
        / sigma[:, None, None, None]
    )

    # H_{t-1} remains clean.
    H_prev_input = torch.stack(
        [
            H_prev.real,
            H_prev.imag,
        ],
        dim=1,
    )

    epsilon_target = torch.stack(
        [
            -eps_real,
            -eps_imag,
        ],
        dim=1,
    )

    epsilon_pred = net(
        H_j_norm,
        H_prev_input,
        sigma,
    )

    loss = (
        epsilon_pred - epsilon_target
    ).pow(2).mean()

    return loss


def train(cfg: dict, config_path: str = ""):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ============================================================
    # Logging
    # ============================================================

    log_dir = cfg.get(
        "log_dir",
        "logs",
    )

    run_name = (
        cfg.get("run_name")
        or os.path.splitext(
            os.path.basename(config_path)
        )[0]
        or "run"
    )

    run_name = (
        f"{run_name}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger, csv_path, log_path = setup_logging(
        log_dir,
        run_name,
    )

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(
        f"Config: {json.dumps(cfg, indent=2, default=str)}"
    )

    # ============================================================
    # Channel configuration
    # ============================================================

    Nr = cfg.get("Nr", 4)
    Nt = cfg.get("Nt", 1)
    K = cfg.get("K", 192)
    Nu = cfg.get("Nu", 1)

    channel_temporal = cfg.get(
        "channel_temporal",
        False,
    )

    alpha = cfg.get(
        "channel_ar1_alpha",
        0.7,
    )

    sequence_length = cfg.get(
        "channel_sequence_length",
        20,
    )

    if channel_temporal:
        if not 0.0 <= alpha < 1.0:
            raise ValueError(
                f"channel_ar1_alpha must satisfy "
                f"0 <= alpha < 1, got {alpha}"
            )

        if sequence_length < 2:
            raise ValueError(
                "channel_sequence_length must be >= 2"
            )

    # IMPORTANT:
    # channel_J is independent from the PVD J.
    channel_J = cfg.get(
        "channel_J",
        cfg.get("J", 50),
    )

    sigma_min = cfg.get(
        "sigma_H_1",
        0.01,
    )

    sigma_max = cfg.get(
        "sigma_H_J",
        10.0,
    )

    if sigma_min <= 0:
        raise ValueError(
            f"sigma_H_1 must be > 0, got {sigma_min}"
        )

    if sigma_max <= sigma_min:
        raise ValueError(
            f"sigma_H_J must be > sigma_H_1, "
            f"got {sigma_min} -> {sigma_max}"
        )

    logger.info(
        f"Channel dimensions: "
        f"Nr={Nr}, Nt={Nt}, K={K}, Nu={Nu}"
    )

    logger.info(
        f"Sigma range: [{sigma_min}, {sigma_max}]"
    )

    logger.info(
        "Continuous log-sigma sampling enabled"
    )

    if channel_temporal:
        logger.info(
            "Channel training mode: TEMPORAL AR(1)"
        )
        logger.info(
            f"AR(1) alpha: {alpha}"
        )
        logger.info(
            f"Sequence length: {sequence_length}"
        )
    else:
        logger.info(
            "Channel training mode: INDEPENDENT"
        )

    # ============================================================
    # Network
    # ============================================================

    if channel_temporal:
        net = ChannelTemporalScoreNet(
            Nr=Nr,
            Nt=Nt,
            K=K,
            hidden_dim=cfg.get(
                "channel_hidden_dim",
                1024,
            ),
            num_layers=cfg.get(
                "channel_num_layers",
                8,
            ),
            time_dim=cfg.get(
                "channel_time_dim",
                512,
            ),
        ).to(device)
    else:
        net = ChannelScoreNet(
            Nr=Nr,
            Nt=Nt,
            K=K,
            hidden_dim=cfg.get(
                "channel_hidden_dim",
                1024,
            ),
            num_layers=cfg.get(
                "channel_num_layers",
                8,
            ),
            time_dim=cfg.get(
                "channel_time_dim",
                512,
            ),
        ).to(device)

    num_parameters = sum(
        p.numel()
        for p in net.parameters()
        if p.requires_grad
    )

    logger.info(
        f"Trainable parameters: {num_parameters:,}"
    )

    # ============================================================
    # Optimizer
    # ============================================================

    learning_rate = cfg.get(
        "score_lr",
        2e-4,
    )

    optimizer = optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=cfg.get(
            "score_weight_decay",
            1e-4,
        ),
    )

    epochs = cfg.get(
        "score_epochs",
        500,
    )

    min_learning_rate = cfg.get(
        "score_lr_min",
        1e-5,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_learning_rate,
    )

    # ============================================================
    # Training configuration
    # ============================================================

    batch_size = cfg.get(
        "score_batch_size",
        256,
    )

    samples_per_epoch = cfg.get(
        "score_samples_per_epoch",
        batch_size * 20,
    )

    if samples_per_epoch < batch_size:
        samples_per_epoch = batch_size

    # ============================================================
    # Checkpoints
    # ============================================================

    ckpt_dir = cfg.get(
        "score_ckpt_dir",
        "score_networks/checkpoints",
    )

    os.makedirs(
        ckpt_dir,
        exist_ok=True,
    )

    if channel_temporal:
        best_checkpoint = os.path.join(
            ckpt_dir,
            "channel_score_temporal_best.pt",
        )

        final_checkpoint = os.path.join(
            ckpt_dir,
            "channel_score_temporal_final.pt",
        )
    else:
        best_checkpoint = os.path.join(
            ckpt_dir,
            "channel_score_best.pt",
        )

        final_checkpoint = os.path.join(
            ckpt_dir,
            "channel_score_final.pt",
        )

    # ============================================================
    # Training state
    # ============================================================

    best_loss = float("inf")
    history = []

    run_start = time.time()

    logger.info(
        f"Epochs: {epochs}"
    )

    logger.info(
        f"Batch size: {batch_size}"
    )

    logger.info(
        f"Samples per epoch: {samples_per_epoch}"
    )

    if channel_temporal:
        logger.info(
            f"Training pairs per epoch: "
            f"{samples_per_epoch * (sequence_length - 1)}"
        )
    else:
        logger.info(
            f"Training samples per epoch: "
            f"{samples_per_epoch}"
        )

    # ============================================================
    # Training
    # ============================================================

    try:
        for epoch in range(epochs):
            epoch_start = time.time()

            net.train()

            # ====================================================
            # Generate data
            # ====================================================

            if channel_temporal:

                # H_seq:
                # (B, T, Nu, Nr*K, Nt*K)

                H_seq = generate_rayleigh_channel_sequence(
                    samples_per_epoch,
                    Nu,
                    Nr,
                    Nt,
                    K,
                    sequence_length,
                    alpha=alpha,
                    device=device,
                )

                # Use first user:
                # (B, T, Nr*K, Nt*K)

                H_seq = H_seq[:, :, 0]

                # Previous/current pairs:
                #
                # H_prev:
                # H_0, H_1, ..., H_{T-2}
                #
                # H_curr:
                # H_1, H_2, ..., H_{T-1}

                H_prev = H_seq[:, :-1]
                H_curr = H_seq[:, 1:]

                # Flatten sequence dimension.
                #
                # (B, T-1, Nr*K, Nt*K)
                #       ↓
                # (B*(T-1), Nr*K, Nt*K)

                H_prev = H_prev.reshape(
                    -1,
                    Nr * K,
                    Nt * K,
                )

                H_curr = H_curr.reshape(
                    -1,
                    Nr * K,
                    Nt * K,
                )

                dataset = TensorDataset(
                    H_curr,
                    H_prev,
                )

            else:

                # H0:
                # (B, Nu, Nr*K, Nt*K)

                H0 = generate_rayleigh_channel(
                    samples_per_epoch,
                    Nu,
                    Nr,
                    Nt,
                    K,
                    device,
                )

                # First user:
                # (B, Nr*K, Nt*K)

                H0 = H0[:, 0]

                dataset = TensorDataset(H0)

            # ====================================================
            # DataLoader
            # ====================================================

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            total_loss = 0.0
            num_steps = 0

            # ====================================================
            # Optimization
            # ====================================================

            for batch in loader:

                if channel_temporal:

                    H_curr_batch, H_prev_batch = batch

                    loss = temporal_epsilon_dsm_loss(
                        net,
                        H_curr_batch,
                        H_prev_batch,
                        sigma_min,
                        sigma_max,
                    )

                else:

                    (H0_batch,) = batch

                    loss = epsilon_dsm_loss(
                        net,
                        H0_batch,
                        sigma_min,
                        sigma_max,
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                total_loss += loss.item()
                num_steps += 1

            # ====================================================
            # Scheduler
            # ====================================================

            scheduler.step()

            avg_loss = (
                total_loss / num_steps
                if num_steps > 0
                else float("inf")
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            # ====================================================
            # Best checkpoint
            # ====================================================

            if avg_loss < best_loss:
                best_loss = avg_loss

                torch.save(
                    net.state_dict(),
                    best_checkpoint,
                )

            # ====================================================
            # History
            # ====================================================

            history_entry = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "best_loss_so_far": best_loss,
            }

            history.append(history_entry)

            # ====================================================
            # CSV
            # ====================================================

            with open(
                csv_path,
                "a",
                newline="",
            ) as f:
                writer = csv.writer(f)

                writer.writerow([
                    epoch + 1,
                    f"{avg_loss:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    f"{best_loss:.6f}",
                    datetime.now().isoformat(),
                ])

            # ====================================================
            # Logging
            # ====================================================

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.6f} | "
                f"lr={current_lr:.8f} | "
                f"best={best_loss:.6f} | "
                f"time={epoch_time:.2f}s"
            )

        # ========================================================
        # Final checkpoint
        # ========================================================

        torch.save(
            net.state_dict(),
            final_checkpoint,
        )

        total_time = (
            time.time()
            - run_start
        )

        logger.info(
            "Channel score network training complete. "
            f"Total time: {total_time:.2f}s"
        )

        # ========================================================
        # Summary
        # ========================================================

        summary = {
            "run_name": run_name,
            "status": "completed",
            "config": cfg,
            "temporal": channel_temporal,
            "ar1_alpha": (
                alpha
                if channel_temporal
                else None
            ),
            "sequence_length": (
                sequence_length
                if channel_temporal
                else None
            ),
            "total_epochs": epochs,
            "total_time_sec": total_time,
            "best_loss": best_loss,
            "final_loss": (
                history[-1]["avg_loss"]
                if history
                else None
            ),
            "device": str(device),
            "checkpoint_best": best_checkpoint,
            "checkpoint_final": final_checkpoint,
            "history": history,
        }

        summary_path = os.path.join(
            log_dir,
            f"{run_name}_summary.json",
        )

        with open(
            summary_path,
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
                default=str,
            )

        logger.info(
            f"Summary written to {summary_path}"
        )

        logger.info(
            f"Per-epoch metrics CSV: {csv_path}"
        )

        logger.info(
            f"Full log: {log_path}"
        )

    except Exception as e:

        # ========================================================
        # Crash report
        # ========================================================

        logger.error(
            f"Training crashed at epoch "
            f"{len(history) + 1}: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        crash_summary = {
            "run_name": run_name,
            "status": "crashed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "epochs_completed": len(history),
            "best_loss": best_loss,
            "temporal": channel_temporal,
            "ar1_alpha": (
                alpha
                if channel_temporal
                else None
            ),
            "sequence_length": (
                sequence_length
                if channel_temporal
                else None
            ),
            "history": history,
        }

        crash_path = os.path.join(
            log_dir,
            f"{run_name}_CRASHED.json",
        )

        with open(
            crash_path,
            "w",
        ) as f:
            json.dump(
                crash_summary,
                f,
                indent=2,
                default=str,
            )

        logger.error(
            f"Crash report written to {crash_path}"
        )

        raise


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(
        cfg,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()