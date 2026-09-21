"""
Train channel score network using a Gaussian VE-SDE.

Forward SDE:
    dH = g(t) dW

with geometric noise schedule

    sigma(t) = sigma_min * (sigma_max / sigma_min)^t,
    t ~ Uniform(0, 1).

Its transition distribution is

    H_t | H_0 ~ N(H_0, sigma(t)^2 I),

so we sample directly as

    H_t = H_0 + sigma(t) * epsilon.

The network directly predicts the score

    s_theta(H_t, t) ~= -epsilon / sigma(t).

VE score-matching loss:

    L = E[
        sigma(t)^2 *
        ||s_theta(H_t,t) + epsilon/sigma(t)||^2
    ].

Usage:
    python -m score_networks.train_channel_score_ve_sde \
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

from .ncsnpp import ChannelScoreNet
from channels.rayleigh import generate_rayleigh_channel


# ================================================================
# Logging
# ================================================================

def setup_logging(log_dir: str, run_name: str):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{run_name}.log")
    csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")

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


# ================================================================
# VE-SDE schedule
# ================================================================

def ve_sigma(
    t: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
):
    """
    Geometric VE noise schedule.

        sigma(t)
            = sigma_min
              * (sigma_max / sigma_min)^t

    t = 0  -> sigma_min
    t = 1  -> sigma_max
    """

    ratio = sigma_max / sigma_min

    return sigma_min * ratio ** t


def sample_time(
    batch_size: int,
    device: torch.device,
):
    """
    Continuous SDE time:

        t ~ Uniform(0, 1)
    """

    return torch.rand(
        batch_size,
        device=device,
    )


# ================================================================
# VE-SDE score-matching loss
# ================================================================

def ve_sde_dsm_loss(
    net: nn.Module,
    H0: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
):
    """
    Gaussian VE-SDE denoising score matching.

    Forward marginal:

        H_t = H_0 + sigma(t) * epsilon

    epsilon ~ N(0, I)

    Conditional score:

        grad log p(H_t | H_0)
            = -epsilon / sigma(t)

    Network:

        score_pred
            = s_theta(H_t, t)

    Loss:

        sigma(t)^2
        * ||score_pred - score_target||^2
    """

    batch_size = H0.shape[0]
    device = H0.device

    # ------------------------------------------------------------
    # Sample continuous SDE time
    # ------------------------------------------------------------

    t = sample_time(
        batch_size,
        device,
    )

    # ------------------------------------------------------------
    # Convert time -> VE noise scale
    # ------------------------------------------------------------

    sigma = ve_sigma(
        t,
        sigma_min,
        sigma_max,
    )

    sigma_3d = sigma[:, None, None]
    sigma_4d = sigma[:, None, None, None]

    # ------------------------------------------------------------
    # Gaussian Brownian perturbation
    # ------------------------------------------------------------

    eps_real = torch.randn_like(
        H0.real
    )

    eps_imag = torch.randn_like(
        H0.imag
    )

    H_t_real = (
        H0.real
        + sigma_3d * eps_real
    )

    H_t_imag = (
        H0.imag
        + sigma_3d * eps_imag
    )

    H_t = torch.stack(
        [
            H_t_real,
            H_t_imag,
        ],
        dim=1,
    )

    # ------------------------------------------------------------
    # Same normalization convention as our other experiments.
    #
    # The physical SDE state is H_t.
    # We normalize only the NN input.
    # ------------------------------------------------------------

    H_t_norm = (
        H_t
        / sigma_4d
    )

    # ------------------------------------------------------------
    # Exact Gaussian conditional score
    #
    # grad log p(H_t | H_0)
    #     = -(H_t - H_0) / sigma^2
    #     = -epsilon / sigma
    # ------------------------------------------------------------

    score_target = torch.stack(
        [
            -eps_real / sigma_3d,
            -eps_imag / sigma_3d,
        ],
        dim=1,
    )

    # ------------------------------------------------------------
    # Network predicts SCORE directly.
    #
    # ChannelScoreNet currently expects its second argument
    # to be the noise level, so pass sigma(t).
    # ------------------------------------------------------------

    score_pred = net(
        H_t_norm,
        sigma,
    )

    # ------------------------------------------------------------
    # VE likelihood / score weighting
    #
    # lambda(t) = sigma(t)^2
    # ------------------------------------------------------------

    loss = (
        sigma_4d.pow(2)
        * (
            score_pred
            - score_target
        ).pow(2)
    ).mean()

    return loss


# ================================================================
# Training
# ================================================================

def train(
    cfg: dict,
    config_path: str = "",
):

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

    base_run_name = (
        cfg.get("run_name")
        or os.path.splitext(
            os.path.basename(config_path)
        )[0]
        or "run"
    )

    run_name = (
        f"{base_run_name}_ve_sde_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger, csv_path, log_path = (
        setup_logging(
            log_dir,
            run_name,
        )
    )

    logger.info(
        f"Starting VE-SDE run: {run_name}"
    )

    logger.info(
        f"Device: {device}"
    )

    logger.info(
        f"Config: "
        f"{json.dumps(cfg, indent=2, default=str)}"
    )

    # ============================================================
    # Channel
    # ============================================================

    Nr = cfg.get("Nr", 4)
    Nt = cfg.get("Nt", 1)
    K = cfg.get("K", 192)
    Nu = cfg.get("Nu", 1)

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
            "sigma_min must be > 0"
        )

    if sigma_max <= sigma_min:
        raise ValueError(
            "sigma_max must be > sigma_min"
        )

    logger.info(
        f"Channel dimensions: "
        f"Nr={Nr}, Nt={Nt}, K={K}, Nu={Nu}"
    )

    logger.info(
        f"VE sigma range: "
        f"[{sigma_min}, {sigma_max}]"
    )

    logger.info(
        "SDE time: t ~ Uniform(0,1)"
    )

    logger.info(
        "Parameterization: DIRECT SCORE"
    )

    # ============================================================
    # Network
    # ============================================================

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
        f"Trainable parameters: "
        f"{num_parameters:,}"
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

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=cfg.get(
                "score_lr_min",
                1e-5,
            ),
        )
    )

    # ============================================================
    # Dataset
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

    best_checkpoint = os.path.join(
        ckpt_dir,
        "channel_score_ve_sde_best.pt",
    )

    final_checkpoint = os.path.join(
        ckpt_dir,
        "channel_score_ve_sde_final.pt",
    )

    # ============================================================
    # State
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
        f"Samples per epoch: "
        f"{samples_per_epoch}"
    )

    # ============================================================
    # Training loop
    # ============================================================

    try:

        for epoch in range(epochs):

            epoch_start = time.time()

            net.train()

            # ----------------------------------------------------
            # Fresh channel samples
            # ----------------------------------------------------

            H0 = generate_rayleigh_channel(
                samples_per_epoch,
                Nu,
                Nr,
                Nt,
                K,
                device,
            )

            H0 = H0[:, 0]

            loader = DataLoader(
                TensorDataset(H0),

                batch_size=batch_size,

                shuffle=True,

                drop_last=False,
            )

            total_loss = 0.0
            num_steps = 0

            for (H0_batch,) in loader:

                loss = ve_sde_dsm_loss(
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

            scheduler.step()

            avg_loss = (
                total_loss / num_steps
            )

            epoch_time = (
                time.time()
                - epoch_start
            )

            current_lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            # ----------------------------------------------------
            # Best checkpoint
            # ----------------------------------------------------

            if avg_loss < best_loss:

                best_loss = avg_loss

                torch.save(
                    net.state_dict(),
                    best_checkpoint,
                )

            # ----------------------------------------------------
            # History
            # ----------------------------------------------------

            history_entry = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "best_loss_so_far": best_loss,
            }

            history.append(
                history_entry
            )

            # ----------------------------------------------------
            # CSV
            # ----------------------------------------------------

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
            "VE-SDE score training complete. "
            f"Total time: {total_time:.2f}s"
        )

        # ========================================================
        # Summary
        # ========================================================

        summary = {

            "run_name": run_name,

            "status": "completed",

            "model": "Gaussian VE-SDE",

            "parameterization":
                "direct_score",

            "score_target":
                "-epsilon / sigma(t)",

            "time_sampling":
                "Uniform(0,1)",

            "sigma_schedule":
                "geometric",

            "loss_weighting":
                "sigma(t)^2",

            "config": cfg,

            "total_epochs": epochs,

            "total_time_sec": total_time,

            "best_loss": best_loss,

            "final_loss": (
                history[-1]["avg_loss"]
                if history
                else None
            ),

            "device": str(device),

            "checkpoint_best":
                best_checkpoint,

            "checkpoint_final":
                final_checkpoint,

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
            f"Summary written to "
            f"{summary_path}"
        )

    except Exception as e:

        logger.error(
            f"Training crashed at epoch "
            f"{len(history) + 1}: {e}"
        )

        logger.error(
            traceback.format_exc()
        )

        raise


# ================================================================
# CLI
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(
        args.config
    ) as f:

        cfg = yaml.safe_load(f)

    train(
        cfg,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()