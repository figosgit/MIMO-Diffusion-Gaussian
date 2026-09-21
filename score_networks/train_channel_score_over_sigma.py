"""
Train channel score network q_{theta_H} using direct-score parameterization.

Given
    H_j = H_0 + sigma_j * epsilon,

the network learns
    s_theta(H_j, sigma_j) ~= -epsilon / sigma_j.

Usage:
    python -m score_networks.train_channel_score_direct --config configs/runpod_minimal.yaml
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


def setup_logging(log_dir: str, run_name: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.log")
    csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console_handler)

    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "avg_loss", "lr", "epoch_time_sec",
            "best_loss_so_far", "timestamp"
        ])

    return logger, csv_path, log_path


def sample_log_sigma(batch_size, sigma_min, sigma_max, device):
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    log_sigma = (
        torch.rand(batch_size, device=device)
        * (log_sigma_max - log_sigma_min)
        + log_sigma_min
    )
    return log_sigma.exp()


def direct_score_dsm_loss(net, H0, sigma_min, sigma_max):
    """
    Direct-score DSM.

    H_j = H_0 + sigma * epsilon
    target = -epsilon / sigma

    We keep the same H_j / sigma input normalization as the epsilon model.

    Loss:
        E[sigma^2 * ||s_theta - (-epsilon/sigma)||^2]

    The sigma^2 weighting prevents tiny sigma values from dominating the
    optimization purely because the direct score scales as 1/sigma.
    """
    batch_size = H0.shape[0]
    device = H0.device

    sigma = sample_log_sigma(batch_size, sigma_min, sigma_max, device)

    eps_real = torch.randn_like(H0.real)
    eps_imag = torch.randn_like(H0.imag)

    sigma_3d = sigma[:, None, None]

    H_j_real = H0.real + sigma_3d * eps_real
    H_j_imag = H0.imag + sigma_3d * eps_imag
    H_j = torch.stack([H_j_real, H_j_imag], dim=1)

    H_j_norm = H_j / sigma[:, None, None, None]

    score_target = torch.stack(
        [
            -eps_real / sigma_3d,
            -eps_imag / sigma_3d,
        ],
        dim=1,
    )

    score_pred = net(H_j_norm, sigma)

    sigma_4d = sigma[:, None, None, None]
    loss = (
        sigma_4d.pow(2)
        * (score_pred - score_target).pow(2)
    ).mean()

    return loss


def train(cfg: dict, config_path: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = cfg.get("log_dir", "logs")
    base_name = (
        cfg.get("run_name")
        or os.path.splitext(os.path.basename(config_path))[0]
        or "run"
    )
    run_name = f"{base_name}_direct_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger, csv_path, log_path = setup_logging(log_dir, run_name)

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    Nr = cfg.get("Nr", 4)
    Nt = cfg.get("Nt", 1)
    K = cfg.get("K", 192)
    Nu = cfg.get("Nu", 1)

    sigma_min = cfg.get("sigma_H_1", 0.01)
    sigma_max = cfg.get("sigma_H_J", 10.0)

    if sigma_min <= 0:
        raise ValueError(f"sigma_H_1 must be > 0, got {sigma_min}")
    if sigma_max <= sigma_min:
        raise ValueError(
            f"sigma_H_J must be > sigma_H_1, got {sigma_min} -> {sigma_max}"
        )

    logger.info(f"Channel dimensions: Nr={Nr}, Nt={Nt}, K={K}, Nu={Nu}")
    logger.info(f"Sigma range: [{sigma_min}, {sigma_max}]")
    logger.info("Continuous log-sigma sampling enabled")
    logger.info("Parameterization: direct score target = -epsilon / sigma")

    net = ChannelScoreNet(
        Nr=Nr,
        Nt=Nt,
        K=K,
        hidden_dim=cfg.get("channel_hidden_dim", 1024),
        num_layers=cfg.get("channel_num_layers", 8),
        time_dim=cfg.get("channel_time_dim", 512),
    ).to(device)

    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {num_parameters:,}")

    learning_rate = cfg.get("score_lr", 2e-4)
    optimizer = optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        weight_decay=cfg.get("score_weight_decay", 1e-4),
    )

    epochs = cfg.get("score_epochs", 500)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=cfg.get("score_lr_min", 1e-5),
    )

    batch_size = cfg.get("score_batch_size", 256)
    samples_per_epoch = cfg.get("score_samples_per_epoch", batch_size * 20)
    if samples_per_epoch < batch_size:
        samples_per_epoch = batch_size

    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Separate checkpoint names: do not overwrite epsilon-model checkpoints.
    best_checkpoint = os.path.join(ckpt_dir, "channel_score_direct_best.pt")
    final_checkpoint = os.path.join(ckpt_dir, "channel_score_direct_final.pt")

    best_loss = float("inf")
    history = []
    run_start = time.time()

    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Samples per epoch: {samples_per_epoch}")
    logger.info(f"Steps per epoch: {math.ceil(samples_per_epoch / batch_size)}")

    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            net.train()

            H0 = generate_rayleigh_channel(
                samples_per_epoch, Nu, Nr, Nt, K, device
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
                loss = direct_score_dsm_loss(
                    net, H0_batch, sigma_min, sigma_max
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_steps += 1

            scheduler.step()

            avg_loss = total_loss / num_steps if num_steps else float("inf")
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(net.state_dict(), best_checkpoint)

            history_entry = {
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "best_loss_so_far": best_loss,
            }
            history.append(history_entry)

            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch + 1,
                    f"{avg_loss:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    f"{best_loss:.6f}",
                    datetime.now().isoformat(),
                ])

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"loss={avg_loss:.6f} | lr={current_lr:.8f} | "
                f"best={best_loss:.6f} | time={epoch_time:.2f}s"
            )

        torch.save(net.state_dict(), final_checkpoint)
        total_time = time.time() - run_start

        logger.info(
            f"Direct-score channel network training complete. "
            f"Total time: {total_time:.2f}s"
        )

        summary = {
            "run_name": run_name,
            "status": "completed",
            "parameterization": "direct_score",
            "score_target": "-epsilon / sigma",
            "loss_weighting": "sigma^2",
            "config": cfg,
            "total_epochs": epochs,
            "total_time_sec": total_time,
            "best_loss": best_loss,
            "final_loss": history[-1]["avg_loss"] if history else None,
            "device": str(device),
            "checkpoint_best": best_checkpoint,
            "checkpoint_final": final_checkpoint,
            "history": history,
        }

        summary_path = os.path.join(log_dir, f"{run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary written to {summary_path}")
        logger.info(f"Per-epoch metrics CSV: {csv_path}")
        logger.info(f"Full log: {log_path}")

    except Exception as e:
        logger.error(f"Training crashed at epoch {len(history) + 1}: {e}")
        logger.error(traceback.format_exc())

        crash_path = os.path.join(log_dir, f"{run_name}_CRASHED.json")
        with open(crash_path, "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "status": "crashed",
                    "parameterization": "direct_score",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "epochs_completed": len(history),
                    "best_loss": best_loss,
                    "history": history,
                },
                f,
                indent=2,
                default=str,
            )
        logger.error(f"Crash report written to {crash_path}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
