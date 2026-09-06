"""
Train channel score network q_{θ_H}.

Learns ∇_{H_j} ln q(H_j | σ_j) via denoising score matching (DSM):
    L(θ) = E_{H0, ε, j} [ ||σ_j * s_θ(H0 + σ_j ε, σ_j) + ε||^2 ]

H0 is drawn from the Rayleigh channel prior (i.i.d. CN(0, I)).

Usage:
    python -m score_networks.train_channel_score --config configs/rayleigh_4x1_Nu4.yaml
"""
import argparse
import os
import math
import csv
import json
import time
import traceback
import logging
from datetime import datetime

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .ncsnpp import ChannelScoreNet, ChannelScoreNet2ndOrder, get_sigmas
from channels.rayleigh import generate_rayleigh_channel, noise_schedule_exponential


def setup_logging(log_dir: str, run_name: str):
    """Set up file logging + CSV metrics logger. Returns (logger, csv_path)."""
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{run_name}.log")
    csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if called twice

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

    # Init CSV with header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "lr", "epoch_time_sec", "best_loss_so_far", "timestamp"])

    return logger, csv_path, log_path


def dsm_loss(net, H0, sigmas, device):
    B = H0.shape[0]
    j = torch.randint(1, len(sigmas), (B,), device=device)
    sigma_j = sigmas[j]

    eps_real = torch.randn_like(H0.real)
    eps_imag = torch.randn_like(H0.imag)

    scale = sigma_j[:, None, None]
    scale_4d = sigma_j[:, None, None, None]

    # Noisy input
    H_j_real = H0.real + scale * eps_real
    H_j_imag = H0.imag + scale * eps_imag
    H_j = torch.stack([H_j_real, H_j_imag], dim=1)

    # ✅ NORMALIZE input
    H_j_norm = H_j / scale_4d

    # ✅ SIMPLE target
    score_target = torch.stack([-eps_real, -eps_imag], dim=1)

    # Forward
    score_pred = net(H_j_norm, sigma_j)

    # sigma^2 weighting cancels the 1/sigma^2 division inside ChannelScoreNet.forward(),
    # making the effective gradient w.r.t. the MLP weights independent of sigma.
    # Without this, high-sigma samples (e.g. sigma=100) contribute 1e8x less gradient
    # than low-sigma samples, so the network never learns at high noise levels.
    loss = (scale_4d ** 2 * (score_pred - score_target) ** 2).mean()

    return loss


def train(cfg: dict, config_path: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logging setup ---
    log_dir = cfg.get("log_dir", "logs")
    run_name = cfg.get("run_name") or os.path.splitext(os.path.basename(config_path))[0] or "run"
    run_name = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, csv_path, log_path = setup_logging(log_dir, run_name)

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    Nr = cfg.get("Nr", 4)
    Nt = cfg.get("Nt", 1)
    K = cfg.get("K", 192)
    Nu = cfg.get("Nu", 1)
    J = cfg.get("J", 30)
    sigma_1 = cfg.get("sigma_H_1", 0.01)
    sigma_J = cfg.get("sigma_H_J", 100.0)

    sigmas = noise_schedule_exponential(sigma_1, sigma_J, J, device)

    net = ChannelScoreNet(Nr=Nr, Nt=Nt, K=K).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.get("score_lr", 2e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("score_epochs", 200),
        eta_min=cfg.get("score_lr_min", 1e-5),
    )

    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = cfg.get("score_epochs", 200)
    batch_size = cfg.get("score_batch_size", 256)

    best_loss = float("inf")
    history = []  # in-memory record, also dumped to JSON at the end
    run_start = time.time()

    try:
        for epoch in range(epochs):
            epoch_start = time.time()
            net.train()
            # Generate fresh channel samples each epoch
            H0 = generate_rayleigh_channel(batch_size * 10, Nu, Nr, Nt, K, device)
            H0 = H0[:, 0]  # (B, NrK, NtK)
            ds = TensorDataset(H0)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            total = 0.0
            for (h,) in loader:
                loss = dsm_loss(net, h, sigmas, device)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(loader)
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]

            if avg < best_loss:
                best_loss = avg
                torch.save(net.state_dict(), os.path.join(ckpt_dir, "channel_score_best.pt"))

            # --- Log every epoch to CSV ---
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f"{avg:.6f}", f"{current_lr:.8f}",
                    f"{epoch_time:.2f}", f"{best_loss:.6f}",
                    datetime.now().isoformat()
                ])

            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "best_loss_so_far": best_loss,
            })

            # Log to console/file every epoch (not just every 20) so overnight
            # runs have a full record; console spam is fine since it's also going to file.
            logger.info(
                f"Epoch {epoch+1}/{epochs} | loss={avg:.6f} | lr={current_lr:.8f} "
                f"| best={best_loss:.6f} | time={epoch_time:.2f}s"
            )

        torch.save(net.state_dict(), os.path.join(ckpt_dir, "channel_score_final.pt"))
        total_time = time.time() - run_start
        logger.info(f"Channel score network training complete. Total time: {total_time:.2f}s")

        # --- Final summary JSON ---
        summary = {
            "run_name": run_name,
            "status": "completed",
            "config": cfg,
            "total_epochs": epochs,
            "total_time_sec": total_time,
            "best_loss": best_loss,
            "final_loss": history[-1]["avg_loss"] if history else None,
            "device": str(device),
            "checkpoint_best": os.path.join(ckpt_dir, "channel_score_best.pt"),
            "checkpoint_final": os.path.join(ckpt_dir, "channel_score_final.pt"),
            "history": history,
        }
        summary_path = os.path.join(log_dir, f"{run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary written to {summary_path}")
        logger.info(f"Per-epoch metrics CSV: {csv_path}")
        logger.info(f"Full log: {log_path}")

    except Exception as e:
        # Make sure a crash overnight still leaves a readable record of what happened
        logger.error(f"Training crashed at epoch {len(history)+1}: {e}")
        logger.error(traceback.format_exc())

        crash_summary = {
            "run_name": run_name,
            "status": "crashed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "epochs_completed": len(history),
            "best_loss": best_loss,
            "history": history,
        }
        crash_path = os.path.join(log_dir, f"{run_name}_CRASHED.json")
        with open(crash_path, "w") as f:
            json.dump(crash_summary, f, indent=2, default=str)
        logger.error(f"Crash report written to {crash_path}")
        raise  # still fail loudly, but now with files on disk to inspect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, config_path=args.config)


if __name__ == "__main__":
    main()