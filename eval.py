"""
Main evaluation script for Blind-MIMOSC.

Runs PVD and all baselines over Monte Carlo trials and reports metrics.

Usage:
    python eval.py --config configs/rayleigh_4x1_Nu4.yaml --snr 10
    python eval.py --config configs/rayleigh_4x1_Nu4.yaml --all-snr
    python eval.py --config configs/stable_noise.yaml --snr 10 --stable
"""
import argparse
import os
import math
import csv
import time
import traceback
import logging
from datetime import datetime
from unittest import loader
from pathlib import Path
from torchvision import transforms
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

from encoder.swin_jscc import DJSCCEncoder, DJSCCDecoder
from score_networks.ncsnpp import NCSNpp, ChannelScoreNet, ChannelScoreNet2ndOrder, ImageScoreNet2ndOrder
from pvd.pvd import PVDSolver
from channels.rayleigh import generate_rayleigh_channel, apply_channel, noise_schedule_exponential
from channels.cdl_c import load_cdlc_channel
from baselines.djscc_mimo import DJSCCMIMOBaseline
from baselines.dps_mimo import DPSMIMOBaseline
from metrics.ms_ssim import ms_ssim
from metrics.nmse import nmse_db


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, run_name: str):
    """Set up file logging + CSV metrics logger. Returns (logger, csv_path, log_path)."""
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{run_name}.log")
    csv_path = os.path.join(log_dir, f"{run_name}_metrics.csv")

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "snr_db", "method", "metric", "mean", "std", "n_samples", "timestamp"
        ])

    return logger, csv_path, log_path


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_encoder(cfg: dict, device: torch.device, logger=None) -> tuple:
    log = logger.info if logger else print
    enc = DJSCCEncoder(
        embed_dim=cfg.get("embed_dim", 96),
        depths=cfg.get("depths", [2, 2, 6, 2]),
        num_heads=cfg.get("num_heads", [3, 6, 12, 24]),
        Nt=cfg.get("Nt", 1),
        K=cfg.get("K", 192),
        T=cfg.get("T", 24),
        Nu=cfg.get("Nu", 1),
        power=cfg.get("power", 1.0),
    ).to(device)
    dec = DJSCCDecoder(
        embed_dim=cfg.get("embed_dim", 96),
        depths=cfg.get("dec_depths", [2, 6, 2, 2]),
        num_heads=cfg.get("dec_num_heads", [24, 12, 6, 3]),
        Nt=cfg.get("Nt", 1),
        K=cfg.get("K", 192),
        T=cfg.get("T", 24),
    ).to(device)

    ckpt_dir = cfg.get("encoder_ckpt_dir", "encoder/checkpoints")
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc.load_state_dict(ckpt["encoder"])
        dec.load_state_dict(ckpt["decoder"])
        log(f"Loaded encoder/decoder from {ckpt_path}")
    else:
        log(f"WARNING: No encoder checkpoint found at {ckpt_path}. Using random weights.")
    return enc.eval(), dec.eval()


def load_score_nets(cfg: dict, device: torch.device, logger=None) -> tuple:
    log = logger.info if logger else print
    Nr, Nt, K = cfg.get("Nr", 4), cfg.get("Nt", 1), cfg.get("K", 192)
    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")

    S_theta_H = ChannelScoreNet(Nr=Nr, Nt=Nt, K=K).to(device)
    S_theta_D = NCSNpp(
        in_channels=3,
        base_channels=cfg.get("score_base_channels", 128),
        ch_mults=tuple(cfg.get("score_ch_mults", [1, 2, 2, 2])),
        num_res_blocks=cfg.get("score_num_res_blocks", 2),
        attn_resolutions=tuple(cfg.get("score_attn_resolutions", [16])),
        dropout=cfg.get("score_dropout", 0.1),
    ).to(device)

    ch_ckpt = os.path.join(ckpt_dir, "channel_score_best.pt")
    img_ckpt = os.path.join(ckpt_dir, "image_score_best.pt")

    if os.path.exists(ch_ckpt):
        S_theta_H.load_state_dict(torch.load(ch_ckpt, map_location=device, weights_only=True))
        log(f"Loaded channel score from {ch_ckpt}")
    else:
        log(f"WARNING: No channel score checkpoint at {ch_ckpt}.")

    if os.path.exists(img_ckpt):
        S_theta_D.load_state_dict(torch.load(img_ckpt, map_location=device, weights_only=True))
        log(f"Loaded image score from {img_ckpt}")
    else:
        log(f"WARNING: No image score checkpoint at {img_ckpt}.")

    s_theta_H = ChannelScoreNet2ndOrder(Nr=Nr, Nt=Nt, K=K).to(device)
    s_theta_D = ImageScoreNet2ndOrder().to(device)

    ch_2nd = os.path.join(ckpt_dir, "channel_score2nd_best.pt")
    img_2nd = os.path.join(ckpt_dir, "image_score2nd_best.pt")
    if os.path.exists(ch_2nd):
        s_theta_H.load_state_dict(torch.load(ch_2nd, map_location=device, weights_only=True))
    if os.path.exists(img_2nd):
        s_theta_D.load_state_dict(torch.load(img_2nd, map_location=device, weights_only=True))

    if not cfg.get("use_second_order", True):
        s_theta_H = None
        s_theta_D = None

    return S_theta_H.eval(), S_theta_D.eval(), s_theta_H, s_theta_D


# ---------------------------------------------------------------------------
# Channel generation
# ---------------------------------------------------------------------------

def get_channel(cfg: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    channel = cfg.get("channel", "rayleigh")
    Nr, Nt, K, Nu = cfg["Nr"], cfg["Nt"], cfg["K"], cfg.get("Nu", 1)
    if channel == "rayleigh":
        return generate_rayleigh_channel(batch_size, Nu, Nr, Nt, K, device)
    elif channel == "cdl_c":
        path = cfg.get("cdl_c_path", "data/cdl_c_channels.npy")
        return load_cdlc_channel(path, batch_size, Nr, Nt, K, device)
    else:
        raise ValueError(f"Unknown channel type: {channel}")


# ---------------------------------------------------------------------------
# Single-trial evaluation
# ---------------------------------------------------------------------------

def evaluate_pvd(
    pvd: PVDSolver,
    D0: torch.Tensor,
    H0: torch.Tensor,
    Y: torch.Tensor,
    sigma_n: float,
) -> Dict[str, float]:
    H_hat, D_hat = pvd.solve(Y, verbose=False)

    D0_01 = (D0.clamp(-1, 1) + 1) / 2
    D_hat_01 = (D_hat.clamp(-1, 1) + 1) / 2

    ms_ssim_val = ms_ssim(D0_01, D_hat_01).mean().item()
    nmse_val = nmse_db(H_hat, H0[:, 0]).item()

    return {
        "ms_ssim": ms_ssim_val,
        "nmse_db": nmse_val,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_at_snr(cfg: dict, snr_db: float, args, device: torch.device,
                    use_analytical_channel_prior: bool = False, logger=None) -> Dict[str, List]:
    log = logger.info if logger else print
    log_err = logger.error if logger else print

    Nr, Nt, K, T, Nu = cfg["Nr"], cfg["Nt"], cfg["K"], cfg["T"], cfg.get("Nu", 1)
    n_trials = cfg.get("n_trials", 300)
    batch_size = args.batch_size

    enc, dec = load_encoder(cfg, device, logger)
    S_theta_H, S_theta_D, s_theta_H, s_theta_D = load_score_nets(cfg, device, logger)

    snr_linear = 10 ** (snr_db / 10.0)
    sigma_n = math.sqrt(1.0 / snr_linear)

    pvd = PVDSolver(
        f_gamma=enc, S_theta_H=S_theta_H, S_theta_D=S_theta_D,
        s_theta_H=s_theta_H, s_theta_D=s_theta_D,
        sigma_n=sigma_n, Nr=Nr, Nt=Nt, K=K, T=T, Nu=Nu,
        J=cfg.get("J", 30), J_in=cfg.get("J_in", 20),
        zeta_H=cfg.get("zeta_H", 1.0), zeta_D=cfg.get("zeta_D", 1.0),
        device=device,
        use_second_order=cfg.get("use_second_order", True),
        use_analytical_channel_prior=use_analytical_channel_prior,
    )

    djscc_perfect = DJSCCMIMOBaseline(enc, dec, Nr, Nt, K, T, Nu, perfect_csi=True)
    djscc_pilot = DJSCCMIMOBaseline(enc, dec, Nr, Nt, K, T, Nu, perfect_csi=False)

    results = {
        "pvd": {"ms_ssim": [], "nmse_db": []},
        "djscc_perfect": {"ms_ssim": [], "nmse_db": []},
        "djscc_pilot": {"ms_ssim": [], "nmse_db": []},
    }
    fail_counts = {"pvd": 0, "djscc_perfect": 0, "djscc_pilot": 0}

    n_done = 0
    pbar = tqdm(total=n_trials, desc=f"SNR={snr_db}dB")

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    ])

    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class FlatFolderDataset(Dataset):
        def __init__(self, root, transform=None):
            self.paths = list(Path(root).glob("*.png"))
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0
        

    dataset = FlatFolderDataset("data/mnist256/all", transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    data_iter = iter(loader)

    trial_idx = 0
    while n_done < n_trials:
        bs = min(batch_size, n_trials - n_done)

        H0 = get_channel(cfg, bs, device)

        try:
            D0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            D0, _ = next(data_iter)

        D0 = D0.to(device)

        X = enc(D0)
        Y, _ = apply_channel(H0, X, snr_db)

        # PVD
        try:
            r_pvd = evaluate_pvd(pvd, D0, H0, Y, sigma_n)
            for k in r_pvd:
                results["pvd"][k].append(r_pvd[k])
        except Exception as e:
            fail_counts["pvd"] += 1
            log_err(f"[SNR={snr_db}dB][trial {trial_idx}] PVD failed: {e}")
            log_err(traceback.format_exc())

        # DJSCC-perfect
        try:
            D_hat_p, H_hat_p = djscc_perfect.run(D0, H0, snr_db, sigma_n)
            D0_01 = (D0 + 1) / 2
            D_hat_p_01 = (D_hat_p.clamp(-1,1) + 1) / 2
            results["djscc_perfect"]["ms_ssim"].append(ms_ssim(D0_01, D_hat_p_01).mean().item())
            results["djscc_perfect"]["nmse_db"].append(0.0)
        except Exception as e:
            fail_counts["djscc_perfect"] += 1
            log_err(f"[SNR={snr_db}dB][trial {trial_idx}] DJSCC-perfect failed: {e}")
            log_err(traceback.format_exc())

        # DJSCC-pilot
        try:
            D_hat_pi, H_hat_pi = djscc_pilot.run(D0, H0, snr_db, sigma_n)
            D0_01 = (D0 + 1) / 2
            D_hat_pi_01 = (D_hat_pi.clamp(-1,1) + 1) / 2
            results["djscc_pilot"]["ms_ssim"].append(ms_ssim(D0_01, D_hat_pi_01).mean().item())
            results["djscc_pilot"]["nmse_db"].append(nmse_db(H_hat_pi, H0[:, 0]).item())
        except Exception as e:
            fail_counts["djscc_pilot"] += 1
            log_err(f"[SNR={snr_db}dB][trial {trial_idx}] DJSCC-pilot failed: {e}")
            log_err(traceback.format_exc())

        n_done += bs
        trial_idx += 1
        pbar.update(bs)

    pbar.close()

    for method, count in fail_counts.items():
        if count > 0:
            log(f"SNR={snr_db}dB | {method}: {count} trial(s) failed out of {n_trials}")

    return results


def summarize(results: Dict) -> Dict:
    summary = {}
    for method, metrics in results.items():
        summary[method] = {}
        for metric, vals in metrics.items():
            if vals:
                arr = np.array(vals)
                summary[method][metric] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n_samples": len(vals),
                }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--all-snr", action="store_true")
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--analytical-channel-prior", action="store_true",
                        help="Use exact Rayleigh score (bypasses trained channel score net)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # --- Logging setup ---
    run_name = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    run_name = f"eval_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, csv_path, log_path = setup_logging(args.log_dir, run_name)

    logger.info(f"Starting eval run: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Config contents: {json.dumps(cfg, indent=2, default=str)}")
    logger.info(f"Args: {vars(args)}")

    run_start = time.time()

    if args.all_snr:
        snr_list = cfg.get("snr_db_range", [-5, 0, 5, 10, 15, 20])
    else:
        snr_list = [args.snr]

    all_results = {}
    try:
        for snr_db in snr_list:
            logger.info(f"{'='*50}")
            logger.info(f"Evaluating at SNR = {snr_db} dB")
            logger.info(f"{'='*50}")
            snr_start = time.time()

            results = evaluate_at_snr(cfg, snr_db, args, device,
                                      use_analytical_channel_prior=args.analytical_channel_prior,
                                      logger=logger)
            summary = summarize(results)
            all_results[str(snr_db)] = summary

            snr_time = time.time() - snr_start
            logger.info(f"SNR={snr_db}dB completed in {snr_time:.2f}s")

            # Console + log table
            table_lines = [f"\nSNR = {snr_db} dB Results:",
                            f"{'Method':<20} {'MS-SSIM':>10} {'NMSE(dB)':>10}",
                            "-" * 42]
            for method, metrics in summary.items():
                ms = metrics.get("ms_ssim", {})
                nm = metrics.get("nmse_db", {})
                ms_str = f"{ms.get('mean', 0):.4f}±{ms.get('std', 0):.4f}" if ms else "N/A"
                nm_str = f"{nm.get('mean', 0):.2f}±{nm.get('std', 0):.2f}" if nm else "N/A"
                table_lines.append(f"{method:<20} {ms_str:>10} {nm_str:>10}")
            logger.info("\n".join(table_lines))

            # Write per-method/per-metric rows to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for method, metrics in summary.items():
                    for metric, stats in metrics.items():
                        writer.writerow([
                            snr_db, method, metric,
                            f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
                            stats["n_samples"], datetime.now().isoformat()
                        ])

        total_time = time.time() - run_start

        # Save raw results JSON (as before, for compatibility)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

        # Save full run summary alongside logs
        run_summary = {
            "run_name": run_name,
            "status": "completed",
            "config_path": args.config,
            "config": cfg,
            "args": vars(args),
            "total_time_sec": total_time,
            "device": str(device),
            "results": all_results,
        }
        summary_path = os.path.join(args.log_dir, f"{run_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2, default=str)

        logger.info(f"Eval complete. Total time: {total_time:.2f}s")
        logger.info(f"Summary JSON: {summary_path}")
        logger.info(f"Metrics CSV: {csv_path}")
        logger.info(f"Full log: {log_path}")

    except Exception as e:
        logger.error(f"Eval crashed: {e}")
        logger.error(traceback.format_exc())
        crash_summary = {
            "run_name": run_name,
            "status": "crashed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "partial_results": all_results,
        }
        crash_path = os.path.join(args.log_dir, f"{run_name}_CRASHED.json")
        with open(crash_path, "w") as f:
            json.dump(crash_summary, f, indent=2, default=str)
        logger.error(f"Crash report written to {crash_path}")
        raise


if __name__ == "__main__":
    main()