"""
Train image epsilon/score network q_{theta_D}.

Parameterization
----------------
The network predicts the normalized Gaussian noise:

    epsilon_theta(D_sigma, sigma) ~= -epsilon

where

    D_sigma = D0 + sigma * epsilon
    epsilon ~ N(0, I)

The corresponding score is recovered as

    s_theta(D_sigma, sigma)
        = epsilon_theta(D_sigma, sigma) / sigma

because the Gaussian conditional score is

    grad_{D_sigma} log p(D_sigma | D0)
        = -epsilon / sigma

Training objective
------------------

    L = E || epsilon_theta(D_sigma, sigma) + epsilon ||^2

Noise levels are sampled continuously log-uniformly:

    log sigma ~ Uniform(log sigma_min, log sigma_max)

This matches the epsilon-prediction parameterization used by the
channel score network.

Usage
-----

    python -m score_networks.train_image_score \
        --config configs/runpod_minimal.yaml
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .ncsnpp import NCSNpp


# ================================================================
# Dataset
# ================================================================

class FFHQDataset(Dataset):

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.05,
    ):

        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"Image dataset directory does not exist: {root}"
            )

        all_files = sorted([
            f
            for f in os.listdir(root)
            if f.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                )
            )
        ])

        if len(all_files) == 0:
            raise RuntimeError(
                f"No images found in: {root}"
            )

        n_val = max(
            1,
            int(
                len(all_files)
                * val_ratio
            ),
        )

        if split == "train":

            self.files = (
                all_files[n_val:]
            )

        elif split == "val":

            self.files = (
                all_files[:n_val]
            )

        else:

            raise ValueError(
                f"Unknown split: {split}"
            )

        self.root = root

        self.transform = transforms.Compose([
            transforms.Resize(
                256
            ),

            transforms.CenterCrop(
                256
            ),

            transforms.ToTensor(),

            # Image domain becomes [-1, 1].
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ])

    def __len__(self):

        return len(
            self.files
        )

    def __getitem__(self, idx):

        path = os.path.join(
            self.root,
            self.files[idx],
        )

        with Image.open(path) as img:

            img = img.convert(
                "RGB"
            )

            return self.transform(
                img
            )


# ================================================================
# Continuous log-uniform sigma sampling
# ================================================================

def sample_log_sigma(
    batch_size,
    sigma_min,
    sigma_max,
    device,
):

    """
    Sample

        log(sigma) ~ Uniform(
            log(sigma_min),
            log(sigma_max)
        )

    Therefore sigma is log-uniform.

    This is equivalent to using

        t ~ U(0,1)

        sigma(t)
            = sigma_min
              * (sigma_max / sigma_min)^t
    """

    log_sigma_min = math.log(
        sigma_min
    )

    log_sigma_max = math.log(
        sigma_max
    )

    log_sigma = (
        torch.rand(
            batch_size,
            device=device,
        )
        * (
            log_sigma_max
            - log_sigma_min
        )
        + log_sigma_min
    )

    return log_sigma.exp()


# ================================================================
# Epsilon prediction loss
# ================================================================

def epsilon_dsm_loss_image(
    net,
    D0,
    sigma_min,
    sigma_max,
):

    """
    Gaussian corruption:

        D_sigma
            = D0 + sigma * epsilon

    epsilon ~ N(0, I)

    Network target:

        epsilon_target = -epsilon

    Loss:

        ||epsilon_pred - epsilon_target||^2

    Score recovery:

        score_pred = epsilon_pred / sigma
    """

    B = D0.shape[0]

    device = D0.device

    # ------------------------------------------------------------
    # One independently sampled sigma per image
    # ------------------------------------------------------------

    sigma = sample_log_sigma(
        B,
        sigma_min,
        sigma_max,
        device,
    )

    scale = sigma[
        :,
        None,
        None,
        None,
    ]

    # ------------------------------------------------------------
    # Gaussian noise
    # ------------------------------------------------------------

    epsilon = torch.randn_like(
        D0
    )

    # ------------------------------------------------------------
    # Forward perturbation
    #
    # D_sigma = D0 + sigma epsilon
    # ------------------------------------------------------------

    D_sigma = (
        D0
        + scale * epsilon
    )

    # ------------------------------------------------------------
    # Target = -epsilon
    # ------------------------------------------------------------

    epsilon_target = (
        -epsilon
    )

    # ------------------------------------------------------------
    # IMPORTANT
    #
    # Unlike ChannelScoreNet, NCSNpp already receives the raw
    # corrupted image here.
    #
    # Do NOT divide D_sigma by sigma unless NCSNpp was explicitly
    # designed/trained using that input convention.
    # ------------------------------------------------------------

    epsilon_pred = net(
        D_sigma,
        sigma,
    )

    # ------------------------------------------------------------
    # Direct epsilon objective
    #
    # NO sigma^2 weighting.
    # ------------------------------------------------------------

    loss = (
        epsilon_pred
        - epsilon_target
    ).pow(2).mean()

    return loss


# ================================================================
# Validation
# ================================================================

@torch.no_grad()
def validate(
    net,
    loader,
    sigma_min,
    sigma_max,
    device,
):

    net.eval()

    total_loss = 0.0

    total_samples = 0

    for D0 in loader:

        D0 = D0.to(
            device,
            non_blocking=True,
        )

        loss = epsilon_dsm_loss_image(
            net,
            D0,
            sigma_min,
            sigma_max,
        )

        batch_size = D0.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    return (
        total_loss
        / max(
            total_samples,
            1,
        )
    )


# ================================================================
# Training
# ================================================================

def train(cfg):

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ============================================================
    # Noise configuration
    # ============================================================

    sigma_min = cfg.get(
        "sigma_D_1",
        0.01,
    )

    sigma_max = cfg.get(
        "sigma_D_J",
        10.0,
    )

    print(
        f"Image sigma range: "
        f"[{sigma_min}, {sigma_max}]"
    )

    print(
        "Continuous log-sigma sampling enabled"
    )

    print(
        "Parameterization: epsilon prediction"
    )

    # ============================================================
    # Network
    # ============================================================

    net = NCSNpp(
        in_channels=3,

        base_channels=cfg.get(
            "score_base_channels",
            128,
        ),

        ch_mults=cfg.get(
            "score_ch_mults",
            (1, 2, 2, 2),
        ),

        num_res_blocks=cfg.get(
            "score_num_res_blocks",
            2,
        ),

        attn_resolutions=cfg.get(
            "score_attn_resolutions",
            (16,),
        ),

        dropout=cfg.get(
            "score_dropout",
            0.1,
        ),

        img_size=256,
    ).to(device)

    trainable_params = sum(
        p.numel()
        for p in net.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ============================================================
    # Optimizer
    # ============================================================

    lr = cfg.get(
        "score_lr",
        2e-4,
    )

    weight_decay = cfg.get(
        "score_weight_decay",
        1e-4,
    )

    optimizer = optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ============================================================
    # Epochs
    # ============================================================

    epochs = cfg.get(
        "score_epochs",
        200,
    )

    # ============================================================
    # Scheduler
    # ============================================================

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

    data_root = cfg.get(
        "data_root",
        "data/ffhq256",
    )

    val_ratio = cfg.get(
        "score_val_ratio",
        0.05,
    )

    train_ds = FFHQDataset(
        data_root,
        split="train",
        val_ratio=val_ratio,
    )

    val_ds = FFHQDataset(
        data_root,
        split="val",
        val_ratio=val_ratio,
    )

    print(
        f"Training images: "
        f"{len(train_ds)}"
    )

    print(
        f"Validation images: "
        f"{len(val_ds)}"
    )

    # ============================================================
    # DataLoader
    # ============================================================

    batch_size = cfg.get(
        "score_image_batch_size",
        cfg.get(
            "score_batch_size",
            8,
        ),
    )

    num_workers = cfg.get(
        "score_num_workers",
        4,
    )

    loader = DataLoader(
        train_ds,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            num_workers > 0
        ),

        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=(
            device.type == "cuda"
        ),

        persistent_workers=(
            num_workers > 0
        ),
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Steps per epoch: "
        f"{len(loader)}"
    )

    # ============================================================
    # Checkpoint directory
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
        "image_score_best.pt",
    )

    final_checkpoint = os.path.join(
        ckpt_dir,
        "image_score_final.pt",
    )

    best_val_loss = float(
        "inf"
    )

    # ============================================================
    # Training loop
    # ============================================================

    for epoch in range(
        epochs
    ):

        net.train()

        total_loss = 0.0
        total_samples = 0

        progress = tqdm(
            loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{epochs}"
            ),
            leave=False,
        )

        for D0 in progress:

            D0 = D0.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ----------------------------------------------------
            # Epsilon prediction loss
            # ----------------------------------------------------

            loss = epsilon_dsm_loss_image(
                net,
                D0,
                sigma_min,
                sigma_max,
            )

            # ----------------------------------------------------
            # Backpropagation
            # ----------------------------------------------------

            loss.backward()

            # ----------------------------------------------------
            # Gradient clipping
            # ----------------------------------------------------

            nn.utils.clip_grad_norm_(
                net.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_n = D0.shape[0]

            total_loss += (
                loss.item()
                * batch_n
            )

            total_samples += (
                batch_n
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        scheduler.step()

        # --------------------------------------------------------
        # Average training loss
        # --------------------------------------------------------

        train_loss = (
            total_loss
            / max(
                total_samples,
                1,
            )
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        val_loss = validate(
            net,
            val_loader,
            sigma_min,
            sigma_max,
            device,
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"Epoch "
            f"{epoch + 1:4d}/"
            f"{epochs} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"lr={current_lr:.8f} | "
            f"best={best_val_loss:.6f}"
        )

        # --------------------------------------------------------
        # Save BEST according to validation loss
        # --------------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            torch.save(
                net.state_dict(),
                best_checkpoint,
            )

            print(
                f"  -> saved best checkpoint "
                f"(val={val_loss:.6f})"
            )

    # ============================================================
    # Final checkpoint
    # ============================================================

    torch.save(
        net.state_dict(),
        final_checkpoint,
    )

    print()
    print("=" * 80)
    print("IMAGE SCORE TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{best_checkpoint}"
    )

    print(
        f"Final checkpoint: "
        f"{final_checkpoint}"
    )


# ================================================================
# Main
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

        cfg = yaml.safe_load(
            f
        )

    train(
        cfg
    )


if __name__ == "__main__":
    main()