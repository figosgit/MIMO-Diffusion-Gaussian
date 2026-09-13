"""
Train image score network q_{θ_D}.

Learns ∇_{D_j} ln q(D_j | σ_j) via denoising score matching (DSM):
    L(θ) = E_{D0, ε, j} [ ||σ_j * s_θ(D0 + σ_j ε, σ_j) + ε||^2 ]

D0 is drawn from the FFHQ-256 dataset.

Usage:
    python -m score_networks.train_image_score --config configs/rayleigh_4x1_Nu4.yaml
"""
import argparse
import os
import math
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from .ncsnpp import NCSNpp, get_sigmas
from channels.rayleigh import noise_schedule_exponential


class FFHQDataset(Dataset):
    def __init__(self, root: str, split: str = "train", val_ratio: float = 0.05):
        all_files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        n_val = max(1, int(len(all_files) * val_ratio))
        self.files = all_files[n_val:] if split == "train" else all_files[:n_val]
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert("RGB")
        return self.transform(img)


def dsm_loss_image(
    net: nn.Module,
    D0: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B = D0.shape[0]
    j = torch.randint(1, len(sigmas), (B,), device=device)
    sigma_j = sigmas[j]  # (B,)
    eps = torch.randn_like(D0)
    scale = sigma_j[:, None, None, None]

    D_j = D0 + scale * eps

    # Epsilon predictor: target is -eps (not -eps/sigma).
    # sigma² weighting ensures high-sigma samples contribute meaningfully to the loss,
    # and the raw MLP only needs to produce O(1) outputs (sigma-independent).
    # This is the same convention used by ChannelScoreNet.
    eps_target = -eps
    eps_pred = net(D_j, sigma_j)
    loss = (scale ** 2 * (eps_pred - eps_target) ** 2).mean()
    return loss


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    J = cfg.get("J", 30)
    sigma_1 = cfg.get("sigma_D_1", 0.01)
    sigma_J = cfg.get("sigma_D_J", 100.0)
    sigmas = noise_schedule_exponential(sigma_1, sigma_J, J, device)

    net = NCSNpp(
        in_channels=3,
        base_channels=cfg.get("score_base_channels", 128),
        ch_mults=cfg.get("score_ch_mults", (1, 2, 2, 2)),
        num_res_blocks=cfg.get("score_num_res_blocks", 2),
        attn_resolutions=cfg.get("score_attn_resolutions", (16,)),
        dropout=cfg.get("score_dropout", 0.1),
        img_size=256,
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=cfg.get("score_lr", 2e-4))
    # CosineAnnealingLR decays smoothly to eta_min, avoiding the abrupt drops
    # that cause loss spikes when sigma²-weighted gradients are large.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.get("score_epochs", 200),
        eta_min=cfg.get("score_lr_min", 1e-5),
    )

    data_root = cfg.get("data_root", "data/ffhq256")
    train_ds = FFHQDataset(data_root, split="train")
    loader = DataLoader(train_ds, batch_size=cfg.get("score_batch_size", 8),
                        shuffle=True, num_workers=4, pin_memory=True)

    ckpt_dir = cfg.get("score_ckpt_dir", "score_networks/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    epochs = cfg.get("score_epochs", 200)
    best_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        total = 0.0
        for D0 in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            D0 = D0.to(device)
            loss = dsm_loss_image(net, D0, sigmas, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        avg = total / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(net.state_dict(), os.path.join(ckpt_dir, "image_score_best.pt"))

    torch.save(net.state_dict(), os.path.join(ckpt_dir, "image_score_final.pt"))
    print("Image score network training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
