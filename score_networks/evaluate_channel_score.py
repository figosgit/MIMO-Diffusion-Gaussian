"""
Evaluate the channel score network.

The current channel network uses epsilon parameterization:

    H_sigma = H_0 + sigma * epsilon

    epsilon_theta(H_sigma, sigma) ~= E[-epsilon | H_sigma]

and therefore:

    score_theta(H_sigma, sigma)
        = epsilon_theta(H_sigma, sigma) / sigma

For an i.i.d. Gaussian/Rayleigh channel, the noisy marginal remains
Gaussian, so the exact marginal score is available analytically.

If each real/imaginary component of H_0 has variance v:

    H_sigma ~ N(0, v + sigma^2)

and therefore:

    score*(H_sigma, sigma)
        = -H_sigma / (v + sigma^2)

This evaluator reports:

1. Epsilon MSE against the sampled -epsilon target.
   This is a DSM training diagnostic, NOT the main quality metric.

2. Analytical score MSE.

3. Analytical score relative error.

4. Score cosine similarity.

5. Mean metrics across all evaluation sigmas.

Usage:

    python -m score_networks.evaluate_channel_score \
        --config configs/runpod_minimal.yaml \
        --checkpoint score_networks/checkpoints/channel_score_best.pt

    python -m score_networks.evaluate_channel_score \
        --config configs/runpod_minimal.yaml \
        --checkpoint score_networks/checkpoints/channel_score_final.pt
"""

import argparse
import json
import math

import torch
import yaml

from .ncsnpp import ChannelScoreNet
from channels.rayleigh import generate_rayleigh_channel


def build_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_sigmas: int,
    device: torch.device,
) -> torch.Tensor:
    """Create logarithmically spaced evaluation noise levels."""
    return torch.exp(
        torch.linspace(
            math.log(sigma_min),
            math.log(sigma_max),
            num_sigmas,
            device=device,
        )
    )


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
):
    """Load either a raw state_dict or a checkpoint dictionary."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


@torch.no_grad()
def estimate_channel_variance(
    H0: torch.Tensor,
) -> float:
    """
    Estimate the variance of one real-valued channel component.

    H0 has shape:

        (B, 2, Nr*K, Nt*K)

    where channel dimension 1 is [real, imag].
    """
    real_var = H0[:, 0].var(unbiased=False)
    imag_var = H0[:, 1].var(unbiased=False)

    variance = 0.5 * (real_var + imag_var)

    return variance.item()


@torch.no_grad()
def evaluate_sigma(
    net: torch.nn.Module,
    H0: torch.Tensor,
    sigma: float,
    channel_variance: float,
):
    """
    Evaluate one noise level on one batch of clean channels.

    The batch must be small enough to fit comfortably on the GPU.
    """
    device = H0.device
    batch_size = H0.shape[0]

    sigma_tensor = torch.full(
        (batch_size,),
        sigma,
        device=device,
        dtype=H0.dtype,
    )

    eps_real = torch.randn_like(H0[:, 0])
    eps_imag = torch.randn_like(H0[:, 0])

    H_sigma_real = (
        H0[:, 0]
        + sigma_tensor[:, None, None] * eps_real
    )

    H_sigma_imag = (
        H0[:, 1]
        + sigma_tensor[:, None, None] * eps_imag
    )

    H_sigma = torch.stack(
        [
            H_sigma_real,
            H_sigma_imag,
        ],
        dim=1,
    )

    # Match the exact preprocessing used during training.
    H_sigma_input = (
        H_sigma
        / sigma_tensor[:, None, None, None]
    )

    # Network predicts epsilon, not the score directly.
    epsilon_pred = net(
        H_sigma_input,
        sigma_tensor,
    )

    epsilon_target = torch.stack(
        [
            -eps_real,
            -eps_imag,
        ],
        dim=1,
    )

    # ------------------------------------------------------------
    # Epsilon diagnostic.
    # ------------------------------------------------------------

    epsilon_error = (
        epsilon_pred - epsilon_target
    )

    epsilon_mse = epsilon_error.pow(2).mean()

    epsilon_relative = (
        torch.linalg.vector_norm(epsilon_error)
        / (
            torch.linalg.vector_norm(epsilon_target)
            + 1e-12
        )
    )

    # ------------------------------------------------------------
    # Convert epsilon prediction to score.
    #
    # epsilon_theta ~= -epsilon
    #
    # score_theta = epsilon_theta / sigma
    # ------------------------------------------------------------

    predicted_score = (
        epsilon_pred
        / sigma_tensor[:, None, None, None]
    )

    # ------------------------------------------------------------
    # Analytical marginal score.
    #
    # Var(H_sigma) = Var(H0) + sigma^2
    #
    # score* = -H_sigma / (Var(H0) + sigma^2)
    # ------------------------------------------------------------

    total_variance = (
        channel_variance
        + sigma * sigma
    )

    true_score = (
        -H_sigma
        / total_variance
    )

    # ------------------------------------------------------------
    # Analytical score error.
    # ------------------------------------------------------------

    score_error = (
        predicted_score - true_score
    )

    score_mse = score_error.pow(2).mean()

    score_relative = (
        torch.linalg.vector_norm(score_error)
        / (
            torch.linalg.vector_norm(true_score)
            + 1e-12
        )
    )

    # ------------------------------------------------------------
    # Cosine similarity.
    # ------------------------------------------------------------

    pred_flat = predicted_score.reshape(
        batch_size,
        -1,
    )

    true_flat = true_score.reshape(
        batch_size,
        -1,
    )

    cosine = torch.nn.functional.cosine_similarity(
        pred_flat,
        true_flat,
        dim=1,
    ).mean()

    # ------------------------------------------------------------
    # Score magnitude.
    # ------------------------------------------------------------

    predicted_score_rms = (
        predicted_score.pow(2).mean().sqrt()
    )

    true_score_rms = (
        true_score.pow(2).mean().sqrt()
    )

    return {
        "epsilon_mse": epsilon_mse.item(),
        "epsilon_relative": epsilon_relative.item(),
        "score_mse": score_mse.item(),
        "score_relative": score_relative.item(),
        "score_cosine": cosine.item(),
        "predicted_score_rms": predicted_score_rms.item(),
        "true_score_rms": true_score_rms.item(),
    }


def evaluate(
    cfg: dict,
    checkpoint_path: str,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------
    # Channel configuration
    # ------------------------------------------------------------

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

    num_sigmas = cfg.get(
        "channel_J",
        cfg.get("J", 50),
    )

    eval_samples = cfg.get(
        "channel_eval_samples",
        10000,
    )

    # IMPORTANT:
    # Evaluation is performed in small GPU batches to avoid OOM.
    eval_batch_size = cfg.get(
        "channel_eval_batch_size",
        128,
    )

    print(
        f"Channel dimensions: "
        f"Nr={Nr}, Nt={Nt}, K={K}, Nu={Nu}"
    )

    print(
        f"Evaluation sigma range: "
        f"[{sigma_min}, {sigma_max}]"
    )

    print(
        f"Evaluation sigmas: {num_sigmas}"
    )

    print(
        f"Evaluation samples: {eval_samples}"
    )

    print(
        f"Evaluation batch size: {eval_batch_size}"
    )

    # ------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------

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

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # ------------------------------------------------------------
    # Load checkpoint.
    # ------------------------------------------------------------

    load_checkpoint(
        net,
        checkpoint_path,
        device,
    )

    net.eval()

    print("Checkpoint loaded successfully.")

    # ------------------------------------------------------------
    # Generate clean channels in manageable batches.
    #
    # We keep the complete H0 dataset on CPU and only move one
    # evaluation batch to GPU at a time.
    # ------------------------------------------------------------

    H0_complex = generate_rayleigh_channel(
        eval_samples,
        Nu,
        Nr,
        Nt,
        K,
        torch.device("cpu"),
    )
    
    H0_complex = H0_complex[:, 0]
    
    H0_full = torch.stack(
        [
            H0_complex.real,
            H0_complex.imag,
        ],
        dim=1,
    )

    print(
        f"H0 shape: {tuple(H0_full.shape)}"
    )

    # ------------------------------------------------------------
    # Determine actual channel normalization from the full
    # evaluation dataset on CPU.
    # ------------------------------------------------------------

    channel_variance = estimate_channel_variance(
        H0_full
    )

    print(
        f"Estimated real/imag variance: "
        f"{channel_variance:.8f}"
    )

    print(
        f"Estimated channel RMS: "
        f"{math.sqrt(channel_variance):.8f}"
    )

    # ------------------------------------------------------------
    # Sigma schedule.
    # ------------------------------------------------------------

    sigmas = build_sigma_schedule(
        sigma_min,
        sigma_max,
        num_sigmas,
        device,
    )

    # ------------------------------------------------------------
    # Evaluate every sigma.
    # ------------------------------------------------------------

    results = []

    print()
    print(
        "sigma        "
        "eps_MSE       "
        "score_MSE     "
        "score_rel     "
        "cosine"
    )
    print("-" * 75)

    for sigma_tensor in sigmas:
        sigma = sigma_tensor.item()

        # Accumulators for this sigma.
        total_epsilon_mse = 0.0
        total_score_mse = 0.0
        total_epsilon_relative = 0.0
        total_score_relative = 0.0
        total_cosine = 0.0
        total_predicted_score_rms = 0.0
        total_true_score_rms = 0.0

        num_batches = 0

        # --------------------------------------------------------
        # Process clean channels in GPU batches.
        # --------------------------------------------------------

        for start in range(
            0,
            eval_samples,
            eval_batch_size,
        ):
            end = min(
                start + eval_batch_size,
                eval_samples,
            )

            H0_batch = H0_full[
                start:end
            ].to(device)

            batch_result = evaluate_sigma(
                net,
                H0_batch,
                sigma,
                channel_variance,
            )

            total_epsilon_mse += (
                batch_result["epsilon_mse"]
            )

            total_epsilon_relative += (
                batch_result["epsilon_relative"]
            )

            total_score_mse += (
                batch_result["score_mse"]
            )

            total_score_relative += (
                batch_result["score_relative"]
            )

            total_cosine += (
                batch_result["score_cosine"]
            )

            total_predicted_score_rms += (
                batch_result["predicted_score_rms"]
            )

            total_true_score_rms += (
                batch_result["true_score_rms"]
            )

            num_batches += 1

            # Release GPU tensors before processing the next batch.
            del H0_batch

        # --------------------------------------------------------
        # Average batch metrics.
        # --------------------------------------------------------

        result = {
            "sigma": sigma,
            "epsilon_mse": (
                total_epsilon_mse
                / num_batches
            ),
            "epsilon_relative": (
                total_epsilon_relative
                / num_batches
            ),
            "score_mse": (
                total_score_mse
                / num_batches
            ),
            "score_relative": (
                total_score_relative
                / num_batches
            ),
            "score_cosine": (
                total_cosine
                / num_batches
            ),
            "predicted_score_rms": (
                total_predicted_score_rms
                / num_batches
            ),
            "true_score_rms": (
                total_true_score_rms
                / num_batches
            ),
        }

        results.append(result)

        print(
            f"{result['sigma']:8.4f}    "
            f"{result['epsilon_mse']:10.6f}    "
            f"{result['score_mse']:10.6f}    "
            f"{result['score_relative']:10.6f}    "
            f"{result['score_cosine']:8.5f}"
        )

    # ------------------------------------------------------------
    # Aggregate metrics across sigmas.
    # ------------------------------------------------------------

    mean_epsilon_mse = sum(
        r["epsilon_mse"]
        for r in results
    ) / len(results)

    mean_epsilon_relative = sum(
        r["epsilon_relative"]
        for r in results
    ) / len(results)

    mean_score_mse = sum(
        r["score_mse"]
        for r in results
    ) / len(results)

    mean_score_relative = sum(
        r["score_relative"]
        for r in results
    ) / len(results)

    mean_score_cosine = sum(
        r["score_cosine"]
        for r in results
    ) / len(results)

    # ------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(
        f"Mean epsilon MSE: "
        f"{mean_epsilon_mse:.8f}"
    )

    print(
        f"Mean epsilon relative error: "
        f"{mean_epsilon_relative:.8f}"
    )

    print(
        f"Mean analytical score MSE: "
        f"{mean_score_mse:.8f}"
    )

    print(
        f"Mean analytical score relative error: "
        f"{mean_score_relative:.8f}"
    )

    print(
        f"Mean score cosine similarity: "
        f"{mean_score_cosine:.8f}"
    )

    print(
        f"Channel component variance: "
        f"{channel_variance:.8f}"
    )

    print("=" * 75)

    # ------------------------------------------------------------
    # Save results.
    # ------------------------------------------------------------

    output_path = cfg.get(
        "channel_eval_output",
        "channel_score_evaluation.json",
    )

    output = {
        "checkpoint": checkpoint_path,
        "device": str(device),
        "channel": {
            "Nr": Nr,
            "Nt": Nt,
            "K": K,
            "Nu": Nu,
        },
        "evaluation": {
            "samples": eval_samples,
            "batch_size": eval_batch_size,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "num_sigmas": num_sigmas,
            "channel_component_variance": channel_variance,
        },
        "summary": {
            "mean_epsilon_mse": mean_epsilon_mse,
            "mean_epsilon_relative": mean_epsilon_relative,
            "mean_score_mse": mean_score_mse,
            "mean_score_relative": mean_score_relative,
            "mean_score_cosine": mean_score_cosine,
        },
        "per_sigma": results,
    }

    with open(output_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"Results written to: {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        cfg,
        args.checkpoint,
    )


if __name__ == "__main__":
    main()