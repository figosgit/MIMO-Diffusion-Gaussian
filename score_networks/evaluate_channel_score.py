"""
Evaluate independent or temporal channel score networks.

Independent model:
    q(H)

Temporal model:
    q(H_t | H_{t-1})

Usage:

Independent:
    python -m score_networks.evaluate_channel_score \
        --config configs/runpod_minimal.yaml \
        --checkpoint score_networks/checkpoints/channel_score_best.pt

Temporal:
    python -m score_networks.evaluate_channel_score \
        --config configs/runpod_minimal.yaml \
        --checkpoint score_networks/checkpoints/channel_score_temporal_best.pt \
        --temporal
"""

import argparse
import json
import math

import torch
import yaml

from .ncsnpp import (
    ChannelScoreNet,
    ChannelTemporalScoreNet,
)

from channels.rayleigh import (
    generate_rayleigh_channel,
    generate_rayleigh_channel_sequence,
)


def build_sigma_schedule(
    sigma_min: float,
    sigma_max: float,
    num_sigmas: int,
    device: torch.device,
) -> torch.Tensor:
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
    real_var = H0[:, 0].var(unbiased=False)
    imag_var = H0[:, 1].var(unbiased=False)

    return (
        0.5 * (real_var + imag_var)
    ).item()


@torch.no_grad()
def evaluate_sigma_independent(
    net: torch.nn.Module,
    H0: torch.Tensor,
    sigma: float,
    channel_variance: float,
):
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

    H_sigma_input = (
        H_sigma
        / sigma_tensor[:, None, None, None]
    )

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

    epsilon_error = (
        epsilon_pred - epsilon_target
    )

    epsilon_mse = (
        epsilon_error.pow(2).mean()
    )

    epsilon_relative = (
        torch.linalg.vector_norm(epsilon_error)
        / (
            torch.linalg.vector_norm(epsilon_target)
            + 1e-12
        )
    )

    predicted_score = (
        epsilon_pred
        / sigma_tensor[:, None, None, None]
    )

    total_variance = (
        channel_variance
        + sigma * sigma
    )

    true_score = (
        -H_sigma
        / total_variance
    )

    score_error = (
        predicted_score - true_score
    )

    score_mse = (
        score_error.pow(2).mean()
    )

    score_relative = (
        torch.linalg.vector_norm(score_error)
        / (
            torch.linalg.vector_norm(true_score)
            + 1e-12
        )
    )

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


@torch.no_grad()
def evaluate_sigma_temporal(
    net: torch.nn.Module,
    H_curr: torch.Tensor,
    H_prev: torch.Tensor,
    sigma: float,
    channel_variance: float,
    alpha: float,
):
    device = H_curr.device
    batch_size = H_curr.shape[0]

    sigma_tensor = torch.full(
        (batch_size,),
        sigma,
        device=device,
        dtype=H_curr.dtype,
    )

    # Add DSM noise ONLY to H_t.
    eps_real = torch.randn_like(H_curr[:, 0])
    eps_imag = torch.randn_like(H_curr[:, 0])

    H_sigma_real = (
        H_curr[:, 0]
        + sigma_tensor[:, None, None] * eps_real
    )

    H_sigma_imag = (
        H_curr[:, 1]
        + sigma_tensor[:, None, None] * eps_imag
    )

    H_sigma = torch.stack(
        [
            H_sigma_real,
            H_sigma_imag,
        ],
        dim=1,
    )

    # Same preprocessing as temporal training.
    H_sigma_input = (
        H_sigma
        / sigma_tensor[:, None, None, None]
    )

    # H_prev remains clean and unnormalized.
    epsilon_pred = net(
        H_sigma_input,
        H_prev,
        sigma_tensor,
    )

    epsilon_target = torch.stack(
        [
            -eps_real,
            -eps_imag,
        ],
        dim=1,
    )

    epsilon_error = (
        epsilon_pred - epsilon_target
    )

    epsilon_mse = (
        epsilon_error.pow(2).mean()
    )

    epsilon_relative = (
        torch.linalg.vector_norm(epsilon_error)
        / (
            torch.linalg.vector_norm(epsilon_target)
            + 1e-12
        )
    )

    # epsilon_theta / sigma = score.
    predicted_score = (
        epsilon_pred
        / sigma_tensor[:, None, None, None]
    )

    # ------------------------------------------------------------
    # Correct conditional analytical score.
    #
    # H_t | H_{t-1}
    #
    # mean = alpha * H_{t-1}
    #
    # variance = (1-alpha^2) * v
    #
    # After DSM noise:
    #
    # variance = (1-alpha^2)*v + sigma^2
    # ------------------------------------------------------------

    conditional_mean = (
        alpha * H_prev
    )

    conditional_variance = (
        (1.0 - alpha * alpha)
        * channel_variance
        + sigma * sigma
    )

    true_score = (
        -(H_sigma - conditional_mean)
        / conditional_variance
    )

    score_error = (
        predicted_score - true_score
    )

    score_mse = (
        score_error.pow(2).mean()
    )

    score_relative = (
        torch.linalg.vector_norm(score_error)
        / (
            torch.linalg.vector_norm(true_score)
            + 1e-12
        )
    )

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
    temporal_override: bool = False,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------
    # Configuration
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

    eval_batch_size = cfg.get(
        "channel_eval_batch_size",
        128,
    )

    temporal = (
        temporal_override
        or cfg.get("channel_temporal", False)
    )

    alpha = cfg.get(
        "channel_ar1_alpha",
        0.7,
    )

    sequence_length = cfg.get(
        "channel_sequence_length",
        20,
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

    if temporal:
        print("Channel evaluation mode: TEMPORAL")
        print(f"AR(1) alpha: {alpha}")
        print(f"Sequence length: {sequence_length}")
    else:
        print("Channel evaluation mode: INDEPENDENT")

    # ------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------

    if temporal:
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

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    load_checkpoint(
        net,
        checkpoint_path,
        device,
    )

    net.eval()

    print("Checkpoint loaded successfully.")

    # ------------------------------------------------------------
    # Generate evaluation data
    # ------------------------------------------------------------

    if temporal:

        H_seq_complex = (
            generate_rayleigh_channel_sequence(
                eval_samples,
                Nu,
                Nr,
                Nt,
                K,
                sequence_length,
                alpha=alpha,
                device=torch.device("cpu"),
            )
        )

        # Nu = 1, same as training.
        H_seq_complex = H_seq_complex[:, :, 0]

        H_prev_complex = H_seq_complex[:, :-1]
        H_curr_complex = H_seq_complex[:, 1:]

        H_prev_complex = H_prev_complex.reshape(
            -1,
            Nr * K,
            Nt * K,
        )

        H_curr_complex = H_curr_complex.reshape(
            -1,
            Nr * K,
            Nt * K,
        )

        H_prev_full = torch.stack(
            [
                H_prev_complex.real,
                H_prev_complex.imag,
            ],
            dim=1,
        )

        H_curr_full = torch.stack(
            [
                H_curr_complex.real,
                H_curr_complex.imag,
            ],
            dim=1,
        )

        print(
            f"Temporal evaluation pairs: "
            f"{H_curr_full.shape[0]}"
        )

        print(
            f"H_prev shape: "
            f"{tuple(H_prev_full.shape)}"
        )

        print(
            f"H_curr shape: "
            f"{tuple(H_curr_full.shape)}"
        )

        channel_variance = (
            estimate_channel_variance(
                H_curr_full
            )
        )

        num_samples = H_curr_full.shape[0]

    else:

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
            f"H0 shape: "
            f"{tuple(H0_full.shape)}"
        )

        channel_variance = (
            estimate_channel_variance(
                H0_full
            )
        )

        num_samples = H0_full.shape[0]

    print(
        f"Estimated real/imag variance: "
        f"{channel_variance:.8f}"
    )

    print(
        f"Estimated channel RMS: "
        f"{math.sqrt(channel_variance):.8f}"
    )

    # ------------------------------------------------------------
    # Sigma schedule
    # ------------------------------------------------------------

    sigmas = build_sigma_schedule(
        sigma_min,
        sigma_max,
        num_sigmas,
        device,
    )

    # ------------------------------------------------------------
    # Evaluate
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

        total_epsilon_mse = 0.0
        total_epsilon_relative = 0.0
        total_score_mse = 0.0
        total_score_relative = 0.0
        total_cosine = 0.0
        total_predicted_score_rms = 0.0
        total_true_score_rms = 0.0

        num_batches = 0

        for start in range(
            0,
            num_samples,
            eval_batch_size,
        ):

            end = min(
                start + eval_batch_size,
                num_samples,
            )

            if temporal:

                H_prev_batch = (
                    H_prev_full[start:end]
                    .to(device)
                )

                H_curr_batch = (
                    H_curr_full[start:end]
                    .to(device)
                )

                batch_result = (
                    evaluate_sigma_temporal(
                        net,
                        H_curr_batch,
                        H_prev_batch,
                        sigma,
                        channel_variance,
                        alpha,
                    )
                )

                del H_prev_batch
                del H_curr_batch

            else:

                H0_batch = (
                    H0_full[start:end]
                    .to(device)
                )

                batch_result = (
                    evaluate_sigma_independent(
                        net,
                        H0_batch,
                        sigma,
                        channel_variance,
                    )
                )

                del H0_batch

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
    # Aggregate
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
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print(
        f"Mode: "
        f"{'TEMPORAL' if temporal else 'INDEPENDENT'}"
    )

    if temporal:
        print(
            f"AR(1) alpha: {alpha:.6f}"
        )

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
    # Save results
    # ------------------------------------------------------------

    output_path = cfg.get(
        "channel_eval_output",
        (
            "channel_score_temporal_evaluation.json"
            if temporal
            else "channel_score_evaluation.json"
        ),
    )

    output = {
        "checkpoint": checkpoint_path,
        "device": str(device),

        "mode": (
            "temporal"
            if temporal
            else "independent"
        ),

        "channel": {
            "Nr": Nr,
            "Nt": Nt,
            "K": K,
            "Nu": Nu,
        },

        "temporal": {
            "alpha": alpha,
            "sequence_length": sequence_length,
        } if temporal else None,

        "evaluation": {
            "samples": eval_samples,
            "effective_samples": num_samples,
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

    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Evaluate ChannelTemporalScoreNet.",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(
        cfg,
        args.checkpoint,
        temporal_override=args.temporal,
    )


if __name__ == "__main__":
    main()