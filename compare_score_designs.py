"""
Train and compare the three Gaussian channel-score implementations.

Models:
    1. Epsilon prediction:
           network -> -epsilon

    2. Direct score:
           network -> -epsilon / sigma

    3. Gaussian stochastic / VE-SDE:
           network -> score

All models are evaluated on EXACTLY the same:
    H0
    epsilon
    sigma
    H_sigma

Metrics computed at every sigma:
    1. Epsilon MSE
    2. Epsilon NMSE
    3. Score MSE
    4. Score NMSE
    5. Score cosine similarity
    6. Score norm ratio:
           ||score_pred|| / ||score_target||
    7. Relative score error:
           ||score_pred - score_target|| / ||score_target||

Cross-sigma consistency:
    For each model and metric we also compute:
        mean over sigma
        standard deviation over sigma
        coefficient of variation over sigma
        min over sigma
        max over sigma

Outputs:
    score_networks/comparison/comparison_results.csv
    score_networks/comparison/comparison_summary.csv

    score_networks/comparison/comparison_epsilon_nmse.png
    score_networks/comparison/comparison_score_nmse.png
    score_networks/comparison/comparison_score_cosine.png
    score_networks/comparison/comparison_score_norm_ratio.png
    score_networks/comparison/comparison_score_relative_error.png

Usage:
    python compare_score_designs.py \
        --config configs/runpod_minimal.yaml
"""

import argparse
import csv
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from score_networks.ncsnpp import ChannelScoreNet
from channels.rayleigh import generate_rayleigh_channel


# ================================================================
# Training
# ================================================================

def run_training(module, config):

    print()
    print("=" * 80)
    print(f"TRAINING: {module}")
    print("=" * 80)
    print()

    subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            config,
        ],
        check=True,
    )


# ================================================================
# Load model
# ================================================================

def load_model(checkpoint, cfg, device):

    net = ChannelScoreNet(
        Nr=cfg["Nr"],
        Nt=cfg["Nt"],
        K=cfg["K"],

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

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    net.load_state_dict(state)

    net.eval()

    return net


# ================================================================
# Metric helpers
# ================================================================

def global_nmse(pred, target, eps=1e-12):
    """
    NMSE = sum ||pred - target||^2 / sum ||target||^2

    This is preferable to averaging per-sample NMSE because
    individual samples with very small target energy can otherwise
    produce extremely large ratios.
    """

    error_energy = (
        (pred - target)
        .pow(2)
        .sum()
    )

    target_energy = (
        target
        .pow(2)
        .sum()
    )

    return (
        error_energy
        / (target_energy + eps)
    ).item()


def mean_norm_ratio(pred, target, eps=1e-12):
    """
    Mean per-sample norm ratio:

        ||pred|| / ||target||

    Ideal value = 1.
    """

    pred_flat = pred.reshape(
        pred.shape[0],
        -1,
    )

    target_flat = target.reshape(
        target.shape[0],
        -1,
    )

    pred_norm = torch.linalg.vector_norm(
        pred_flat,
        dim=1,
    )

    target_norm = torch.linalg.vector_norm(
        target_flat,
        dim=1,
    )

    ratio = (
        pred_norm
        / (target_norm + eps)
    )

    return ratio.mean().item()


def mean_relative_error(pred, target, eps=1e-12):
    """
    Mean per-sample relative L2 error:

        ||pred - target|| / ||target||
    """

    pred_flat = pred.reshape(
        pred.shape[0],
        -1,
    )

    target_flat = target.reshape(
        target.shape[0],
        -1,
    )

    error_norm = torch.linalg.vector_norm(
        pred_flat - target_flat,
        dim=1,
    )

    target_norm = torch.linalg.vector_norm(
        target_flat,
        dim=1,
    )

    relative_error = (
        error_norm
        / (target_norm + eps)
    )

    return relative_error.mean().item()


# ================================================================
# Evaluation
# ================================================================

@torch.no_grad()
def evaluate_models(
    models,
    cfg,
    device,
    samples_per_sigma=4096,
    num_sigmas=50,
):

    Nr = cfg["Nr"]
    Nt = cfg["Nt"]
    K = cfg["K"]
    Nu = cfg.get("Nu", 1)

    sigma_min = cfg.get(
        "sigma_H_1",
        0.01,
    )

    sigma_max = cfg.get(
        "sigma_H_J",
        10.0,
    )

    # ------------------------------------------------------------
    # Fixed logarithmic sigma grid
    # ------------------------------------------------------------

    sigmas = torch.logspace(
        start=torch.log10(
            torch.tensor(
                sigma_min,
                device=device,
            )
        ),
        end=torch.log10(
            torch.tensor(
                sigma_max,
                device=device,
            )
        ),
        steps=num_sigmas,
        device=device,
    )

    metric_names = [
        "epsilon_mse",
        "epsilon_nmse",

        "score_mse",
        "score_nmse",

        "score_cosine",

        "score_norm_ratio",

        "score_relative_error",
    ]

    results = {
        name: {
            metric: []
            for metric in metric_names
        }
        for name in models
    }

    # ============================================================
    # Evaluate each sigma
    # ============================================================

    for sigma_value in sigmas:

        sigma_float = sigma_value.item()

        print(
            f"Evaluating sigma = "
            f"{sigma_float:.6f}"
        )

        # --------------------------------------------------------
        # ONE shared evaluation batch.
        #
        # Every model sees exactly the same:
        #     H0
        #     epsilon
        #     sigma
        #     H_sigma
        # --------------------------------------------------------

        H0 = generate_rayleigh_channel(
            samples_per_sigma,
            Nu,
            Nr,
            Nt,
            K,
            device,
        )

        H0 = H0[:, 0]

        eps_real = torch.randn_like(
            H0.real
        )

        eps_imag = torch.randn_like(
            H0.imag
        )

        sigma = torch.full(
            (samples_per_sigma,),
            sigma_float,
            device=device,
        )

        sigma_3d = (
            sigma[:, None, None]
        )

        sigma_4d = (
            sigma[:, None, None, None]
        )

        # --------------------------------------------------------
        # H_sigma = H0 + sigma * epsilon
        # --------------------------------------------------------

        H_sigma_real = (
            H0.real
            + sigma_3d * eps_real
        )

        H_sigma_imag = (
            H0.imag
            + sigma_3d * eps_imag
        )

        H_sigma = torch.stack(
            [
                H_sigma_real,
                H_sigma_imag,
            ],
            dim=1,
        )

        # --------------------------------------------------------
        # Same input normalization used during training
        # --------------------------------------------------------

        H_input = (
            H_sigma
            / sigma_4d
        )

        # --------------------------------------------------------
        # Ground-truth epsilon representation
        #
        # epsilon_target = -epsilon
        # --------------------------------------------------------

        epsilon_target = torch.stack(
            [
                -eps_real,
                -eps_imag,
            ],
            dim=1,
        )

        # --------------------------------------------------------
        # Ground-truth Gaussian conditional score
        #
        # score = -epsilon / sigma
        # --------------------------------------------------------

        score_target = (
            epsilon_target
            / sigma_4d
        )

        # ========================================================
        # Evaluate every model
        # ========================================================

        for name, info in models.items():

            net = info["net"]

            output = net(
                H_input,
                sigma,
            )

            # ====================================================
            # Convert all outputs to both representations
            # ====================================================

            if info["type"] == "epsilon":

                # Raw model output:
                #     -epsilon

                epsilon_pred = output

                # Convert epsilon -> score
                #
                #     score = epsilon / sigma

                score_pred = (
                    epsilon_pred
                    / sigma_4d
                )

            elif info["type"] == "score":

                # Raw model output:
                #     score

                score_pred = output

                # Convert score -> epsilon
                #
                #     epsilon = sigma * score

                epsilon_pred = (
                    score_pred
                    * sigma_4d
                )

            else:

                raise ValueError(
                    f"Unknown model type: "
                    f"{info['type']}"
                )

            # ====================================================
            # 1. Epsilon MSE
            # ====================================================

            epsilon_mse = (
                (
                    epsilon_pred
                    - epsilon_target
                )
                .pow(2)
                .mean()
                .item()
            )

            # ====================================================
            # 2. Epsilon NMSE
            #
            # sum ||epsilon_pred - epsilon||^2
            # ---------------------------------
            # sum ||epsilon||^2
            # ====================================================

            epsilon_nmse = global_nmse(
                epsilon_pred,
                epsilon_target,
            )

            # ====================================================
            # 3. Score MSE
            # ====================================================

            score_mse = (
                (
                    score_pred
                    - score_target
                )
                .pow(2)
                .mean()
                .item()
            )

            # ====================================================
            # 4. Score NMSE
            #
            # sum ||score_pred - score||^2
            # ------------------------------
            # sum ||score||^2
            # ====================================================

            score_nmse = global_nmse(
                score_pred,
                score_target,
            )

            # ====================================================
            # 5. Score cosine similarity
            #
            # Measures direction.
            #
            # Ideal = 1
            # ====================================================

            pred_flat = (
                score_pred
                .reshape(
                    samples_per_sigma,
                    -1,
                )
            )

            target_flat = (
                score_target
                .reshape(
                    samples_per_sigma,
                    -1,
                )
            )

            score_cosine = (
                F.cosine_similarity(
                    pred_flat,
                    target_flat,
                    dim=1,
                )
                .mean()
                .item()
            )

            # ====================================================
            # 6. Score norm ratio
            #
            # ||score_pred||
            # --------------
            # ||score_target||
            #
            # Ideal = 1
            # ====================================================

            score_norm_ratio = mean_norm_ratio(
                score_pred,
                score_target,
            )

            # ====================================================
            # 7. Relative score error
            #
            # ||score_pred - score_target||
            # --------------------------------
            # ||score_target||
            # ====================================================

            score_relative_error = mean_relative_error(
                score_pred,
                score_target,
            )

            # ====================================================
            # Store metrics
            # ====================================================

            results[name][
                "epsilon_mse"
            ].append(
                epsilon_mse
            )

            results[name][
                "epsilon_nmse"
            ].append(
                epsilon_nmse
            )

            results[name][
                "score_mse"
            ].append(
                score_mse
            )

            results[name][
                "score_nmse"
            ].append(
                score_nmse
            )

            results[name][
                "score_cosine"
            ].append(
                score_cosine
            )

            results[name][
                "score_norm_ratio"
            ].append(
                score_norm_ratio
            )

            results[name][
                "score_relative_error"
            ].append(
                score_relative_error
            )

    return (
        sigmas.detach().cpu().tolist(),
        results,
    )


# ================================================================
# Save detailed CSV
# ================================================================

def save_csv(
    sigmas,
    results,
    output_path,
):

    names = list(
        results.keys()
    )

    metrics = list(
        next(iter(results.values())).keys()
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        header = [
            "sigma",
        ]

        for name in names:

            for metric in metrics:

                header.append(
                    f"{name}_{metric}"
                )

        writer.writerow(
            header
        )

        for i, sigma in enumerate(sigmas):

            row = [
                sigma
            ]

            for name in names:

                for metric in metrics:

                    row.append(
                        results[name][metric][i]
                    )

            writer.writerow(
                row
            )


# ================================================================
# Cross-sigma consistency summary
# ================================================================

def save_summary_csv(
    results,
    output_path,
):

    """
    Summarize how stable each metric is across sigma.

    std:
        absolute variation across sigma.

    coefficient_of_variation:
        std / |mean|

    Smaller CV generally means the metric is more consistent
    across the noise schedule.

    Note:
        For cosine similarity and norm ratio, interpret CV with
        care because their ideal values are non-zero reference
        values rather than zero-error quantities.
    """

    with open(
        output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "metric",
            "mean",
            "std_across_sigma",
            "coefficient_of_variation",
            "min",
            "max",
        ])

        for name, model_results in results.items():

            for metric, values in model_results.items():

                tensor = torch.tensor(
                    values,
                    dtype=torch.float64,
                )

                mean = tensor.mean().item()

                std = tensor.std(
                    unbiased=False
                ).item()

                minimum = tensor.min().item()

                maximum = tensor.max().item()

                cv = (
                    std
                    / (abs(mean) + 1e-12)
                )

                writer.writerow([
                    name,
                    metric,
                    mean,
                    std,
                    cv,
                    minimum,
                    maximum,
                ])


# ================================================================
# Print summary
# ================================================================

def print_summary(results):

    print()
    print("=" * 80)
    print("CROSS-SIGMA SUMMARY")
    print("=" * 80)

    important_metrics = [
        "epsilon_nmse",
        "score_nmse",
        "score_cosine",
        "score_norm_ratio",
        "score_relative_error",
    ]

    for name, model_results in results.items():

        print()
        print(f"MODEL: {name}")
        print("-" * 80)

        for metric in important_metrics:

            values = torch.tensor(
                model_results[metric],
                dtype=torch.float64,
            )

            mean = values.mean().item()

            std = values.std(
                unbiased=False
            ).item()

            minimum = values.min().item()

            maximum = values.max().item()

            cv = (
                std
                / (abs(mean) + 1e-12)
            )

            print(
                f"{metric:28s} "
                f"mean={mean:.6f}  "
                f"std={std:.6f}  "
                f"CV={cv:.6f}  "
                f"min={minimum:.6f}  "
                f"max={maximum:.6f}"
            )


# ================================================================
# Plot
# ================================================================

def plot_metric(
    sigmas,
    results,
    metric,
    ylabel,
    filename,
    log_y=False,
    reference_line=None,
):

    plt.figure(
        figsize=(10, 6)
    )

    for name in results:

        plt.plot(
            sigmas,
            results[name][metric],
            marker="o",
            markersize=3,
            label=name,
        )

    plt.xscale(
        "log"
    )

    if log_y:

        plt.yscale(
            "log"
        )

    # Optional ideal/reference line
    if reference_line is not None:

        plt.axhline(
            reference_line,
            linestyle="--",
            linewidth=1.5,
            label=f"Ideal = {reference_line}",
        )

    plt.xlabel(
        "Noise scale sigma"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        f"{ylabel} vs noise scale"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
    )

    plt.close()


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip training and use "
            "existing checkpoints."
        ),
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--eval-sigmas",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    # ============================================================
    # Load config
    # ============================================================

    with open(
        args.config
    ) as f:

        cfg = yaml.safe_load(f)

    # ============================================================
    # Train all three
    # ============================================================

    if not args.skip_training:

        # --------------------------------------------------------
        # 1. Epsilon prediction
        # --------------------------------------------------------

        run_training(
            "score_networks.train_channel_score",
            args.config,
        )

        # --------------------------------------------------------
        # 2. Direct score
        # --------------------------------------------------------

        run_training(
            "score_networks.train_channel_score_over_sigma",
            args.config,
        )

        # --------------------------------------------------------
        # 3. Gaussian stochastic / VE-SDE
        # --------------------------------------------------------

        run_training(
            "score_networks.train_channel_score_stochastic",
            args.config,
        )

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_dir = cfg.get(
        "score_ckpt_dir",
        "score_networks/checkpoints",
    )

    # ============================================================
    # Checkpoints
    # ============================================================

    epsilon_checkpoint = os.path.join(
        ckpt_dir,
        "channel_score_best.pt",
    )

    over_sigma_checkpoint = os.path.join(
        ckpt_dir,
        "channel_score_direct_best.pt",
    )

    stochastic_checkpoint = os.path.join(
        ckpt_dir,
        "channel_score_ve_sde_best.pt",
    )

    print()
    print("=" * 80)
    print("LOADING CHECKPOINTS")
    print("=" * 80)

    print(
        f"Epsilon:     {epsilon_checkpoint}"
    )

    print(
        f"Over sigma:  {over_sigma_checkpoint}"
    )

    print(
        f"Stochastic:  {stochastic_checkpoint}"
    )

    # ============================================================
    # Verify checkpoints
    # ============================================================

    for path in [
        epsilon_checkpoint,
        over_sigma_checkpoint,
        stochastic_checkpoint,
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Checkpoint does not exist: {path}\n"
                f"Check the checkpoint filename saved by "
                f"the corresponding training script."
            )

    # ============================================================
    # Load models
    # ============================================================

    models = {

        "epsilon": {

            "net": load_model(
                epsilon_checkpoint,
                cfg,
                device,
            ),

            "type": "epsilon",
        },

        "epsilon_over_sigma": {

            "net": load_model(
                over_sigma_checkpoint,
                cfg,
                device,
            ),

            "type": "score",
        },

        "stochastic": {

            "net": load_model(
                stochastic_checkpoint,
                cfg,
                device,
            ),

            "type": "score",
        },
    }

    # ============================================================
    # Shared evaluation
    # ============================================================

    print()
    print("=" * 80)
    print("STARTING SHARED EVALUATION")
    print("=" * 80)
    print()

    sigmas, results = evaluate_models(
        models,
        cfg,
        device,

        samples_per_sigma=(
            args.eval_samples
        ),

        num_sigmas=(
            args.eval_sigmas
        ),
    )

    # ============================================================
    # Output directory
    # ============================================================

    output_dir = os.path.join(
        "score_networks",
        "comparison",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ============================================================
    # Detailed CSV
    # ============================================================

    csv_path = os.path.join(
        output_dir,
        "comparison_results.csv",
    )

    save_csv(
        sigmas,
        results,
        csv_path,
    )

    # ============================================================
    # Cross-sigma summary CSV
    # ============================================================

    summary_path = os.path.join(
        output_dir,
        "comparison_summary.csv",
    )

    save_summary_csv(
        results,
        summary_path,
    )

    print_summary(
        results
    )

    # ============================================================
    # Plot 1: epsilon NMSE
    # ============================================================

    plot_metric(
        sigmas,
        results,

        metric="epsilon_nmse",

        ylabel="Epsilon NMSE",

        filename=os.path.join(
            output_dir,
            "comparison_epsilon_nmse.png",
        ),

        log_y=False,
    )

    # ============================================================
    # Plot 2: score NMSE
    # ============================================================

    plot_metric(
        sigmas,
        results,

        metric="score_nmse",

        ylabel="Score NMSE",

        filename=os.path.join(
            output_dir,
            "comparison_score_nmse.png",
        ),

        log_y=False,
    )

    # ============================================================
    # Plot 3: score cosine similarity
    # ============================================================

    plot_metric(
        sigmas,
        results,

        metric="score_cosine",

        ylabel="Score cosine similarity",

        filename=os.path.join(
            output_dir,
            "comparison_score_cosine.png",
        ),

        log_y=False,

        reference_line=1.0,
    )

    # ============================================================
    # Plot 4: score norm ratio
    #
    # Ideal = 1
    # ============================================================

    plot_metric(
        sigmas,
        results,

        metric="score_norm_ratio",

        ylabel="Score norm ratio",

        filename=os.path.join(
            output_dir,
            "comparison_score_norm_ratio.png",
        ),

        log_y=False,

        reference_line=1.0,
    )

    # ============================================================
    # Plot 5: relative score error
    #
    # Ideal = 0
    # ============================================================

    plot_metric(
        sigmas,
        results,

        metric="score_relative_error",

        ylabel="Relative score error",

        filename=os.path.join(
            output_dir,
            "comparison_score_relative_error.png",
        ),

        log_y=False,

        reference_line=0.0,
    )

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)

    print(
        f"Detailed results:\n"
        f"    {csv_path}"
    )

    print()

    print(
        f"Cross-sigma summary:\n"
        f"    {summary_path}"
    )

    print()

    print(
        "Graphs:\n"
        f"    {output_dir}/comparison_epsilon_nmse.png\n"
        f"    {output_dir}/comparison_score_nmse.png\n"
        f"    {output_dir}/comparison_score_cosine.png\n"
        f"    {output_dir}/comparison_score_norm_ratio.png\n"
        f"    {output_dir}/comparison_score_relative_error.png"
    )


if __name__ == "__main__":
    main()