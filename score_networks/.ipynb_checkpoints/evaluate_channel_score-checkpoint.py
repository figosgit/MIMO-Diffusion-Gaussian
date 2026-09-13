"""
Evaluate the channel score network.

The channel network uses epsilon parameterization:

```
H_sigma = H_0 + sigma * epsilon

epsilon_theta(H_sigma, sigma) ~= E[-epsilon | H_sigma]
```

and therefore:

```
score_theta(H_sigma, sigma)
    = epsilon_theta(H_sigma, sigma) / sigma
```

For an i.i.d. Gaussian/Rayleigh channel, the noisy marginal remains
Gaussian, so the exact marginal score is available analytically.

If each real/imaginary component of H_0 has variance v:

```
H_sigma ~ N(0, v + sigma^2)
```

and therefore:

```
score*(H_sigma, sigma)
    = -H_sigma / (v + sigma^2)
```

This evaluator supports two modes.

1. NON-TEMPORAL

   Evaluate the network on i.i.d. Rayleigh channels.

2. TEMPORAL DATA

   Evaluate the same non-temporal network on an AR(1) Rayleigh channel:

   ```
   H_t = alpha * H_{t-1}
         + sqrt(1 - alpha^2) * W_t
   ```

   where W_t is an independent Rayleigh channel.

   The non-temporal network still receives only:

   ```
   H_sigma / sigma, sigma
   ```

   and does NOT receive H_{t-1}.

   However, the analytical score target is the conditional temporal
   score:

   ```
   score*(H_sigma | H_prev)
       = -(H_sigma - alpha * H_prev)
         / ((1 - alpha^2) * v + sigma^2)
   ```

This allows us to test whether the independent model can predict the
temporal conditional score without access to temporal context.

Metrics:

1. Epsilon MSE against sampled -epsilon.
   This is a DSM diagnostic.

2. Epsilon relative error.

3. Analytical score MSE.

4. Analytical score relative error.

5. Score cosine similarity.

6. Predicted score RMS.

7. True score RMS.

Usage:

```
# Normal i.i.d. evaluation

python -m score_networks.evaluate_channel_score \
    --config configs/runpod_minimal.yaml \
    --checkpoint score_networks/checkpoints/channel_score_best.pt

# Evaluate the old non-temporal checkpoint on temporal AR(1) data

python -m score_networks.evaluate_channel_score \
    --config configs/runpod_minimal.yaml \
    --checkpoint score_networks/checkpoints/channel_score_best.pt \
    --temporal-data

# Explicit temporal parameters

python -m score_networks.evaluate_channel_score \
    --config configs/runpod_minimal.yaml \
    --checkpoint score_networks/checkpoints/channel_score_best.pt \
    --temporal-data \
    --alpha 0.7 \
    --sequence-length 20
```

"""

import argparse
import json
import math

import torch
import yaml

from .ncsnpp import ChannelScoreNet
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

```
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
```

@torch.no_grad()
def estimate_channel_variance(
H0: torch.Tensor,
) -> float:
"""
Estimate the variance of one real-valued channel component.

```
H0 has shape:

    (B, 2, Nr*K, Nt*K)

where channel dimension 1 is [real, imag].
"""
real_var = H0[:, 0].var(unbiased=False)
imag_var = H0[:, 1].var(unbiased=False)

variance = 0.5 * (real_var + imag_var)

return variance.item()
```

@torch.no_grad()
def evaluate_sigma(
net: torch.nn.Module,
H0: torch.Tensor,
sigma: float,
channel_variance: float,
H_prev: torch.Tensor | None = None,
alpha: float = 0.7,
):
"""
Evaluate one noise level on one batch.

```
Parameters
----------
net:
    Non-temporal ChannelScoreNet.

H0:
    Clean current channel with shape:

        (B, 2, Nr*K, Nt*K)

    In temporal mode this is H_curr.

sigma:
    Noise level.

channel_variance:
    Estimated variance of one real/imaginary channel component.

H_prev:
    Previous clean channel.

    If None:
        use the i.i.d. marginal analytical score.

    If provided:
        use the temporal conditional analytical score.

alpha:
    AR(1) correlation coefficient.
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

# ------------------------------------------------------------
# Network input.
#
# This is exactly the preprocessing used during training.
#
# IMPORTANT:
# The non-temporal model receives NO H_prev.
# ------------------------------------------------------------

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
        \+ 1e-12
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
# Analytical score.
#
# NON-TEMPORAL:
#
#   score* = -H_sigma / (v + sigma^2)
#
# TEMPORAL:
#
#   score*(H_sigma | H_prev)
#       = -(H_sigma - alpha H_prev)
#         / ((1-alpha^2)v + sigma^2)
#
# The temporal network would use H_prev as input.
# This network does NOT. We only use H_prev here to construct
# the ground-truth conditional score.
# ------------------------------------------------------------

if H_prev is None:
    total_variance = (
        channel_variance
        + sigma * sigma
    )

    true_score = (
        -H_sigma
        / total_variance
    )

else:
    conditional_variance = (
        (1.0 - alpha * alpha)
        * channel_variance
        + sigma * sigma
    )

    true_score = (
        -(H_sigma - alpha * H_prev)
        / conditional_variance
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
        \+ 1e-12
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
```

@torch.no_grad()
def estimate_temporal_channel_variance(
eval_samples: int,
eval_batch_size: int,
Nu: int,
Nr: int,
Nt: int,
K: int,
sequence_length: int,
alpha: float,
) -> float:
"""
Estimate the channel component variance from temporal AR(1) data.

```
The AR(1) process is stationary because:

    H_0 ~ CN(0, I)

    H_t = alpha H_{t-1}
          + sqrt(1-alpha^2) W_t

so the marginal distribution of H_t remains the same Rayleigh
distribution as the i.i.d. channel.

Only a small number of sequences are needed for the variance
estimate. The full evaluation dataset is not stored in memory.
"""
device = torch.device("cpu")

estimate_samples = min(
    eval_samples,
    max(eval_batch_size, 1000),
)

H_seq = generate_rayleigh_channel_sequence(
    estimate_samples,
    Nu,
    Nr,
    Nt,
    K,
    sequence_length,
    alpha,
    device,
)

# Select first user.
#
# Shape:
#   (B, T, Nr*K, Nt*K)
H_seq = H_seq[:, :, 0]

# Use all time steps for the variance estimate.
H_real = H_seq.real
H_imag = H_seq.imag

real_var = H_real.var(unbiased=False)
imag_var = H_imag.var(unbiased=False)

variance = (
    0.5 * (real_var + imag_var)
)

return variance.item()
```

def evaluate(
cfg: dict,
checkpoint_path: str,
temporal_data: bool = False,
alpha: float = 0.7,
sequence_length: int = 20,
):
device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

```
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

eval_batch_size = cfg.get(
    "channel_eval_batch_size",
    128,
)

# ------------------------------------------------------------
# Configuration override from YAML.
#
# CLI arguments are already resolved by main().
# ------------------------------------------------------------

if temporal_data:
    alpha = float(alpha)
    sequence_length = int(sequence_length)

    if not 0 <= alpha < 1:
        raise ValueError(
            f"alpha must satisfy 0 <= alpha < 1, got {alpha}"
        )

    if sequence_length < 2:
        raise ValueError(
            "sequence_length must be >= 2"
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
# Mode.
# ------------------------------------------------------------

if temporal_data:
    print()
    print("Mode: TEMPORAL DATA")
    print(f"AR(1) alpha: {alpha}")
    print(f"Sequence length: {sequence_length}")
    print(
        f"Temporal pairs: "
        f"{eval_samples * (sequence_length - 1)}"
    )
    print(
        "Model input: H_curr / sigma, sigma"
    )
    print(
        "H_prev: used only for conditional analytical target"
    )
else:
    print()
    print("Mode: NON-TEMPORAL")
    print(
        "Model input: H_sigma / sigma, sigma"
    )
    print(
        "Analytical target: marginal Gaussian score"
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
# Determine channel variance.
# ------------------------------------------------------------

if temporal_data:
    channel_variance = estimate_temporal_channel_variance(
        eval_samples,
        eval_batch_size,
        Nu,
        Nr,
        Nt,
        K,
        sequence_length,
        alpha,
    )
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
        f"H0 shape: {tuple(H0_full.shape)}"
    )

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

    # --------------------------------------------------------
    # Accumulators.
    #
    # We accumulate weighted sums rather than averaging batch
    # means, so the final partial batch is handled correctly.
    # --------------------------------------------------------

    total_epsilon_mse = 0.0
    total_score_mse = 0.0
    total_epsilon_relative = 0.0
    total_score_relative = 0.0
    total_cosine = 0.0
    total_predicted_score_rms = 0.0
    total_true_score_rms = 0.0

    total_examples = 0

    # ========================================================
    # NON-TEMPORAL DATA
    # ========================================================

    if not temporal_data:

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

            batch_size = H0_batch.shape[0]

            batch_result = evaluate_sigma(
                net,
                H0_batch,
                sigma,
                channel_variance,
            )

            total_epsilon_mse += (
                batch_result["epsilon_mse"]
                * batch_size
            )

            total_epsilon_relative += (
                batch_result["epsilon_relative"]
                * batch_size
            )

            total_score_mse += (
                batch_result["score_mse"]
                * batch_size
            )

            total_score_relative += (
                batch_result["score_relative"]
                * batch_size
            )

            total_cosine += (
                batch_result["score_cosine"]
                * batch_size
            )

            total_predicted_score_rms += (
                batch_result["predicted_score_rms"]
                * batch_size
            )

            total_true_score_rms += (
                batch_result["true_score_rms"]
                * batch_size
            )

            total_examples += batch_size

            del H0_batch

    # ========================================================
    # TEMPORAL DATA
    # ========================================================

    else:

        # ----------------------------------------------------
        # Each generated sequence contributes:
        #
        #   sequence_length - 1
        #
        # temporal pairs:
        #
        #   (H_0, H_1)
        #   (H_1, H_2)
        #   ...
        #   (H_{T-2}, H_{T-1})
        #
        # We flatten those pairs into a normal batch because
        # the non-temporal model only accepts the current
        # channel.
        # ----------------------------------------------------

        for start in range(
            0,
            eval_samples,
            eval_batch_size,
        ):
            end = min(
                start + eval_batch_size,
                eval_samples,
            )

            sequence_batch_size = (
                end - start
            )

            H_seq = generate_rayleigh_channel_sequence(
                sequence_batch_size,
                Nu,
                Nr,
                Nt,
                K,
                sequence_length,
                alpha,
                torch.device("cpu"),
            )

            # Select first user.
            #
            # Shape:
            #
            #   (B, T, Nr*K, Nt*K)
            #
            H_seq = H_seq[:, :, 0]

            # ------------------------------------------------
            # Build temporal pairs.
            # ------------------------------------------------

            H_prev_complex = H_seq[:, :-1]

            H_curr_complex = H_seq[:, 1:]

            num_pairs = (
                sequence_batch_size
                * (sequence_length - 1)
            )

            # Flatten sequence/time dimensions.
            H_prev_complex = H_prev_complex.reshape(
                num_pairs,
                Nr * K,
                Nt * K,
            )

            H_curr_complex = H_curr_complex.reshape(
                num_pairs,
                Nr * K,
                Nt * K,
            )

            H_prev = torch.stack(
                [
                    H_prev_complex.real,
                    H_prev_complex.imag,
                ],
                dim=1,
            ).to(device)

            H_curr = torch.stack(
                [
                    H_curr_complex.real,
                    H_curr_complex.imag,
                ],
                dim=1,
            ).to(device)

            batch_result = evaluate_sigma(
                net,
                H_curr,
                sigma,
                channel_variance,
                H_prev=H_prev,
                alpha=alpha,
            )

            total_epsilon_mse += (
                batch_result["epsilon_mse"]
                * num_pairs
            )

            total_epsilon_relative += (
                batch_result["epsilon_relative"]
                * num_pairs
            )

            total_score_mse += (
                batch_result["score_mse"]
                * num_pairs
            )

            total_score_relative += (
                batch_result["score_relative"]
                * num_pairs
            )

            total_cosine += (
                batch_result["score_cosine"]
                * num_pairs
            )

            total_predicted_score_rms += (
                batch_result["predicted_score_rms"]
                * num_pairs
            )

            total_true_score_rms += (
                batch_result["true_score_rms"]
                * num_pairs
            )

            total_examples += num_pairs

            del H_seq
            del H_prev_complex
            del H_curr_complex
            del H_prev
            del H_curr

    # --------------------------------------------------------
    # Average metrics.
    # --------------------------------------------------------

    result = {
        "sigma": sigma,
        "epsilon_mse": (
            total_epsilon_mse
            / total_examples
        ),
        "epsilon_relative": (
            total_epsilon_relative
            / total_examples
        ),
        "score_mse": (
            total_score_mse
            / total_examples
        ),
        "score_relative": (
            total_score_relative
            / total_examples
        ),
        "score_cosine": (
            total_cosine
            / total_examples
        ),
        "predicted_score_rms": (
            total_predicted_score_rms
            / total_examples
        ),
        "true_score_rms": (
            total_true_score_rms
            / total_examples
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

mean_predicted_score_rms = sum(
    r["predicted_score_rms"]
    for r in results
) / len(results)

mean_true_score_rms = sum(
    r["true_score_rms"]
    for r in results
) / len(results)

# ------------------------------------------------------------
# Print summary.
# ------------------------------------------------------------

print()
print("=" * 75)
print("SUMMARY")
print("=" * 75)

if temporal_data:
    print("Mode: TEMPORAL DATA")
    print(f"AR(1) alpha: {alpha:.6f}")
    print(f"Sequence length: {sequence_length}")
    print(
        f"Temporal pairs: "
        f"{eval_samples * (sequence_length - 1)}"
    )
    print(
        "Target: conditional score "
        "p(H_sigma | H_prev)"
    )
else:
    print("Mode: NON-TEMPORAL")
    print("Target: marginal score p(H_sigma)")

print()

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
    f"Mean predicted score RMS: "
    f"{mean_predicted_score_rms:.8f}"
)

print(
    f"Mean true score RMS: "
    f"{mean_true_score_rms:.8f}"
)

print(
    f"Channel component variance: "
    f"{channel_variance:.8f}"
)

print("=" * 75)

# ------------------------------------------------------------
# Save results.
# ------------------------------------------------------------

if temporal_data:
    default_output = (
        "channel_score_temporal_data_evaluation.json"
    )
else:
    default_output = (
        "channel_score_evaluation.json"
    )

output_path = cfg.get(
    "channel_eval_output",
    default_output,
)

output = {
    "checkpoint": checkpoint_path,
    "device": str(device),
    "mode": (
        "temporal_data"
        if temporal_data
        else "non_temporal"
    ),
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
    "temporal": {
        "enabled": temporal_data,
        "alpha": alpha if temporal_data else None,
        "sequence_length": (
            sequence_length
            if temporal_data
            else None
        ),
        "temporal_pairs": (
            eval_samples * (sequence_length - 1)
            if temporal_data
            else None
        ),
        "target": (
            "conditional_score_p(H_sigma | H_prev)"
            if temporal_data
            else "marginal_score_p(H_sigma)"
        ),
        "model_receives_H_prev": False,
    },
    "summary": {
        "mean_epsilon_mse": mean_epsilon_mse,
        "mean_epsilon_relative": mean_epsilon_relative,
        "mean_score_mse": mean_score_mse,
        "mean_score_relative": mean_score_relative,
        "mean_score_cosine": mean_score_cosine,
        "mean_predicted_score_rms": (
            mean_predicted_score_rms
        ),
        "mean_true_score_rms": (
            mean_true_score_rms
        ),
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
```

def main():
parser = argparse.ArgumentParser()

```
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
    "--temporal-data",
    action="store_true",
    help=(
        "Evaluate the non-temporal checkpoint on "
        "temporally correlated AR(1) channel data."
    ),
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.7,
    help="AR(1) temporal correlation coefficient.",
)

parser.add_argument(
    "--sequence-length",
    type=int,
    default=20,
    help="Length of each AR(1) channel sequence.",
)

args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# Allow YAML values to be used when CLI values were not
# explicitly changed from their defaults.
alpha = args.alpha
sequence_length = args.sequence_length

if temporal_config := cfg.get("channel_temporal", False):
    if not args.temporal_data:
        temporal_data = False
    else:
        temporal_data = True
else:
    temporal_data = args.temporal_data

if args.temporal_data:
    alpha = cfg.get(
        "channel_ar1_alpha",
        args.alpha,
    )

    sequence_length = cfg.get(
        "channel_sequence_length",
        args.sequence_length,
    )

evaluate(
    cfg,
    args.checkpoint,
    temporal_data=temporal_data,
    alpha=alpha,
    sequence_length=sequence_length,
)
```

if **name** == "**main**":
main()
