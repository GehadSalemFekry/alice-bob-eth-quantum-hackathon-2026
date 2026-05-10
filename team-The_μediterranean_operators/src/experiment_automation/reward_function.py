from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration for converting Tx/Tz measurements into one scalar score.

    target_bias is compared in log-space so that being off by x2 and x1/2
    is penalized symmetrically.
    """

    target_bias: float = 10.0
    bias_definition: str = "Tx_over_Tz"
    lambda_bias: float = 1.0
    invalid_reward: float = -1e12


def _is_valid_positive(x: float) -> bool:
    try:
        x = float(x)
    except Exception:
        return False
    return math.isfinite(x) and x > 0.0


def compute_bias(Tx: float, Tz: float, definition: str = "Tx_over_Tz") -> float:
    """Compute the noise-bias proxy from the two measured lifetimes."""

    if definition == "Tx_over_Tz":
        return float(Tx) / float(Tz)

    if definition == "Tz_over_Tx":
        return float(Tz) / float(Tx)

    raise ValueError(f"Unknown bias definition: {definition}")


def score_calibration(
    Tx: float,
    Tz: float,
    config: RewardConfig | None = None,
) -> dict:
    """
    Convert measured lifetimes into one scalar reward.

    Larger Tx and Tz are rewarded. Bias far from the target is penalized.
    """

    if config is None:
        config = RewardConfig()

    if not _is_valid_positive(Tx) or not _is_valid_positive(Tz):
        return {
            "reward": config.invalid_reward,
            "Tx": Tx,
            "Tz": Tz,
            "bias": None,
            "lifetime_score": None,
            "bias_penalty": None,
            "valid": False,
        }

    bias = compute_bias(Tx, Tz, config.bias_definition)

    if not _is_valid_positive(bias):
        return {
            "reward": config.invalid_reward,
            "Tx": Tx,
            "Tz": Tz,
            "bias": bias,
            "lifetime_score": None,
            "bias_penalty": None,
            "valid": False,
        }

    lifetime_score = math.log(float(Tx)) + math.log(float(Tz))
    bias_error = math.log(float(bias)) - math.log(float(config.target_bias))
    bias_penalty = config.lambda_bias * bias_error**2

    reward = lifetime_score - bias_penalty

    return {
        "reward": float(reward),
        "Tx": float(Tx),
        "Tz": float(Tz),
        "bias": float(bias),
        "lifetime_score": float(lifetime_score),
        "bias_penalty": float(bias_penalty),
        "valid": True,
    }
