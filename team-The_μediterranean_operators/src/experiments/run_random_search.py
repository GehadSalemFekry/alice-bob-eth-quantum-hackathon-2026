import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from evaluate_calibration import DEFAULT_MODEL, evaluate_calibration, sample_knobs_from_bounds
from reward_function import RewardConfig


def main():
    rng = np.random.default_rng(123)
    model_name = DEFAULT_MODEL

    config = RewardConfig(
        target_bias=10.0,
        bias_definition="Tx_over_Tz",
        lambda_bias=1.0,
    )

    n_trials = 10
    best = None

    for trial in range(n_trials):
        knobs = sample_knobs_from_bounds(rng, model_name=model_name)

        try:
            result = evaluate_calibration(
                knobs,
                model_name=model_name,
                tfinal=5.0,
                n_time_points=100,
                reward_config=config,
            )
        except Exception as exc:
            print(f"[{trial:03d}] failed: {exc}")
            continue

        if best is None or result["reward"] > best["reward"]:
            best = result

        print(
            f"[{trial:03d}] "
            f"reward={result['reward']:.4g} "
            f"Tx={result['Tx']:.4g} "
            f"Tz={result['Tz']:.4g} "
            f"bias={result['bias']:.4g} "
            f"knobs={result['knobs']}"
        )

    print("\nBest result")
    print("-----------")
    if best is None:
        print("No valid result found.")
    else:
        for key, value in best.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
