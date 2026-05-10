import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from evaluate_calibration import DEFAULT_MODEL, default_knobs, evaluate_calibration
from reward_function import RewardConfig


def main():
    model_name = DEFAULT_MODEL
    knobs = default_knobs(model_name)

    config = RewardConfig(
        target_bias=10.0,
        bias_definition="Tx_over_Tz",
        lambda_bias=1.0,
    )

    result = evaluate_calibration(
        knobs,
        model_name=model_name,
        tfinal=5.0,
        n_time_points=100,
        reward_config=config,
    )

    print("\nBaseline result")
    print("---------------")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
