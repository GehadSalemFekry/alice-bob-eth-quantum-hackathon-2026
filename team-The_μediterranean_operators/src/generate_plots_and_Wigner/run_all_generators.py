"""Master script to run all plot and Wigner generators."""

import sys
import subprocess
from pathlib import Path

# Get the directory of this script
script_dir = Path(__file__).parent

generators = [
    "generate_plots_and_Wigner_standard.py",
    "generate_plots_and_Wigner_Kerr_static.py",
    "generate_plots_and_Wigner_moon_cat.py",
    "generate_plots_and_Wigner_cross_kerr.py",
    "generate_plots_and_Wigner_amplitude_drift.py",
    "generate_plots_and_Wigner_frequency_drift.py",
    "generate_plots_and_Wigner_Kerr_drift.py",
    "generate_plots_and_Wigner_TLS_drift.py",
]

# generators = [
#     "generate_plots_and_Wigner_moon_cat.py",
# ]


def run_all():
    """Run all generators sequentially."""
    for gen in generators:
        gen_path = script_dir / gen
        print(f"\n{'='*70}")
        print(f"Running: {gen}")
        print(f"{'='*70}\n")
        
        try:
            subprocess.run([sys.executable, str(gen_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {gen} failed with exit code {e.returncode}")
        except Exception as e:
            print(f"ERROR: {gen} failed with exception: {e}")


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*70}")
    print("All generators completed!")
    print(f"{'='*70}\n")
