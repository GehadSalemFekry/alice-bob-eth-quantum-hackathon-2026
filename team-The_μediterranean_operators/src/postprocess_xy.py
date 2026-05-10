from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model_to_halflife import measure_lifetime
from Wigner_function_plotting import create_wigner_gif


def main() -> None:
    out_dir = Path(tempfile.gettempdir()) / "qhack26_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for state in ["+x", "-x"]:
        plt.close("all")
        data = {
            "initial_state": state,
            "Hilbert_space_large": 12,
            "Hilbert_space_cutted_for_solution": 4,
            "knobs": [4.0, 0.0, 1.0, 0.0],
            "tfinal": 1.0,
            "plot": True,
            "plotWigner": True,
        }

        halflife , _ = measure_lifetime(data)

        saved = []
        state_tag = state.replace("+", "plus").replace("-", "minus")
        for i, fignum in enumerate(plt.get_fignums(), start=1):
            fig = plt.figure(fignum)
            out_path = out_dir / f"{state_tag}_plot_{i}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            saved.append(out_path)

        print(f"\nInitial state: {state}")
        print(f"Halflife: {halflife}")
        print("Saved plots:")
        for path in saved:
            print(path)

        print("Generating Wigner function GIF...")
        gif_path = out_dir / f"{state_tag}_wigner_evolution.gif"
        create_wigner_gif(data, gif_path)
        print(f"Saved Wigner GIF: {gif_path}")


if __name__ == "__main__":
    main()
