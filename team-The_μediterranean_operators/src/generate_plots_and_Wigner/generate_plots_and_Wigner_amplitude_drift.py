"""Generate plots and Wigner GIFs for amplitude drift Hamiltonian."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import plot_expectation_value as PlotExp
import Wigner_representation as Wrep
import drift_hamiltonians as driftH


def generate_amplitude_drift():
    """Generate plots and Wigner GIFs for amplitude drift with parameter variations."""
    
    # Consistent parameters across all figures
    g2_re = 1.0
    eps_d_re = 4.0
    
    for initial_state in ["+z", "+x"]:
        data_build_H = {
            "Hilbert_space_large": 18,
            "Hilbert_space_cutted_for_solution": 6,
            "knobs": [g2_re, 0.0, eps_d_re, 0.0],
            # stronger amplitude drift: factor 2
            "f": lambda t: 2.0 * jnp.sin(2.0 * jnp.pi * t),
        }
        
        H = driftH.build_amplitude_drift(data_build_H)
        
        # Expectation value plot
        tfinal = 200.0 if initial_state == "+z" else 1.0
        data_plot = {
            "initial_state": initial_state,
            "Hamiltonian": H,
            "kappa_a": 1.0,
            "kappa_b": 10.0,
            "tfinal": tfinal,
            "plotSave": f"Plots/amplitude_drift_{initial_state}_g2{g2_re}_T{tfinal}.png",
        }
        data_plot.update(data_build_H)
        
        print(f"[PLOT] Amplitude drift: {initial_state}, g2={g2_re}, T={tfinal}")
        PlotExp.plot_expectation_value(data_plot)
        
        # Wigner GIF
        wigner_tfinal = tfinal / 2.0
        data_wigner = {
            "initial_state": initial_state,
            "Hamiltonian": H,
            "kappa_a": 1.0,
            "kappa_b": 10.0,
            "tfinal": wigner_tfinal,
            "nframes": 30,
            "frame_duration_ms": 200,
            "path_to_save": f"Wigner/amplitude_drift_{initial_state}_g2{g2_re}_T{wigner_tfinal}.gif",
        }
        data_wigner.update(data_build_H)
        
        print(f"[WIGNER] Amplitude drift: {initial_state}, g2={g2_re}, T={wigner_tfinal}, slow-fps")
        Wrep.show_wigner_evolution(data_wigner)


if __name__ == "__main__":
    generate_amplitude_drift()
