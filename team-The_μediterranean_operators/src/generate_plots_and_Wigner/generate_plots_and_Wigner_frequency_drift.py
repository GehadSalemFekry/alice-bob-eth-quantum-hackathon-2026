"""Generate plots and Wigner GIFs for frequency drift Hamiltonian."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import plot_expectation_value as PlotExp
import Wigner_representation as Wrep
import drift_hamiltonians as driftH


def generate_frequency_drift():
    """Generate plots and Wigner GIFs for frequency drift with parameter variations."""
    
    # Parameters to sweep
    # increase frequency-drift amplitude effects (x2)
    amp_values = [0.2, 0.6]
    g2_re = 1.0
    eps_d_re = 4.0
    omega = 2.0 * jnp.pi
    phi = 0.0
    
    for amp in amp_values:
        for initial_state in ["+z", "+x"]:
            data_build_H = {
                "Hilbert_space_large": 16,
                "Hilbert_space_cutted_for_solution": 5,
                "knobs": [g2_re, 0.0, eps_d_re, 0.0],
                "drift_params": [amp, omega, phi],
            }
            
            H = driftH.build_frequency_drift(data_build_H)
            
            # Expectation value plot
            tfinal = 200.0 if initial_state == "+z" else 1.0
            data_plot = {
                "initial_state": initial_state,
                "Hamiltonian": H,
                "kappa_a": 1.0,
                "kappa_b": 10.0,
                "tfinal": tfinal,
                "plotSave": f"Plots/frequency_drift_{initial_state}_A{amp}_T{tfinal}.png",
            }
            data_plot.update(data_build_H)
            
            print(f"[PLOT] Frequency drift: {initial_state}, A={amp}, T={tfinal}")
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
                "path_to_save": f"Wigner/frequency_drift_{initial_state}_A{amp}_T{wigner_tfinal}.gif",
            }
            data_wigner.update(data_build_H)
            
            print(f"[WIGNER] Frequency drift: {initial_state}, A={amp}, T={wigner_tfinal}, slow-fps")
            Wrep.show_wigner_evolution(data_wigner)


if __name__ == "__main__":
    generate_frequency_drift()
