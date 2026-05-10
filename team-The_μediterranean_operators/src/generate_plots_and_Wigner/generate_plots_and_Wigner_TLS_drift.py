"""Generate plots and Wigner GIFs for TLS drift Hamiltonian."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import plot_expectation_value as PlotExp
import Wigner_representation as Wrep
import drift_hamiltonians as driftH


def generate_tls_drift():
    """Generate plots and Wigner GIFs for TLS drift with parameter variations."""
    
    # Parameters to sweep
    # increase TLS-driven coupling amplitudes (x2)
    tls_amp_values = [0.1, 0.2]
    g2_re = 1.0
    eps_d_re = 4.0
    omega = 2.0 * jnp.pi
    phi = 0.0
    
    for tls_amp in tls_amp_values:
        for initial_state in ["+z", "+x"]:
            data_build_H = {
                "Hilbert_space_large": 15,
                "Hilbert_space_cutted_for_solution": 5,
                "knobs": [g2_re, 0.0, eps_d_re, 0.0],
                "drift_params": [tls_amp, omega, phi],
                "tls_dim": 2,
            }
            
            H = driftH.build_tls_drift(data_build_H)
            
            # Expectation value plot
            tfinal = 200.0 if initial_state == "+z" else 1.0
            data_plot = {
                "initial_state": initial_state,
                "Hamiltonian": H,
                "kappa_a": 1.0,
                "kappa_b": 10.0,
                "tfinal": tfinal,
                "plotSave": f"Plots/tls_drift_{initial_state}_A{tls_amp}_T{tfinal}.png",
            }
            data_plot.update(data_build_H)
            
            print(f"[PLOT] TLS drift: {initial_state}, A={tls_amp}, T={tfinal}")
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
                "path_to_save": f"Wigner/tls_drift_{initial_state}_A{tls_amp}_T{wigner_tfinal}.gif",
            }
            data_wigner.update(data_build_H)
            
            print(f"[WIGNER] TLS drift: {initial_state}, A={tls_amp}, T={wigner_tfinal}, slow-fps")
            Wrep.show_wigner_evolution(data_wigner)


if __name__ == "__main__":
    generate_tls_drift()
