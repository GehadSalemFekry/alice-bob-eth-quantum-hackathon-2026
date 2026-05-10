"""Generate plots and Wigner GIFs for standard cat state Hamiltonian."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
import plot_expectation_value as PlotExp
import Wigner_representation as Wrep
import static_hamiltonians as statH


def generate_standard_cat():
    """Generate plots and Wigner GIFs for standard cat state with parameter variations."""
    
    # Consistent parameters across all figures
    g2_re = 1.0
    eps_d_re = 4.0
    
    for initial_state in ["+z", "+x"]:
        data_build_H = {
            "Hilbert_space_large": 16,
            "Hilbert_space_cutted_for_solution": 5,
            "knobs": [g2_re, 0.0, eps_d_re, 0.0],
        }
        
        H = statH.build_standard_cat(data_build_H)
        
        # Expectation value plot
        tfinal = 200.0 if initial_state == "+z" else 1.0
        data_plot = {
            "initial_state": initial_state,
            "Hamiltonian": H,
            "kappa_a": 1.0,
            "kappa_b": 10.0,
            "tfinal": tfinal,
            "plotSave": f"Plots/standard_cat_{initial_state}_g2{g2_re}_eps{eps_d_re}_T{tfinal}.png",
        }
        data_plot.update(data_build_H)
        
        print(f"[PLOT] Standard cat: {initial_state}, g2={g2_re}, eps_d={eps_d_re}, T={tfinal}")
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
            "path_to_save": f"Wigner/standard_cat_{initial_state}_g2{g2_re}_eps{eps_d_re}_T{wigner_tfinal}.gif",
        }
        data_wigner.update(data_build_H)
        
        print(f"[WIGNER] Standard cat: {initial_state}, g2={g2_re}, eps_d={eps_d_re}, T={wigner_tfinal}, slow-fps")
        Wrep.show_wigner_evolution(data_wigner)


if __name__ == "__main__":
    generate_standard_cat()
