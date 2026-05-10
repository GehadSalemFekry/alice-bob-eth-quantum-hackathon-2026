import jax.numpy as jnp
import dynamiqs as dq

from  Model_to_halflife import measure_lifetime
import drift_hamiltonians as driftH
import drift_hamiltonian_correction_extentions as driftCorr
import static_hamiltonians as statH
import Wigner_representation as Wrep
import plot_expectation_value as PlotExp


def IasonassToy1(state, epsiloRe, epsilonIm, gRe, gIm):
    """
    choose state = +z or +x strings
    """
    data = {
        "initial_state": state,
        "Hilbert_space_large": 20,
        "Hilbert_space_cutted_for_solution": 10,
        "kappa_b": 1.0,
        "eps_d": jnp.complex64(epsiloRe + epsilonIm*1j) ,
        "g_2": jnp.complex64(gRe + gIm*1j),
        "kappa_a": 2.0,
        "Hamiltonian": None,
        "Jump_a": None,
        "Jump_b": None,
        "tfinal": 100.0
    }
    halflife , _ = measure_lifetime(data)
    return halflife

def IasonassToy2():
    """
    This one is for plotting
    """
    data = {
        "initial_state": "+x",
        "Hilbert_space_large": 30,
        "Hilbert_space_cutted_for_solution": 10,
        "kappa_b": 1.0,
        "eps_d": jnp.complex64(4 + 0*1j) ,
        "g_2": jnp.complex64(1 + 0*1j),
        "kappa_a": 2.0,
        "Hamiltonian": None,
        "Jump_a": None,
        "Jump_b": None,
        "tfinal": 1.0,
        "plot": True
    }
    halflife , _ = measure_lifetime(data)
    return halflife

def IasonassToy3():
    """
    This one is for choosing static Hamiltonians
    """
    data_build_H = {
        "Hilbert_space_large": 15,
        "Hilbert_space_cutted_for_solution": 5,
        "knobs": [1, 0, 4, 0],
    }
    H = statH.build_standard_cat(data_build_H)
    data_halflife = {
        "initial_state": "+z",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 50.0,
        "plot": True}
    data = {**data_halflife, **data_build_H}
    halflife , _ = measure_lifetime(data)
    return halflife

def IasonassToy4():
    """Testing Drift"""
    data_build_H = {
        "Hilbert_space_large": 30,
        "Hilbert_space_cutted_for_solution": 10,
        "knobs": [1, 0, 4, 0],
        "f": lambda t: jnp.sin(2.0 * jnp.pi * t),
    }
    H = driftH.build_amplitude_drift(data_build_H)
    data_halflife = {
        "initial_state": "+z",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 10.0,
        "plot": True}
    data = {**data_halflife, **data_build_H}
    halflife , _ = measure_lifetime(data)
    return halflife

def IasonassToy5():
    from QHack26.src.mesolver_method_optimization import mesosolver_method_opt
    """Testing time per mesolve method"""
    data_build_H = {
        "Hilbert_space_large": 30,
        "Hilbert_space_cutted_for_solution": 10,
        "knobs": [1, 0, 4, 0],
        "f": lambda t: jnp.sin(2.0 * jnp.pi * t),
    }
    H = driftH.build_amplitude_drift(data_build_H)
    data_halflife = {
        "initial_state": "+z",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 10.0,
        "plot": False}
    data = {**data_halflife, **data_build_H}
    time_per_method = mesosolver_method_opt(data)
    return time_per_method

def IasonassToy6():
    """Testing drifting, amplification function"""
    data_build_H = {
        "Hilbert_space_large": 30,
        "Hilbert_space_cutted_for_solution": 10,
        "knobs": [2, 0, 4, 0],
    }
    H = driftH.build_amplitude_drift(data_build_H)
    data_halflife = {
        "initial_state": "+z",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 10.0,
        "plot": True}
    data = {**data_halflife, **data_build_H}
    halflife , _ = measure_lifetime(data)
    return halflife

def IasonassToy7():
    """Testing drifting, Kerr function"""
    data_build_H = {
        "Hilbert_space_large": 15,
        "Hilbert_space_cutted_for_solution": 5,
        "knobs": [1, 0, 3, -0.5],
        "drift_params": [0.5, 2.0 * jnp.pi, 0.0], # A, omega, phi for frequency drift
    }
    H = driftH.build_kerr_drift(data_build_H)
    data_halflife = {
        "initial_state": "+x",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 2.0,
        "plot": True}
    data = {**data_halflife, **data_build_H}
    halflife , _ = measure_lifetime(data)
    return halflife


def IasonassToy8():
    """Run Wigner evolution similar to Toy7 and save GIF to Figures/Wigner/My_first_test."""
    data_build_H = {
        "Hilbert_space_large": 15,
        "Hilbert_space_cutted_for_solution": 5,
        "knobs": [1, 0, 6, 0,0.4,0],
        "drift_params": [0.5, 2.0 * jnp.pi, 0.0], # A, omega, phi for frequency drift
    }
    H = statH.build_moon_cat(data_build_H)
    data_halflife = {
        "initial_state": "+x",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 2.0,
        "plot": True}
    data = {**data_halflife, **data_build_H}
    data["path_to_save"] = "Wigner/My_first_test/moon_cat_wigner.gif"

    out = Wrep.show_wigner_evolution(data)
    return out


def IasonassToy9():
    """Decay test for the corrected drift Hamiltonian with detuning."""
    data_build_H = {
        "Hilbert_space_large": 18,
        "Hilbert_space_cutted_for_solution": 6,
        # g2_Re, g2_Im, eps_d_Re, eps_d_Im, Delta
        "knobs": [1.0, 0.0, 4.0, 0.0, 0.35],
        "drift_params": [0.25, 2.0 * jnp.pi, 0.0],
    }
    H = driftCorr.build_frequency_drift(data_build_H)
    data_halflife = {
        "initial_state": "+x",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 1.0,
        "plot": True,
    }
    data = {**data_halflife, **data_build_H}
    halflife, _ = measure_lifetime(data)
    return halflife


def IasonassToy10():
    """Wigner test for the corrected drift Hamiltonian with detuning."""
    data_build_H = {
        "Hilbert_space_large": 14,
        "Hilbert_space_cutted_for_solution": 5,
        # g2_Re, g2_Im, eps_d_Re, eps_d_Im, Delta
        "knobs": [1.0, 0.0, 4.0, 0.0, 0.35],
        "drift_params": [0.25, 2.0 * jnp.pi, 0.0],
    }
    H = driftCorr.build_frequency_drift(data_build_H)
    data_halflife = {
        "initial_state": "+x",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 1.0,
        "nframes": 30,
    }
    data = {**data_halflife, **data_build_H}
    data["path_to_save"] = "Wigner/My_first_test/detuning_wigner.gif"

    out = Wrep.show_wigner_evolution(data)
    return out


def IasonassToy11():
    """Plot expectation value decay with saved figure."""
    data_build_H = {
        "Hilbert_space_large": 15,
        "Hilbert_space_cutted_for_solution": 5,
        "knobs": [1.0, 0.0, 4.0, 0.0],
    }
    H = statH.build_standard_cat(data_build_H)
    data_halflife = {
        "initial_state": "+x",
        "Hamiltonian": H,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "tfinal": 2.0,
    }
    data = {**data_halflife, **data_build_H}
    data["plotSave"] = "Expectation_values/standard_cat_decay.png"

    halflife, plot_path = PlotExp.plot_expectation_value(data)
    return halflife, plot_path


if __name__ == "__main__":
    IasonassToy10()
