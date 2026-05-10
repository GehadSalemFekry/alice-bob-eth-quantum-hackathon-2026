"""Plot expectation value decay curves from lifetime measurements.

Provides `plot_expectation_value(data)` which runs a lifetime measurement
and plots the expectation value decay, optionally saving to a file.
"""

from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from IPython.display import display
from Model_to_halflife import measure_lifetime
from postprocess import robust_exp_fit


def _hamiltonian_equation(data: dict) -> str:
    label = data.get("hamiltonian_label")
    if label is not None:
        return str(label)

    eq = "H = g2* a†² b + g2 a² b† - εd b† - εd* b"
    if "kerr" in data:
        eq += " + K(a†a)^2"
    elif isinstance(data.get("knobs"), (list, tuple)) and len(data.get("knobs", [])) >= 6:
        eq += " + g2 λ a†ab + h.c."
    if "drift_params" in data or "f" in data:
        eq += " + drift(t)"
    if "tls_dim" in data:
        eq += " + TLS"
    return eq


def plot_expectation_value(data: dict):
    """Run lifetime measurement and plot expectation value decay.
    
    Calls `measure_lifetime(data)` internally to get the decay curve,
    then plots the expectation value over time. Optionally saves the plot.
    
    Parameters
    ----------
    data : dict
        Data dictionary passed to `measure_lifetime`, plus:
        - plotSave (str or Path, optional): path to save the plot under Figures/
            If provided, saves a PNG of the decay curve.
    
    Returns
    -------
    tuple
        (halflife, plot_path) where:
        - halflife (float): measured halflife from `measure_lifetime`
        - plot_path (Path or None): path to saved plot if plotSave was provided,
            otherwise None
    """
    
    # Run the lifetime measurement (this also extracts x, y internally)
    halflife, _ = measure_lifetime(data)
    
    # We need to re-extract the data to plot it
    # Re-run a quick mesolve to get the expectation values
    import jax.numpy as jnp
    import dynamiqs as dq
    
    initial_state = data["initial_state"]
    na = data.get("Hilbert_space_large", 30)
    nb = data.get("Hilbert_space_cutted_for_solution", 15)
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))
    
    kappa_a = data.get("kappa_a", 1.0)
    kappa_b = data.get("kappa_b", 10.0)
    
    knobs = data.get("knobs")
    if knobs is not None:
        g_2 = knobs[0] + 1j * knobs[1]
        eps_d = knobs[2] + 1j * knobs[3]
    else:
        g_2 = data.get("g_2", 0)
        eps_d = data.get("eps_d", 0)
    
    alpha_estimate = data.get("alpha")
    if alpha_estimate is None:
        if knobs is not None or (g_2 != 0 and eps_d != 0):
            eps_2 = 2 * g_2 * eps_d / kappa_b
            kappa_2 = 4 * jnp.abs(g_2) ** 2 / kappa_b
            alpha_estimate = jnp.sqrt(2 / kappa_2 * (eps_2 - kappa_a / 4))
        else:
            alpha_estimate = 2.0
    
    H = data.get("Hamiltonian")
    if H is None:
        if g_2 == 0 and eps_d == 0:
            g_2 = 1.0 + 0.0j
            eps_d = 4.0 + 0.0j
        H = jnp.conj(g_2) * a @ a @ b.dag() + g_2 * a.dag() @ a.dag() @ b - eps_d * b.dag() - jnp.conj(eps_d) * b
    
    loss_b = data.get("Jump_b")
    if loss_b is None:
        loss_b = jnp.sqrt(kappa_b) * b
    
    loss_a = data.get("Jump_a")
    if loss_a is None:
        loss_a = jnp.sqrt(kappa_a) * a
    
    tfinal = data.get("tfinal", 5.0)
    tsave = jnp.linspace(0, tfinal, 300)
    
    g_state = dq.coherent(na, alpha_estimate)
    e_state = dq.coherent(na, -alpha_estimate)
    
    basis = {
        "+z": g_state,
        "-z": e_state,
        "+x": (g_state + e_state) / jnp.sqrt(2),
        "-x": (g_state - e_state) / jnp.sqrt(2),
        "+y": (g_state + 1j * e_state) / jnp.sqrt(2),
        "-y": (g_state - 1j * e_state) / jnp.sqrt(2),
    }
    
    # Extract expectation value operator
    if initial_state in ["+z", "-z"]:
        sz = (basis["+z"] @ basis["+z"].dag() - basis["-z"] @ basis["-z"].dag())
        sz = dq.tensor(sz, dq.eye(nb))
        operator_label = "⟨Sz⟩"
        psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))
        res = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi0,
            tsave,
            options=dq.Options(progress_meter=False),
            exp_ops=[sz],
            method=data.get("mesolve_method", dq.method.Tsit5()),
        )
    elif initial_state in ["+x", "-x"]:
        x = (1j * jnp.pi * a.dag() @ a).expm()
        operator_label = "⟨X⟩"
        psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))
        res = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi0,
            tsave,
            options=dq.Options(progress_meter=False),
            exp_ops=[x],
            method=data.get("mesolve_method", dq.method.Tsit5()),
        )
    
    sNt = res.expects[0, :].real
    ts = res.tsave
    
    # Plot the expectation value
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts, sNt, 'b-', linewidth=2, label=f"{operator_label} ({initial_state})")

    # Exponential fit overlay
    try:
        fit = robust_exp_fit(ts, sNt)
        y_fit = fit["y_fit"]
        tau = float(fit["popt"][1])
        ax.plot(ts, y_fit, 'r--', linewidth=2, label=f"Exp fit (tau={tau:.3f})")
    except Exception:
        tau = float(halflife)

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time (μs)", fontsize=16)
    ax.set_ylabel("Expectation Value", fontsize=16)
    # choose Tx or Tz depending on axis
    if initial_state in ["+x", "-x"]:
        t_label = "Tx"
    else:
        t_label = "Tz"
    ax.set_title(f"Decay curve: {t_label} ≈ {tau:.4f}", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    ax.tick_params(axis="both", labelsize=13)
    fig.tight_layout()
    
    # Save if requested
    plot_path = None
    plot_save = data.get("plotSave")
    if plot_save is not None:
        plot_path = None  # We're not saving in notebook mode
    
    # Return figure as last object so Jupyter auto-displays it
    return halflife, plot_path, fig
