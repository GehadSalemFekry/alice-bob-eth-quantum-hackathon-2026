"""Wigner representation helper.

Provides `show_wigner_evolution(data)` which runs a simulation (like
`measure_lifetime`) and displays/saves a Wigner-function GIF.

Expected `data` keys (same as `measure_lifetime`):
- initial_state (str): "+x", "+z", "-z", "-x", "+y", "-y"
    - Alpha is auto-estimated from knobs/kappa unless explicitly provided
- Hilbert_space_large (int)
- Hilbert_space_cutted_for_solution (int)
- knobs (sequence) [g2_Re, g2_Im, eps_d_Re, eps_d_Im]
- kappa_a (float)
- kappa_b (float)
- alpha (float) optional: if provided, uses this value; otherwise computes from
    α = √(2/κ₂ · (ε₂ - κₐ/4)) where ε₂ = 2g₂ε_d/κ_b and κ₂ = 4|g₂|²/κ_b
- Jump_a, Jump_b optional
- Hamiltonian optional
- tfinal (float)
- path_to_save (str) optional: directory name under ../Figures to save GIF

This module delegates plotting to `Wigner_function_plotting` if available.
"""

from pathlib import Path
from typing import Optional
import jax.numpy as jnp
import dynamiqs as dq


def show_wigner_evolution(data: dict) -> Optional[Path]:
    """Run a short simulation and show/save Wigner evolution GIF.

    Returns the path to the saved GIF when `path_to_save` is provided,
    otherwise returns None. In a Jupyter notebook the GIF will be displayed
    inline when possible.
    """
    # Build Hilbert-space operators
    na = data.get("Hilbert_space_large", 30)
    nb = data.get("Hilbert_space_cutted_for_solution", 10)
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    # parse knobs / explicit parameters
    knobs = data.get("knobs")
    if knobs is not None:
        g_2 = knobs[0] + 1j * knobs[1]
        eps_d = knobs[2] + 1j * knobs[3]
    else:
        g_2 = data.get("g_2", 0)
        eps_d = data.get("eps_d", 0)

    kappa_a = data.get("kappa_a", 1.0)
    kappa_b = data.get("kappa_b", 10.0)

    # Build Hamiltonian and losses if not provided
    H = data.get("Hamiltonian")
    if H is None:
        if g_2 == 0 and eps_d == 0:
            g_2 = 1.0 + 0.0j
            eps_d = 4.0 + 0.0j
        H = jnp.conj(g_2) * a @ a @ b.dag() + g_2 * a.dag() @ a.dag() @ b - eps_d * b.dag() - jnp.conj(eps_d) * b

    loss_b = data.get("Jump_b") or jnp.sqrt(kappa_b) * b
    loss_a = data.get("Jump_a") or jnp.sqrt(kappa_a) * a

    tfinal = data.get("tfinal", 5.0)
    tsave = jnp.linspace(0, tfinal, data.get("nframes", 40))

    # estimate alpha if needed (same heuristic as measure_lifetime)
    alpha = data.get("alpha")
    if alpha is None and g_2 != 0 and eps_d != 0:
        eps_2 = 2 * g_2 * eps_d / kappa_b
        kappa_2 = 4 * jnp.abs(g_2) ** 2 / kappa_b
        alpha = jnp.sqrt(2 / kappa_2 * (eps_2 - kappa_a / 4))
    if alpha is None:
        alpha = 2.0

    g_state = dq.coherent(na, alpha)
    e_state = dq.coherent(na, -alpha)
    basis = {
        "+z": g_state,
        "-z": e_state,
        "+x": (g_state + e_state) / jnp.sqrt(2),
        "-x": (g_state - e_state) / jnp.sqrt(2),
        "+y": (g_state + 1j * e_state) / jnp.sqrt(2),
        "-y": (g_state - 1j * e_state) / jnp.sqrt(2),
    }

    initial_state = data.get("initial_state", "+z")
    psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))

    # Delegate plotting to existing helper if available
    try:
        import Wigner_function_plotting as Wfp
    except Exception:
        Wfp = None

    path_to_save = data.get("path_to_save")
    out_path = None
    if path_to_save:
        path_str = str(path_to_save)
        figures_root = Path(__file__).resolve().parents[1] / "Figures"
        if path_str.endswith("/") or path_str.endswith("\\"):
            out_dir = figures_root / Path(path_str)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "wigner.gif"
        else:
            target = Path(path_str)
            if target.suffix.lower() != ".gif":
                target = target.with_suffix(target.suffix + ".gif") if target.suffix != "" else target.with_suffix('.gif')
            out_path = figures_root / target
            out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run evolution once
    res = dq.mesolve(
        H,
        [loss_b, loss_a],
        psi0,
        tsave,
        options=dq.Options(progress_meter=False),
        method=data.get("mesolve_method", dq.method.Tsit5()),
    )

    if Wfp is not None and hasattr(Wfp, "create_wigner_gif_from_result"):
        if out_path is not None:
            Wfp.create_wigner_gif_from_result(data, res, out_path, n_frames=data.get("nframes", 40))
            if out_path.exists():
                print(f"Wigner GIF created: {out_path}")
            else:
                print(f"Warning: GIF was not created at {out_path}")
            return out_path if out_path.exists() else None
        return None

    # Fallback: try original create_wigner_gif
    if Wfp is not None and hasattr(Wfp, "create_wigner_gif"):
        if out_path is not None:
            Wfp.create_wigner_gif(data, out_path, n_frames=data.get("nframes", 40))
            if out_path.exists():
                print(f"Wigner GIF saved to: {out_path}")
                return out_path
            else:
                print(f"Warning: create_wigner_gif did not produce file at {out_path}")
                return None
        return None

    # If helper not available, use plot_wigner for static display
    if Wfp is not None and hasattr(Wfp, "plot_wigner"):
        Wfp.plot_wigner(data, res)
        return None

    raise ImportError(
        "Wigner plotting helper not available. Install or provide Wigner_function_plotting with create_wigner_gif_from_result(data,res,out_path)."
    )
