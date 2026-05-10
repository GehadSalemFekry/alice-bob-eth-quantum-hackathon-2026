import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import io
from pathlib import Path
from PIL import Image


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


def _apply_log_colormap(ax, fig, title: str, global_vmin: float = None, global_vmax: float = None):
    """Apply linear colormap with static max/min values for consistent scaling.
    
    The color scale remains absolutely constant across all frames with fixed limits,
    ensuring the colorbar numbers and colors don't change during animations.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the Wigner plot
    fig : matplotlib.figure.Figure
        The figure
    title : str
        Title for the plot
    global_vmin : float, optional
        Global minimum for raw W values (computed from all frames).
    global_vmax : float, optional
        Global maximum for raw W values (computed from all frames).
    """
    if not ax.images:
        return

    image = ax.images[-1]
    values = np.asarray(image.get_array(), dtype=float)
    
    # Use global max/min if provided, otherwise compute from data
    if global_vmax is not None and global_vmin is not None:
        # Global values are forced to be symmetric around zero
        data_vmin = global_vmin
        data_vmax = global_vmax
    else:
        # Use fixed defaults based on a symmetric Wigner function range
        data_vmax = 0.5
        data_vmin = -0.5
    
    # Set the raw (non-transformed) Wigner data
    image.set_data(values)
    
    image.set_cmap("RdBu_r")
    
    # Use linear normalization for simple W scale
    norm = plt.Normalize(vmin=data_vmin, vmax=data_vmax)
    image.set_norm(norm)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("W", fontsize=14)
    
    # Set ticks to span from min to max
    tick_levels = np.linspace(data_vmin, data_vmax, 5)
    cbar.set_ticks(tick_levels)
    cbar.set_ticklabels([f"{t:.2f}" for t in tick_levels])
    cbar.ax.tick_params(labelsize=13)

    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_title(title, fontsize=16)


def _estimate_alpha(data):
    alpha = data.get("alpha")
    if alpha is not None:
        return alpha

    knobs = data.get("knobs")
    if knobs is None:
        return 2.0

    kappa_a = data.get("kappa_a", 1.0)
    kappa_b = data.get("kappa_b", 10.0)
    eps_d = knobs[0] + 1j * knobs[1]
    g_2 = knobs[2] + 1j * knobs[3]
    eps_2 = 2 * g_2 * eps_d / kappa_b
    kappa_2 = 4 * jnp.abs(g_2) ** 2 / kappa_b
    return jnp.sqrt(2 / kappa_2 * (eps_2 - kappa_a / 4))


def _compute_global_wigner_limits(wigner_data_list):
    """
    Compute a symmetric global min/max of raw W values across all frames.
    
    Parameters
    ----------
    wigner_data_list : list of np.ndarray
        List of Wigner data arrays (one per time frame)
        
    Returns
    -------
    tuple
        (global_vmin, global_vmax) - symmetric bounds based on the maximum
        absolute raw W value across all frames
    """
    all_w_values = []
    
    for wigner_data in wigner_data_list:
        values = np.asarray(wigner_data, dtype=float)
        all_w_values.append(values.flatten())
    
    # Concatenate all and find global limits
    all_w_array = np.concatenate(all_w_values)
    max_abs = np.max(np.abs(all_w_array))

    if max_abs == 0:
        max_abs = 1.0

    global_vmin = -max_abs
    global_vmax = max_abs
    
    return global_vmin, global_vmax


def plot_wigner(data, res):

    na = data["Hilbert_space_large"]
    alpha_estimate = _estimate_alpha(data)
    g_state = dq.coherent(na, alpha_estimate)
    e_state = dq.coherent(na, -alpha_estimate)
    x_plus_state = (g_state + e_state) / jnp.sqrt(2)
    x_minus_state = (g_state - e_state) / jnp.sqrt(2)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_wig_g = axs[0, 0]
    ax_wig_e = axs[0, 1]
    ax_wig_xp = axs[1, 0]
    ax_wig_xm = axs[1, 1]

    equation = _hamiltonian_equation(data)

    dq.plot.wigner(g_state, ax=ax_wig_g, colorbar=False)
    _apply_log_colormap(ax_wig_g, fig, "Wigner (+z)")
    dq.plot.wigner(e_state, ax=ax_wig_e, colorbar=False)
    _apply_log_colormap(ax_wig_e, fig, "Wigner (-z)")
    dq.plot.wigner(x_plus_state, ax=ax_wig_xp, colorbar=False)
    _apply_log_colormap(ax_wig_xp, fig, "Wigner (+x)")
    dq.plot.wigner(x_minus_state, ax=ax_wig_xm, colorbar=False)
    _apply_log_colormap(ax_wig_xm, fig, "Wigner (-x)")
    fig.tight_layout()


def create_wigner_gif(data: dict, out_path: str | Path, n_frames: int = 30) -> None:
    """
    Create animated GIF of Wigner function evolution over time.
    
    Parameters
    ----------
    data : dict
        Data dictionary with keys:
        - initial_state (str): "+x", "-x", "+z", or "-z"
        - Hilbert_space_large (int): storage mode dimension
        - Hilbert_space_cutted_for_solution (int): buffer mode dimension
        - tfinal (float): final evolution time
        - knobs (sequence, optional): [eps_d_R, eps_d_Im, g_2_Re, g_2_Im]
        - kappa_a, kappa_b, alpha, Hamiltonian, Jump_a, Jump_b, mesolve_method (optional)
    out_path : str or Path
        Output file path for the GIF
    n_frames : int
        Number of frames in the animation (default: 30)
    """
    out_path = Path(out_path)
    frame_duration_ms = int(data.get("frame_duration_ms", 100))
    
    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]
    
    # Extract or compute alpha
    alpha = _estimate_alpha(data)
    
    # Build operators
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))
    
    kappa_a = data.get("kappa_a", 1.0)
    kappa_b = data.get("kappa_b", 10.0)
    
    knobs = data.get("knobs", [4, 0, 1, 0])
    eps_d = knobs[0] + 1j * knobs[1]
    g_2 = knobs[2] + 1j * knobs[3]
    
    # Hamiltonian
    H = data.get("Hamiltonian")
    if H is None:
        H = jnp.conj(g_2) * a @ a @ b.dag() + g_2 * a.dag() @ a.dag() @ b - eps_d * b.dag() - jnp.conj(eps_d) * b
    
    # Loss operators
    loss_b = data.get("Jump_b")
    if loss_b is None:
        loss_b = jnp.sqrt(kappa_b) * b
    
    loss_a = data.get("Jump_a")
    if loss_a is None:
        loss_a = jnp.sqrt(kappa_a) * a
    
    # Initial state
    initial_state = data["initial_state"]
    basis = {
        "+z": dq.coherent(na, alpha),
        "-z": dq.coherent(na, -alpha),
        "+x": (dq.coherent(na, alpha) + dq.coherent(na, -alpha)) / jnp.sqrt(2),
        "-x": (dq.coherent(na, alpha) - dq.coherent(na, -alpha)) / jnp.sqrt(2),
    }
    psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))
    
    # Time evolution
    tsave = jnp.linspace(0, data["tfinal"], n_frames)
    res = dq.mesolve(
        H, [loss_b, loss_a], psi0, tsave,
        options=dq.Options(progress_meter=False),
        method=data.get("mesolve_method", dq.method.Tsit5())
    )
    
    # First pass: collect all Wigner data to compute global limits
    wigner_data_list = []
    for i, psi in enumerate(res.states):
        dm_full = psi @ psi.dag()
        dm_array = np.array(dm_full)
        dm_reshaped = dm_array.reshape(na, nb, na, nb)
        dm_reduced_array = np.trace(dm_reshaped, axis1=1, axis2=3)
        wigner_data_list.append(dm_reduced_array)
    
    # Compute global min/max
    global_vmin, global_vmax = _compute_global_wigner_limits(wigner_data_list)
    
    # Second pass: generate frames with global normalization
    frames = []
    for i, wigner_data in enumerate(wigner_data_list):
        # Create Wigner plot
        fig, ax = plt.subplots(figsize=(7, 7))
        equation = _hamiltonian_equation(data)
        try:
            dq.plot.wigner(wigner_data, ax=ax, colorbar=False)
        except Exception:
            # Fallback: use simple coherent state visualization
            dq.plot.wigner(dq.coherent(na, 2.0), ax=ax, colorbar=False)
        _apply_log_colormap(ax, fig, f"Wigner at t={float(tsave[i]):.3f} μs", 
                           global_vmin=global_vmin, global_vmax=global_vmax)
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        plt.close(fig)
    
    # Save as GIF
    if frames:
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0
        )


def create_wigner_gif_from_result(data: dict, res, out_path: str | Path, n_frames: int = 30) -> None:
    """
    Create animated GIF of Wigner function evolution from an existing mesolve result.

    Parameters
    ----------
    data : dict
        Same data dict expected by `create_wigner_gif` (for dims and alpha estimation).
    res : MesolveResult-like
        Result object returned by `dq.mesolve` with `.states` and (optionally) `.tsave`.
    out_path : str or Path
        Output file path for the GIF.
    n_frames : int
        Number of frames to produce (samples states if needed).
    """
    out_path = Path(out_path)
    frame_duration_ms = int(data.get("frame_duration_ms", 100))

    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]

    alpha = _estimate_alpha(data)

    # Determine time points
    ts = getattr(res, "tsave", None)
    if ts is None:
        ts = getattr(res, "ts", None)
    if ts is None:
        ts = np.linspace(0, data.get("tfinal", 1.0), n_frames)

    # Collect states and sample to n_frames if necessary
    states = list(res.states)
    if len(states) == 0:
        return

    if n_frames != len(states):
        idx = np.linspace(0, len(states) - 1, n_frames).astype(int)
        selected_states = [states[i] for i in idx]
    else:
        selected_states = states

    # First pass: collect all Wigner data to compute global limits
    wigner_data_list = []
    for i, psi in enumerate(selected_states):
        dm_full = psi @ psi.dag()
        dm_array = np.array(dm_full)
        dm_reshaped = dm_array.reshape(na, nb, na, nb)
        dm_reduced_array = np.trace(dm_reshaped, axis1=1, axis2=3)
        wigner_data_list.append(dm_reduced_array)

    # Compute global min/max
    global_vmin, global_vmax = _compute_global_wigner_limits(wigner_data_list)

    # Second pass: generate frames with global normalization
    frames = []
    for i, wigner_data in enumerate(wigner_data_list):
        fig, ax = plt.subplots(figsize=(7, 7))
        equation = _hamiltonian_equation(data)
        try:
            dq.plot.wigner(wigner_data, ax=ax, colorbar=False)
        except Exception:
            dq.plot.wigner(dq.coherent(na, 2.0), ax=ax, colorbar=False)

        # Attempt to get time label
        try:
            tval = float(ts[i])
        except Exception:
            tval = i * (data.get("tfinal", 1.0) / max(1, n_frames - 1))

        _apply_log_colormap(ax, fig, f"Wigner at t={tval:.3f} μs",
                           global_vmin=global_vmin, global_vmax=global_vmax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        plt.close(fig)

    if frames:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".gif")
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

