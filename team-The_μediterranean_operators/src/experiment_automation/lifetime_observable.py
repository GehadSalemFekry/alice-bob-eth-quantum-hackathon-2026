from __future__ import annotations

import jax.numpy as jnp
import dynamiqs as dq

from Model_to_halflife import robust_exp_fit


def _auto_observable(initial_state: str) -> str:
    if "x" in initial_state:
        return "sx"
    if "z" in initial_state:
        return "sz"
    return "sz"


def measure_lifetime_observable(
    data: dict,
    observable: str = "auto",
    return_trace: bool = False,
) -> float | dict:
    """
    Lifetime measurement with explicit observable selection.

    The provided Model_to_halflife.measure_lifetime currently fits sz. This
    adapter keeps the same simulation idea but lets us choose the fitted signal:
        +z -> sz -> Tz
        +x -> sx -> Tx

    This file is our measurement adapter; it does not modify the provided files.
    """

    initial_state = data["initial_state"]
    na = int(data["Hilbert_space_large"])
    nb = int(data["Hilbert_space_cutted_for_solution"])

    if observable == "auto":
        observable = _auto_observable(initial_state)

    if observable not in {"sx", "sz"}:
        raise ValueError(f"Unknown observable: {observable}")

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    kappa_b = data.get("kappa_b", 10.0)
    kappa_a = data.get("kappa_a", 1.0)
    eps_d = data.get("eps_d", 4.0)
    g_2 = data.get("g_2", 1.0)

    alpha_estimate = data.get("alpha")
    if alpha_estimate is None:
        eps_2 = 2 * g_2 * eps_d / kappa_b
        kappa_2 = 4 * jnp.abs(g_2) ** 2 / kappa_b
        alpha_estimate = jnp.sqrt(2 / kappa_2 * (eps_2 - kappa_a / 4))

    H = data.get("Hamiltonian")
    if H is None:
        H = (
            jnp.conj(g_2) * a @ a @ b.dag()
            + g_2 * a.dag() @ a.dag() @ b
            - eps_d * b.dag()
            - jnp.conj(eps_d) * b
        )

    loss_b = data.get("Jump_b")
    if loss_b is None:
        loss_b = jnp.sqrt(kappa_b) * b

    loss_a = data.get("Jump_a")
    if loss_a is None:
        loss_a = jnp.sqrt(kappa_a) * a

    tsave = jnp.linspace(0, data["tfinal"], data.get("n_time_points", 100))

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

    if initial_state not in basis:
        raise ValueError(f"Unknown initial_state: {initial_state}")

    sx = (1j * jnp.pi * (a.dag() @ a)).expm()

    sz_single_mode = basis["+z"] @ basis["+z"].dag() - basis["-z"] @ basis["-z"].dag()
    sz = dq.tensor(sz_single_mode, dq.eye(nb))

    psi0 = dq.tensor(basis[initial_state], dq.fock(nb, 0))

    res = dq.mesolve(
        H,
        [loss_b, loss_a],
        psi0,
        tsave,
        options=dq.Options(progress_meter=False),
        exp_ops=[sx, sz],
    )

    sx_t = res.expects[0, :].real
    sz_t = res.expects[1, :].real

    y = sx_t if observable == "sx" else sz_t
    x = res.tsave

    fit = robust_exp_fit(x, y)
    lifetime = float(fit["popt"][1])

    if return_trace:
        return {
            "lifetime": lifetime,
            "observable": observable,
            "initial_state": initial_state,
            "alpha_estimate": alpha_estimate,
            "t": x,
            "y": y,
            "y_fit": fit["y_fit"],
        }

    return lifetime
