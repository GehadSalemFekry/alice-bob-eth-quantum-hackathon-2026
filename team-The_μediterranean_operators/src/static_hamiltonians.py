import jax.numpy as jnp
import dynamiqs as dq


def _build_ops(na, nb):
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))
    return a, b


#Standar equation
def build_standard_cat(data):
    """Build standard cat Hamiltonian.

    Inputs: data (dict) containing model parameters
    - data["knobs"] = [gRe, gIm, epsRe, epsIm]
    - data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"] for Hilbert space sizes
    Output: H (Hamiltonian), [loss_b, loss_a], a, b
    """

    a, b = _build_ops(data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"])
    knobs = data["knobs"]
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]

    # H = g2* a†a†b + g2 a†a†b†? no, static cat Hamiltonian
    H = (jnp.conj(g2) * a @ a @ b.dag()
         + g2 * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b)

    return H

def build_kerr_cat(data):
    """Build kerr cat Hamiltonian.

    Inputs: data (dict) containing model parameters
    - data["knobs"] = [gRe, gIm, epsRe, epsIm, KRe, KIm]
    - data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"] for Hilbert space sizes
    Output: H (Hamiltonian), [loss_b, loss_a], a, b
    """

    a, b = _build_ops(data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"])
    knobs = data["knobs"]
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]
    K = knobs[4] + 1j * knobs[5]

    # H = g2* a†a†b + g2 a†a†b†? no, static cat Hamiltonian
    H = (jnp.conj(g2) * a @ a @ b.dag()
         + g2 * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b
         + K * a.dag() @ a @ a.dag() @ a)
         

    return H


def build_moon_cat(data):
    """
    Build moon cat Hamiltonian.
    Inputs: data (dict) containing model parameters
    - data["knobs"] = [gRe, gIm, epsRe, epsIm, lamRe, lamIm]
    - data["alpha_moon"] (float, optional): shift in the nonlinear moon term
    - data["kappa_b"], data["kappa_a"] for loss rates
    - data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"] for Hilbert space sizes
    Output: H (Hamiltonian), [loss_b, loss_a], a, b
    """
    # parse inputs to extract g2 and lam; reuse standard builder for base H
    knobs = data["knobs"]
    g2 = knobs[0] + 1j * knobs[1]
    eps = knobs[2] + 1j * knobs[3]
    lam = knobs[4] + 1j * knobs[5]
    alpha_moon = data.get("alpha_moon", 0.0)

    a, b = _build_ops(data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"])
    H = build_standard_cat(data)

    # add moon-specific lambda coupling: g2 * lam * (a†a - alpha_moon^2) b + h.c.
    moon_op = a.dag() @ a - alpha_moon**2
    H = H + (g2 * lam * moon_op @ b + jnp.conj(g2 * lam) * moon_op @ b.dag())

    return H
def build_cross_kerr(data):
    """Build a full cat Hamiltonian with an added cross-Kerr term.

    Inputs: data (dict) containing
    - data["knobs"] = [gRe, gIm, epsRe, epsIm, chiRe, chiIm]
      If chi is omitted, defaults to 1.0 + 0.0j.
    - data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"]

    Returns:
    - H = g2* a†² b + g2 a² b† - εd b† - εd* b + chi a†a b†b
    """
    a, b = _build_ops(data["Hilbert_space_large"], data["Hilbert_space_cutted_for_solution"])
    knobs = data.get("knobs", [1.0, 0.0, 4.0, 0.0, 1.0, 0.0])

    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]
    chi = knobs[4] + 1j * knobs[5]

    H = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
        + chi * (a.dag() @ a @ b.dag() @ b)
    )

    return H




