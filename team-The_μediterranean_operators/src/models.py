import jax.numpy as jnp
import dynamiqs as dq

NA = 12
NB = 4
KAPPA_B = 10.0
KAPPA_A = 1.0


def _build_ops(na, nb):
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))
    return a, b


def compute_alpha(model_name, knobs):
    if model_name == "effective_cat":
        eps_2 = knobs[0] + 1j * knobs[1]
        kappa_2 = 1.0
    else:
        g2 = knobs[0] + 1j * knobs[1]
        eps_d = knobs[2] + 1j * knobs[3]
        eps_2 = 2 * g2 * eps_d / KAPPA_B
        kappa_2 = 4 * jnp.abs(g2) ** 2 / KAPPA_B
    return jnp.sqrt(2 / kappa_2 * (eps_2 - KAPPA_A / 4))


def build_standard_cat(knobs, drift_amp=0.0, drift_delta=0.0):
    a, b = _build_ops(NA, NB)
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]

    g2_d = g2 * (1 + drift_amp)

    H = (jnp.conj(g2_d) * a @ a @ b.dag()
         + g2_d * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b
         + drift_delta * a.dag() @ a)

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    return H, [loss_b, loss_a], a, b


def build_moon_cat(knobs, drift_amp=0.0, drift_delta=0.0):
    a, b = _build_ops(NA, NB)
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]
    lam = knobs[4] + 1j * knobs[5]

    g2_d = g2 * (1 + drift_amp)

    # H = g2* a†a†b + g2* aab† - eps_d b† - eps_d* b + g2*lam a†ab + (g2*lam)* a†ab†
    H = (jnp.conj(g2_d) * a @ a @ b.dag()
         + g2_d * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b
         + g2_d * lam * a.dag() @ a @ b
         + jnp.conj(g2_d * lam) * a.dag() @ a @ b.dag()
         + drift_delta * a.dag() @ a)

    loss_b = jnp.sqrt(KAPPA_B) * b
    loss_a = jnp.sqrt(KAPPA_A) * a

    return H, [loss_b, loss_a], a, b


def build_drift_cat(knobs, drift_amp=0.0, drift_delta=0.0):
    a, b = _build_ops(NA, NB)
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]
    delta_d = knobs[4]

    g2_d = g2 * (1 + drift_amp)

    # H = g2* a†a†b + g2* aab† - eps_d b† - eps_d* b
    total_detuning = drift_delta + delta_d
    H = (jnp.conj(g2_d) * a @ a @ b.dag()
         + g2_d * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b
         + total_detuning * a.dag() @ a)

    # L_b = sqrt(kappa_b) b
    loss_b = jnp.sqrt(KAPPA_B) * b
    # L_a = sqrt(kappa_a) a
    loss_a = jnp.sqrt(KAPPA_A) * a

    return H, [loss_b, loss_a], a, b


def build_kerr_cat(knobs, drift_amp=0.0, drift_delta=0.0):
    a, b = _build_ops(NA, NB)
    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]
    K = knobs[4]

    g2_d = g2 * (1 + drift_amp)

    # H = g2* a†a†b + g2* aab† - eps_d b† - eps_d* b + K a†a†aa
    H = (jnp.conj(g2_d) * a @ a @ b.dag()
         + g2_d * a.dag() @ a.dag() @ b
         - eps_d * b.dag() - jnp.conj(eps_d) * b
         + K * a.dag() @ a.dag() @ a @ a
         + drift_delta * a.dag() @ a)

    # L_b = sqrt(kappa_b) b
    loss_b = jnp.sqrt(KAPPA_B) * b
    # L_a = sqrt(kappa_a) a
    loss_a = jnp.sqrt(KAPPA_A) * a

    return H, [loss_b, loss_a], a, b


def build_effective_cat(knobs, drift_amp=0.0, drift_delta=0.0):
    nb_eff = 1
    a = dq.tensor(dq.destroy(NA), dq.eye(nb_eff))
    eps_2 = knobs[0] + 1j * knobs[1]
    kappa_2 = 1.0

    # H = i eps_2 a†a† - i eps_2* aa
    H = (1j * eps_2 * a.dag() @ a.dag()
         - 1j * jnp.conj(eps_2) * a @ a
         + drift_delta * a.dag() @ a)

    # L_2 = sqrt(kappa_2) aa
    loss_2 = jnp.sqrt(kappa_2) * a @ a
    # L_a = sqrt(kappa_a) a
    loss_a = jnp.sqrt(KAPPA_A) * a

    return H, [loss_2, loss_a], a, None


MODELS = {
    "standard_cat": {
        "label": "Standard Cat",
        "knob_names": ["Re(g₂)", "Im(g₂)", "Re(ε_d)", "Im(ε_d)"],
        "n_knobs": 4,
        "default_mean": [1.0, 0.0, 4.0, 0.0],
        "bounds": [[0.1, 2.0], [-1.0, 1.0], [0.5, 8.0], [-2.0, 2.0]],
        "build": build_standard_cat,
    },
    "moon_cat": {
        "label": "Moon Cat",
        "knob_names": ["Re(g₂)", "Im(g₂)", "Re(ε_d)", "Im(ε_d)", "Re(λ)", "Im(λ)"],
        "n_knobs": 6,
        "default_mean": [1.0, 0.0, 4.0, 0.0, 0.3, 0.0],
        "bounds": [
            [0.1, 2.0], [-1.0, 1.0], [0.5, 8.0], [-2.0, 2.0],
            [-1.0, 1.0], [-1.0, 1.0],
        ],
        "build": build_moon_cat,
    },
    "drift_compensated_cat": {
        "label": "Drift-Compensated Cat",
        "knob_names": ["Re(g₂)", "Im(g₂)", "Re(ε_d)", "Im(ε_d)", "Δ_d"],
        "n_knobs": 5,
        "default_mean": [1.0, 0.0, 4.0, 0.0, 0.0],
        "bounds": [
            [0.1, 2.0], [-1.0, 1.0], [0.5, 8.0], [-2.0, 2.0],
            [-1.0, 1.0],
        ],
        "build": build_drift_cat,
    },
    "kerr_cat": {
        "label": "Kerr Cat",
        "knob_names": ["Re(g₂)", "Im(g₂)", "Re(ε_d)", "Im(ε_d)", "K"],
        "n_knobs": 5,
        "default_mean": [1.0, 0.0, 4.0, 0.0, 0.0],
        "bounds": [
            [0.1, 2.0], [-1.0, 1.0], [0.5, 8.0], [-2.0, 2.0],
            [0.0, 0.5],
        ],
        "build": build_kerr_cat,
    },
    "effective_cat": {
        "label": "Effective Single-Mode Cat",
        "knob_names": ["Re(ε₂)", "Im(ε₂)"],
        "n_knobs": 2,
        "default_mean": [2.0, 0.0],
        "bounds": [[0.5, 5.0], [-2.0, 2.0]],
        "build": build_effective_cat,
    },
}
