import jax.numpy as jnp
import dynamiqs as dq

#==========Functions for Drifting=============
def amplitude_drift(epoch, amplitude=0.5, frequency=0.01):
    return amplitude * jnp.sin(2 * jnp.pi * frequency * epoch)


def frequency_drift(epoch, amplitude=0.5, frequency=0.01):
    return amplitude * jnp.sin(2 * jnp.pi * frequency * epoch)



#==========Apply Drift===========

#THIS FUNCTION ONLY USES EPSILON FUNCTION= A*exp(2*a*t)
def build_amplitude_drift(data) -> dq.QArray:
    """
    Inputs a dictionary data with keys:
    - Hilbert_space_large (int)
    - Hilbert_space_cutted_for_solution (int)
    - knobs (sequence) [g_2_Re, g_2_Im, A,a]
    epsilon drift is eps_d(t) = A * exp(a*t)
    Returns a time-dependent Hamiltonian with amplitude drift.
    """
    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]
    knobs = data["knobs"]

    g2 = knobs[0] + 1j * knobs[1]
    eps_fn = lambda t: knobs[2] * jnp.exp(knobs[3] * t)

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    # H = g2 a†² b + g2* a² b† - ε_d(t) b† - ε_d*(t) b
    H_static = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
    )

    H_drive = (
        -dq.modulated(eps_fn, b.dag())
        -dq.modulated(lambda t: jnp.conj(eps_fn(t)), b)
    )

    return dq.constant(H_static) + H_drive 

#I ONLY MADE FOR A * sin(omega * t + phi)
def build_frequency_drift(data) -> dq.QArray:
    """
    Frequency drift in the storage resonator.

    Inputs:
    - Hilbert_space_large (int)
    - Hilbert_space_cutted_for_solution (int)
    - knobs: [g2_Re, g2_Im, eps_d_Re, eps_d_Im]
    - drift_params: [A, omega, phi]
    drift is Δ(t) = A * sin(omega * t + phi)
    Returns:
    Time-dependent Hamiltonian with storage frequency drift.
    """

    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]
    knobs = data["knobs"]

    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]

    A = data["drift_params"][0]
    omega = data["drift_params"][1]
    phi = data["drift_params"][2]
    Delta_fn = lambda t: A * jnp.sin(omega * t + phi)
    

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    # H = Δ(t) a† a + g2 a†² b + g2* a² b† - ε_d b† - ε_d* b
    H_static = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )

    Delta_fn = lambda t: A

    H_drift = dq.modulated(Delta_fn, a.dag() @ a)

    return dq.constant(H_static) + H_drift

def build_kerr_drift(data) -> dq.QArray:
    """
    Kerr nonlinearity in the storage resonator.

    Inputs (data dict):
    - Hilbert_space_large (int): storage mode Hilbert space dimension (na)
    - Hilbert_space_cutted_for_solution (int): buffer mode dimension (nb)
    - knobs (sequence): [g2_Re, g2_Im, eps_d_Re, eps_d_Im]
      - g2: two-photon coupling strength (complex)
      - eps_d: buffer drive amplitude (complex)
    - drift_params: [A, omega, phi] for Kerr drift K(t) = A * sin(omega * t + phi)
    """

    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]
    knobs = data["knobs"]

    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]

    A = data["drift_params"][0]
    omega = data["drift_params"][1]
    phi = data["drift_params"][2]
    K_fn = lambda t: A * jnp.sin(omega * t + phi)

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    # H = K(t) a†² a² + g2 a†² b + g2* a² b† - eps_d b† - eps_d* b
    H_static = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )

    H_kerr = dq.modulated(K_fn, a.dag() @ a @ a.dag() @ a)

    return dq.constant(H_static) + H_kerr


#NEEDS TESTING, PROBABLY NEEDS TO CHANGE THE JUMP OPERATORS L WITH THIS DRIFT
def build_tls_drift(data) -> dq.QArray:
    """
        Coupling to a TLS resonant with the storage mode.

        Inputs:
        - Hilbert_space_large (int): storage mode dimension `na`.
        - Hilbert_space_cutted_for_solution (int): buffer mode dimension `nb`.
        - knobs (sequence): [g2_Re, g2_Im, eps_d_Re, eps_d_Im].
        - drift_params (optional sequence): [A, omega, phi] for
            g_tls(t) = A * sin(omega * t + phi).

        Output:
        - Time-dependent Hamiltonian with TLS coupling
            g_tls(t) * (a sigma_+ + a^† sigma_-).
    """

    na = data["Hilbert_space_large"]
    nb = data["Hilbert_space_cutted_for_solution"]
    knobs = data["knobs"]

    g2 = knobs[0] + 1j * knobs[1]
    eps_d = knobs[2] + 1j * knobs[3]

    tls_dim = data.get("tls_dim", 2)

    a = dq.tensor(dq.destroy(na), dq.eye(nb), dq.eye(tls_dim))
    b = dq.tensor(dq.eye(na), dq.destroy(nb), dq.eye(tls_dim))
    tls = dq.tensor(dq.eye(na), dq.eye(nb), dq.destroy(tls_dim))

    # H = g2 a†² b + g2* a² b† - eps_d b† - eps_d* b
    H_static = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )

    sigma_plus = tls.dag()
    sigma_minus = tls

    drift_params = data.get("drift_params", [0.1, 0.15, 0.0])
    A = drift_params[0]
    omega = drift_params[1]
    phi = drift_params[2]
    g_tls_fn = lambda t: A * jnp.sin(omega * t + phi)

    H_tls = dq.modulated(g_tls_fn, a @ sigma_plus + a.dag() @ sigma_minus)

    return dq.constant(H_static) + H_tls


def get_drift_func(drift_type):
    if drift_type == "amplitude":
        return lambda ep: [amplitude_drift(ep, 0.5, 0.01)]
    elif drift_type == "frequency":
        return lambda ep: [frequency_drift(ep, 0.5, 0.01)]
    elif drift_type == "both":
        return lambda ep: [amplitude_drift(ep, 0.5, 0.01),
                           frequency_drift(ep, 0.3, 0.008)]
    else:
        return lambda ep: []


FULL_HAMILTONIANS_WITH_DRIFT = {
    "none": {
        "label": "No Drift",
        "knob_labels": [],
        "n_knobs": 0,
        "get_drift": lambda ep: [],
    },
    "amplitude": {
        "label": "Amplitude Drift",
        "knob_labels": ["amp drift"],
        "n_knobs": 1,
        "get_drift": lambda ep: [amplitude_drift(ep, 0.5, 0.01)],
    },
    "frequency": {
        "label": "Frequency Drift",
        "knob_labels": ["freq drift Δ"],
        "n_knobs": 1,
        "get_drift": lambda ep: [frequency_drift(ep, 0.5, 0.01)],
    },
    "both": {
        "label": "Amplitude + Frequency Drift",
        "knob_labels": ["amp drift", "freq drift Δ"],
        "n_knobs": 2,
        "get_drift": lambda ep: [amplitude_drift(ep, 0.5, 0.01),
                                  frequency_drift(ep, 0.3, 0.008)],
    },
}