import jax.numpy as jnp
import dynamiqs as dq
import Wigner_function_plotting as Wfp
from time import time
from postprocess import postprocess_halflife

def measure_lifetime(data: dict) -> float:
    """
    data dict entries:
    - initial_state (str) Have +x or +z to see the Tx and Tz
    - Hilbert_space_large (int)
    - Hilbert_space_cutted_for_solution (int)

    - knobs (sequence) [g_2_Re, g_2_Im, eps_d_R, eps_d_Im] (Now it is 4 but it can be 6)

    - kappa_a (float)
    - kappa_b (float)
    - alpha (float) [auto-estimated]
    - Jump_a (dq.Operator) [None]
    - Jump_b (dq.Operator) [None]

    - Hamiltonian (dq.Operator) [None]
    - tfinal (float)

    - SNR_addition (float) [0.0] - for testing robustness of fit with added noise
    
    - plot (bool) [False] - diagnostic plotting only; use plot_expectation_value.py for publication plots
    """

    initial_state = data["initial_state"]
    na = data.get("Hilbert_space_large",30)
    nb = data.get("Hilbert_space_cutted_for_solution",15)
    a = dq.tensor(dq.destroy(na), dq.eye(nb)) # annihilaiton operator
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

    tsave = jnp.linspace(0, data["tfinal"], 100)

    g_state = dq.coherent(na, alpha_estimate)
    e_state = dq.coherent(na, -alpha_estimate)

    basis = {
        "+z": g_state,
        "-z": e_state,
        "+x": (g_state + e_state) / jnp.sqrt(2),
        "-x": (g_state - e_state) / jnp.sqrt(2),
        "+y": (g_state + 1j*e_state) / jnp.sqrt(2),
        "-y": (g_state - 1j*e_state) / jnp.sqrt(2),
    }

    
    # This construction of sigmaz will not work without a good estimate of alpha, which is hard to come by in experiment.
    if initial_state in ["+z", "-z"]:
        sz = (basis["+z"] @ basis["+z"].dag() - basis["-z"] @ basis["-z"].dag()) # normalization to keep it in the same scale as a and b
        sz = dq.tensor(sz, dq.eye(nb))
        psi0 = dq.tensor(basis[initial_state], dq.fock(nb,0)) # initial state
        res = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi0,
            tsave,
            options=dq.Options(progress_meter=False),
            exp_ops=[sz],
            method=data.get("mesolve_method", dq.method.Tsit5())
        )
    elif initial_state in ["+x", "-x"]:
        x = (1j * jnp.pi * a.dag() @ a).expm() # PARITY OPERATOR. VERY IMPORTANT
        psi0 = dq.tensor(basis[initial_state], dq.fock(nb,0)) # initial state
        res = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi0,
            tsave,
            options=dq.Options(progress_meter=False),
            exp_ops=[x],
            method=data.get("mesolve_method", dq.method.Tsit5())
        )

    sNt = res.expects[0,:].real
    ts = res.tsave

    y = sNt 
    x = ts 
    SNR_addition = data.get("SNR_addition", 0.0)
    if SNR_addition > 0.0:
        y = y + SNR_addition

    if data.get("plotWigner", False):
        Wfp.plot_wigner(data,res)

    Halflife = postprocess_halflife(x, y, data)

    # Adaptive tfinal: adjust if lifetime is poorly measured
    tfinal = data["tfinal"]

    if Halflife < tfinal / 10:
        # Lifetime too short relative to tfinal; reduce it
        data["tfinal"] = tfinal / 8
    elif Halflife > tfinal:
        # Lifetime too long relative to tfinal; increase it
        data["tfinal"] = tfinal * 2

    return Halflife, data


