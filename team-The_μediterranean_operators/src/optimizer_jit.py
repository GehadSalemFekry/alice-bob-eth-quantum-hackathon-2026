import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import dynamiqs as dq
from jax import jit, vmap
from cmaes import SepCMA
from models import MODELS, NA, NB, KAPPA_B, KAPPA_A, compute_alpha
from Model_to_halflife import jax_exp_fit
from drift_scenarios import DRIFT_CONFIGS
from objectives import OBJECTIVES

OBJECTIVE = "log_sum_penalty"
OBJECTIVE_KWARGS = {}


def _make_jit_measure(model_name):
    """Create a JIT-compilable function that returns (T_Z, T_X, eta) for one sample."""
    model = MODELS[model_name]
    build_fn = model["build"]
    nb = 1 if model_name == "effective_cat" else NB

    @jit
    def measure(knobs, drift_amp, drift_delta):
        H, losses, a_out, _ = build_fn(knobs, drift_amp, drift_delta)
        alpha = compute_alpha(model_name, knobs)

        g_state = dq.coherent(NA, alpha)
        e_state = dq.coherent(NA, -alpha)

        sz_pure = g_state @ g_state.dag() - e_state @ e_state.dag()
        sz = dq.tensor(sz_pure, dq.eye(nb))

        sx = (1j * jnp.pi * a_out.dag() @ a_out).expm()

        b_ground = dq.fock(nb, 0)

        # --- T_Z: +z initial state, measure sz ---
        psi0_z = dq.tensor(g_state, b_ground)
        ts_z = jnp.linspace(0, 200.0, 100)
        res_z = dq.mesolve(H, losses, psi0_z, ts_z,
                           options=dq.Options(progress_meter=False),
                           exp_ops=[sz])
        T_Z = jax_exp_fit(ts_z, res_z.expects[0].real)

        # --- T_X: +x initial state, measure parity (sx) ---
        x_state = (g_state + e_state) / jnp.sqrt(2)
        psi0_x = dq.tensor(x_state, b_ground)
        ts_x = jnp.linspace(0, 1.0, 100)
        res_x = dq.mesolve(H, losses, psi0_x, ts_x,
                           options=dq.Options(progress_meter=False),
                           exp_ops=[sx])
        T_X = jax_exp_fit(ts_x, res_x.expects[0].real)

        eta = T_Z / jnp.maximum(T_X, 1e-6)
        return jnp.array([T_Z, T_X, eta])

    return jit(vmap(measure, in_axes=(0, None, None)))


def run_optimization(
    model_name="standard_cat",
    drift_type="none",
    target_bias=100.0,
    lambda_penalty=0.001,
    batch_size=12,
    n_epochs=80,
    seed=0,
    objective=OBJECTIVE,
    objective_kwargs=None,
):
    drift_cfg = DRIFT_CONFIGS[drift_type]
    n_drift = drift_cfg["n_knobs"]
    model = MODELS[model_name]
    n_knobs = model["n_knobs"]

    mean0 = jnp.array(model["default_mean"])
    sigma0 = 0.3
    bounds = jnp.array(model["bounds"])

    optimizer = SepCMA(
        mean=mean0, sigma=sigma0, bounds=bounds,
        population_size=batch_size, seed=seed,
    )

    # Precompile the JIT-batched measurement function
    batched_measure = _make_jit_measure(model_name)

    # Warm-up: run one batch to trigger JIT compilation
    print("  Compiling JIT graph...")
    warmup_xs = jnp.tile(mean0, (batch_size, 1))
    _ = batched_measure(warmup_xs, 0.0, 0.0)
    print("  Done.\n")

    objective_fn = OBJECTIVES[objective]["fn"]
    obj_kwargs = objective_kwargs or {}

    history = {
        "model_name": model_name,
        "drift_type": drift_type,
        "target_bias": target_bias,
        "objective": objective,
        "reward": [],
        "mean_params": [],
        "tz": [],
        "tx": [],
        "eta": [],
        "drift_truth": [],
        "drift_labels": drift_cfg["knob_labels"],
    }

    t_start = time.time()

    for epoch in range(n_epochs):
        drift_vals = drift_cfg["get_drift"](epoch)
        drift_amp = drift_vals[0] if n_drift > 0 else 0.0
        drift_delta = drift_vals[1] if n_drift > 1 else 0.0

        xs = jnp.array([optimizer.ask() for _ in range(optimizer.population_size)])

        # Batched JIT call — single mesolve per initial state, vectorized
        results = batched_measure(xs, drift_amp, drift_delta)
        tzs = results[:, 0]
        txs = results[:, 1]
        etas = results[:, 2]

        # Compute rewards outside JIT (trivial arithmetic)
        rewards_list = []
        for j in range(len(xs)):
            r = objective_fn(tzs[j], txs[j], target_bias, lambda_penalty, **obj_kwargs)
            rewards_list.append((xs[j], float(r), float(tzs[j]), float(txs[j]), float(etas[j])))

        optimizer.tell([(x, -r) for x, r, _, _, _ in rewards_list])

        mean_r = float(jnp.mean(jnp.array([r for _, r, _, _, _ in rewards_list])))

        # Evaluate mean params
        mean_results = batched_measure(jnp.expand_dims(jnp.array(optimizer.mean), 0), drift_amp, drift_delta)[0]
        tz_m, tx_m, eta_m = float(mean_results[0]), float(mean_results[1]), float(mean_results[2])
        r_m = float(objective_fn(mean_results[0], mean_results[1], target_bias, lambda_penalty, **obj_kwargs))

        history["reward"].append(r_m)
        history["mean_params"].append(list(optimizer.mean))
        history["tz"].append(tz_m)
        history["tx"].append(tx_m)
        history["eta"].append(eta_m)
        history["drift_truth"].append(drift_vals)

        log_interval = max(1, n_epochs // 10)
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch:3d}/{n_epochs} | reward={r_m:.3f} | "
                f"T_Z={tz_m:.1f} | T_X={tx_m:.3f} | η={eta_m:.0f} | "
                f"[{elapsed:.0f}s]"
            )

    print(f"\nDone. Final: T_Z={tz_m:.1f}, T_X={tx_m:.3f}, η={eta_m:.0f}")
    print(f"Params: {[f'{v:.4f}' for v in optimizer.mean]}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="standard_cat")
    parser.add_argument("--drift", choices=list(DRIFT_CONFIGS.keys()),
                        default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()

    print(f"Model: {MODELS[args.model]['label']}")
    print(f"Drift: {DRIFT_CONFIGS[args.drift]['label']}")
    print(f"Objective: {OBJECTIVES[OBJECTIVE]['label']}")
    run_optimization(
        model_name=args.model,
        drift_type=args.drift,
        n_epochs=args.epochs,
        batch_size=args.batch,
        objective=OBJECTIVE,
        objective_kwargs=OBJECTIVE_KWARGS,
    )
