import os, time

# ── JAX performance config (set before importing jax) ──
_ncpu = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import jax.numpy as jnp
from cmaes import SepCMA
from models import MODELS, NA, NB, compute_alpha
from Model_to_halflife import measure_lifetime
from drift_hamiltonians import FULL_HAMILTONIANS_WITH_DRIFT
from objectives import OBJECTIVES

OBJECTIVE = "log_sum_penalty"
OBJECTIVE_KWARGS = {}


def cat_reward(knobs, model_name, target_bias=100.0, lambda_penalty=0.001,
               drift_amp=0.0, drift_delta=0.0,
               objective=OBJECTIVE, objective_kwargs=None):
    model = MODELS[model_name]
    H, [loss_b, loss_a], _, _ = model["build"](knobs, drift_amp, drift_delta)

    nb = 1 if model_name == "effective_cat" else NB
    alpha = compute_alpha(model_name, knobs)

    def get_lifetime(state, tfinal):
        data = {
            "initial_state": state,
            "Hilbert_space_large": NA,
            "Hilbert_space_cutted_for_solution": nb,
            "tfinal": tfinal,
            "Hamiltonian": H,
            "Jump_a": loss_a,
            "Jump_b": loss_b,
            "alpha": alpha,
            "raise_on_bad_fit": False,
        }
        try:
            val, _ = measure_lifetime(data)
            return val
        except ValueError:
            return 0.0

    T_Z = get_lifetime("+z", 200.0)
    T_X = get_lifetime("+x", 1.0)

    if T_Z == 0.0 or T_X == 0.0:
        return -1e6, 0.0, 0.0, 0.0

    eta = T_Z / max(T_X, 1e-6)
    kwargs = objective_kwargs if objective_kwargs else {}
    reward = OBJECTIVES[objective]["fn"](
        T_Z, T_X, target_bias, lambda_penalty, **kwargs
    )

    return float(reward), float(T_Z), float(T_X), float(eta)


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
    drift_cfg = FULL_HAMILTONIANS_WITH_DRIFT[drift_type]
    n_drift = drift_cfg["n_knobs"]
    model = MODELS[model_name]
    n_knobs = model["n_knobs"]
    nb = 1 if model_name == "effective_cat" else NB

    mean0 = jnp.array(model["default_mean"])
    sigma0 = 0.3
    bounds = jnp.array(model["bounds"])

    optimizer = SepCMA(
        mean=mean0, sigma=sigma0, bounds=bounds,
        population_size=batch_size, seed=seed,
    )

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

        xs = [optimizer.ask() for _ in range(optimizer.population_size)]

        rewards = []
        with ThreadPoolExecutor(max_workers=_ncpu) as pool:
            futures = {
                pool.submit(
                    cat_reward, x, model_name, target_bias, lambda_penalty,
                    drift_amp=drift_amp, drift_delta=drift_delta,
                    objective=objective, objective_kwargs=objective_kwargs,
                ): x for x in xs
            }
            for future in as_completed(futures):
                x = futures[future]
                r, tz, tx, eta = future.result()
                rewards.append((x, r, tz, tx, eta))

        optimizer.tell([(x, -r) for x, r, _, _, _ in rewards])

        mean_r = float(np.mean([r for _, r, _, _, _ in rewards]))

        _, tz, tx, eta = cat_reward(
            jnp.array(optimizer.mean), model_name, target_bias, lambda_penalty,
            drift_amp=drift_amp, drift_delta=drift_delta,
            objective=objective, objective_kwargs=objective_kwargs,
        )
        tz = float(tz)
        tx = float(tx)
        eta = float(eta)

        history["reward"].append(mean_r)
        history["mean_params"].append(list(np.array(optimizer.mean)))
        history["tz"].append(tz)
        history["tx"].append(tx)
        history["eta"].append(eta)
        history["drift_truth"].append(drift_vals)

        log_interval = max(1, n_epochs // 10)
        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch:3d}/{n_epochs} | reward={mean_r:.3f} | "
                f"T_Z={tz:.1f} | T_X={tx:.3f} | η={eta:.0f} | "
                f"[{elapsed:.0f}s]"
            )

    print(f"\nDone. Final: T_Z={tz:.1f}, T_X={tx:.3f}, η={eta:.0f}")
    print(f"Params: {[f'{v:.4f}' for v in optimizer.mean]}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="standard_cat")
    parser.add_argument("--drift", choices=list(FULL_HAMILTONIANS_WITH_DRIFT.keys()),
                        default="none")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=6)
    args = parser.parse_args()

    print(f"Model: {MODELS[args.model]['label']}")
    print(f"Drift: {FULL_HAMILTONIANS_WITH_DRIFT[args.drift]['label']}")
    print(f"Objective: {OBJECTIVES[OBJECTIVE]['label']}")
    run_optimization(
        model_name=args.model,
        drift_type=args.drift,
        n_epochs=args.epochs,
        batch_size=args.batch,
        objective=OBJECTIVE,
        objective_kwargs=OBJECTIVE_KWARGS,
    )
