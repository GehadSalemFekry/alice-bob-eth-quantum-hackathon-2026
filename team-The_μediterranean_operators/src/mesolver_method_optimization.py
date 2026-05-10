import jax.numpy as jnp
import dynamiqs as dq
from time import time
from Model_to_halflife import measure_lifetime

def mesosolver_method_opt(data: dict) -> dict:
    #Iterates through all methods and comptues the time.
    methods = [
        dq.method.Tsit5(),        # deterministic ODE solver (adaptive Runge–Kutta)
        dq.method.Dopri5(),       # deterministic ODE solver (adaptive Runge–Kutta)
        dq.method.Dopri8(),       # deterministic ODE solver (higher-order adaptive Runge–Kutta)

        dq.method.Kvaerno3(),     # deterministic stiff ODE solver
        dq.method.Kvaerno5(),     # deterministic stiff ODE solver

        dq.method.Rouchon1(0.1),     # stochastic (quantum filtering / stochastic master equation)
        dq.method.Rouchon2(0.1),     # stochastic (higher-order stochastic integration)
        dq.method.Rouchon3(0.1),     # stochastic (higher-order stochastic integration)
    ]
    #Following methdos are not used
        # "Expm",               # deterministic (exact matrix exponential evolution)

        # "Event",              # event handling (not a solver; detects discontinuities / triggers)

        # "JumpMonteCarlo",     # stochastic (quantum jump Monte Carlo trajectories)
        # "DiffusiveMonteCarlo" # stochastic (diffusive quantum trajectories / SME sampling)
    #EULER HAS BEEN SKIPPED

    time_per_method = {}
    baseline_time = None
    for method in methods:
        data["mesolve_method"] = method
        try:
            t0 = time()
            res , _ = measure_lifetime(data)
            elapsed = time() - t0
        except Exception as exc:
            time_per_method[method] = None
            print(f"Method: {method}, failed with error: {exc}")
            continue

        if baseline_time is None:
            baseline_time = elapsed
            time_per_method[method] = elapsed
            print(f"Method: {method}, Time: {elapsed:.4f} seconds")
            continue

        if elapsed > 3 * baseline_time:
            time_per_method[method] = None
            print(
                f"Method: {method}, Time: {elapsed:.4f} seconds (too slow, stopping)"
            )
            break

        time_per_method[method] = elapsed
        print(f"Method: {method}, Time: {elapsed:.4f} seconds")
    return time_per_method