"""Postprocess fitting results: fit exponential decay and validate."""

import jax.numpy as jnp
from scipy.optimize import least_squares


def model(p, t):
    """Exponential decay model: y = A * exp(-t/tau) + C."""
    A, tau, C = p
    return A * jnp.exp(-t/tau) + C


def residuals(p, x, y):
    """Residuals between model and data."""
    return model(p, x) - y


def robust_exp_fit(x, y):
    """Fit exponential decay with robust loss function."""
    A0 = y.max() - y.min()
    C0 = y.min()
    tau0 = (x.max() - x.min())
    p0 = [A0, tau0, C0]

    res = least_squares(
        residuals,
        p0,
        args=(x, y),
        bounds=([0, 0, -jnp.inf], [jnp.inf, jnp.inf, jnp.inf]),
        loss="soft_l1",
        f_scale=0.1
    )

    A, tau, C = res.x
    y_fit = model(res.x, x)
    residual_norm = jnp.sum((residuals(res.x, x, y)) ** 2)

    return {
        "popt": res.x,
        "y_fit": y_fit,
        "residual_norm": residual_norm,
    }


def postprocess_halflife(x, y, data):
    """
    Fit exponential decay and validate fit quality.
    
    If fit error exceeds threshold, raise error and plot.
    Otherwise return the halflife (tau).
    """
    initial_state = data["initial_state"]
    fit = robust_exp_fit(x, y)
    y_fit = fit["y_fit"]
    residual_norm = fit["residual_norm"]
    Halflife = fit["popt"][1]

    # Check fit quality: residual norm relative to data range
    data_range = y.max() - y.min()
    relative_error = residual_norm / (data_range ** 2 + 1e-10)
    error_threshold = 0.1  # 10% relative error tolerance

    if relative_error > error_threshold:
        # Plot fit if quality is poor
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label='Data')
        plt.plot(x, y_fit, '-', label='Fit')
        plt.xlabel('Time')
        if initial_state in ["+z", "-z"]:
            plt.ylabel('<Sz>')
        elif initial_state in ["+x", "-x"]:
            plt.ylabel('<Sx>')
        plt.title(f'Poor Fit Quality (error={relative_error:.3f})')
        plt.legend()
        plt.grid()
        plt.show()

        raise ValueError(
            f"Fit quality too poor: relative error = {relative_error:.4f} "
            f"(threshold = {error_threshold}). Try adjusting tfinal or knobs."
        )

    if data.get("plot", False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o', label='Data')
        plt.plot(x, y_fit, '-', label='Fit')
        plt.xlabel('Time')
        if initial_state in ["+z", "-z"]:
            plt.ylabel('<Sz>')
        elif initial_state in ["+x", "-x"]:
            plt.ylabel('<Sx>')
        plt.title(f'Estimated Lifetime: {Halflife:.2f} units')
        plt.legend()
        plt.grid()
        plt.show()

    return Halflife
