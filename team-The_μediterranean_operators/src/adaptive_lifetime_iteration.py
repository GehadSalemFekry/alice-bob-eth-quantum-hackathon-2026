
def measure_lifetime_adaptive(
    data,
    n_points=100,
    slope_tol=1e-3,
    tail_points=12,
    max_iter=8,
    tfinal_scale=1.5,
):
    
    """
    Iteratively increases tfinal until the fitted tail is flat.

    Flatness criterion (must satisfy both):
    1) |d/dt y_fit(t_end)| < slope_tol
    2) |tail linear slope| < slope_tol

    Returns:
        dict with lifetime estimate and diagnostic information.
    """
    current_tfinal = float(data["tfinal"])
    base_dt = current_tfinal / (n_points - 1)

    history = []

    for i in range(max_iter):
        current_n_points = int(round(current_tfinal / base_dt)) + 1

        x, y = _simulate_sz(data, tfinal=current_tfinal, n_points=current_n_points)
        fit = robust_exp_fit(x, y)

        y_fit = fit["y_fit"]
        popt = fit["popt"]

        end_slope = float(jnp.abs(model_derivative(popt, x[-1])))

        tail_n = min(tail_points, len(x))
        tail_slope = float(jnp.abs(_tail_linear_slope(x[-tail_n:], y_fit[-tail_n:])))

        is_flat = end_slope < slope_tol and tail_slope < slope_tol

        history.append(
            {
                "iteration": i + 1,
                "tfinal": float(current_tfinal),
                "n_points": int(current_n_points),
                "tau": float(popt[1]),
                "end_slope": end_slope,
                "tail_slope": tail_slope,
                "flat": bool(is_flat),
            }
        )

        if is_flat:
            return {
                "tau": float(popt[1]),
                "fit_params": [float(v) for v in popt],
                "x": x,
                "y": y,
                "y_fit": y_fit,
                "chosen_tfinal": float(current_tfinal),
                "chosen_n_points": int(current_n_points),
                "history": history,
                "converged": True,
            }

        current_tfinal *= tfinal_scale

    # If no flat tail was found, return the last fit and diagnostics.
    return {
        "tau": float(popt[1]),
        "fit_params": [float(v) for v in popt],
        "x": x,
        "y": y,
        "y_fit": y_fit,
        "chosen_tfinal": float(current_tfinal / tfinal_scale),
        "chosen_n_points": int(current_n_points),
        "history": history,
        "converged": False,
    }