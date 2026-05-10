import jax.numpy as jnp


# R = ln(T_Z) + ln(T_X) - λ * (η - η₀)²
def log_sum_penalty(T_Z, T_X, target_bias, lambda_penalty, **_):
    eta = T_Z / jnp.maximum(T_X, 1e-6)
    return float(jnp.log(T_Z) + jnp.log(T_X) - lambda_penalty * (eta - target_bias) ** 2)


# R = w_Z * ln(T_Z) + w_X * ln(T_X) - λ * (η - η₀)²
def weighted_log(T_Z, T_X, target_bias, lambda_penalty, w_Z=1.0, w_X=100.0, **_):
    eta = T_Z / jnp.maximum(T_X, 1e-6)
    return float(w_Z * jnp.log(T_Z) + w_X * jnp.log(T_X) - lambda_penalty * (eta - target_bias) ** 2)


# R = ln(T_Z) - γ * max(|η - η₀| - ε, 0)
def bias_prioritized(T_Z, T_X, target_bias, lambda_penalty=0.01, tolerance=10.0, gamma=1.0, **_):
    eta = T_Z / jnp.maximum(T_X, 1e-6)
    bias_penalty = gamma * jnp.maximum(jnp.abs(eta - target_bias) - tolerance, 0)
    return float(jnp.log(T_Z) - bias_penalty)


# R = ln(T_Z)  if |η - η₀| < ε,  else  -∞
def bias_constrained(T_Z, T_X, target_bias, lambda_penalty=0.01, tolerance=10.0, **_):
    eta = T_Z / jnp.maximum(T_X, 1e-6)
    return float(jnp.log(T_Z)) if abs(float(eta - target_bias)) < tolerance else -1e6


# R = ln(T_Z) + ln(T_X) - λ * (ln(η) - ln(η₀))²
def log_relative_penalty(T_Z, T_X, target_bias, lambda_penalty=0.001, **_):
    eta = T_Z / jnp.maximum(T_X, 1e-6)
    return float(
        jnp.log(T_Z) + jnp.log(T_X)
        - lambda_penalty * (jnp.log(eta) - jnp.log(target_bias)) ** 2
    )


OBJECTIVES = {
    "log_sum_penalty": {
        "label": "Log sum + quadratic bias penalty",
        "fn": log_sum_penalty,
        "eq": "R = ln(T_Z) + ln(T_X) − λ·(η − η₀)²",
    },
    "weighted_log": {
        "label": "Weighted log sum + quadratic bias penalty",
        "fn": weighted_log,
        "eq": "R = w_Z·ln(T_Z) + w_X·ln(T_X) − λ·(η − η₀)²",
    },
    "bias_prioritized": {
        "label": "Bias-prioritized (tolerance + linear penalty)",
        "fn": bias_prioritized,
        "eq": "R = ln(T_Z) − γ·max(|η − η₀| − ε, 0)",
    },
    "bias_constrained": {
        "label": "Bias hard constraint",
        "fn": bias_constrained,
        "eq": "R = ln(T_Z)  if |η − η₀| < ε  else  −∞",
    },
    "log_relative_penalty": {
        "label": "Log sum + relative bias penalty",
        "fn": log_relative_penalty,
        "eq": "R = ln(T_Z) + ln(T_X) − λ·(ln(η) − ln(η₀))²",
    },
}
