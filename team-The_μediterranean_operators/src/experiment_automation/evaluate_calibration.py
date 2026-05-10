from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import models
from QHack26.src.experiment_automation.lifetime_observable import measure_lifetime_observable
from QHack26.src.experiment_automation.reward_function import RewardConfig, score_calibration


DEFAULT_MODEL = "standard_cat"


def available_models() -> list[str]:
    return list(models.MODELS.keys())


def _model_config(model_name: str) -> dict:
    if model_name not in models.MODELS:
        raise ValueError(
            f"Unknown model_name={model_name!r}. Available models: {available_models()}"
        )
    return models.MODELS[model_name]


def _validate_knobs(knobs, model_name: str) -> np.ndarray:
    config = _model_config(model_name)
    knobs = np.asarray(knobs, dtype=float)
    expected = int(config["n_knobs"])

    if knobs.shape != (expected,):
        raise ValueError(
            f"{model_name} expects {expected} knobs, got shape {knobs.shape}. "
            f"Knob names: {config['knob_names']}"
        )

    return knobs


def default_knobs(model_name: str = DEFAULT_MODEL) -> np.ndarray:
    return np.asarray(_model_config(model_name)["default_mean"], dtype=float)


def sample_knobs_from_bounds(
    rng: np.random.Generator,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    bounds = np.asarray(_model_config(model_name)["bounds"], dtype=float)
    low = bounds[:, 0]
    high = bounds[:, 1]
    return rng.uniform(low, high)


def build_data_for_state(
    knobs,
    state: str,
    model_name: str = DEFAULT_MODEL,
    drift_amp: float = 0.0,
    drift_delta: float = 0.0,
    tfinal: float = 10.0,
    n_time_points: int = 100,
) -> dict:
    """
    Build the data dictionary used by the lifetime measurement.

    The updated team repo exposes model metadata through models.MODELS. This
    function uses that registry, so new model definitions can be plugged in
    without rewriting our evaluator.
    """

    config = _model_config(model_name)
    knobs = _validate_knobs(knobs, model_name)
    knobs_jnp = jnp.asarray(knobs)

    H, losses, _, b = config["build"](
        knobs_jnp,
        drift_amp=drift_amp,
        drift_delta=drift_delta,
    )

    alpha = models.compute_alpha(model_name, knobs_jnp)

    if model_name == "effective_cat":
        nb = 1
        g_2 = jnp.complex64(1.0 + 0.0j)
        eps_d = jnp.complex64(knobs[0] + 1j * knobs[1])
    else:
        nb = models.NB
        g_2 = jnp.complex64(knobs[0] + 1j * knobs[1])
        eps_d = jnp.complex64(knobs[2] + 1j * knobs[3])

    return {
        "initial_state": state,
        "Hilbert_space_large": models.NA,
        "Hilbert_space_cutted_for_solution": nb,
        "kappa_b": models.KAPPA_B,
        "kappa_a": models.KAPPA_A,
        "eps_d": eps_d,
        "g_2": g_2,
        "alpha": alpha,
        "Hamiltonian": H,
        "Jump_b": losses[0],
        "Jump_a": losses[1],
        "tfinal": tfinal,
        "n_time_points": n_time_points,
    }


def evaluate_calibration(
    knobs,
    model_name: str = DEFAULT_MODEL,
    drift_amp: float = 0.0,
    drift_delta: float = 0.0,
    tfinal: float = 10.0,
    n_time_points: int = 100,
    reward_config: RewardConfig | None = None,
) -> dict:
    """
    Main evaluation function from our side.

    Input:
        knobs: calibration parameters for one model.

    Output:
        dict with Tx, Tz, bias and reward.
    """

    config = _model_config(model_name)

    data_z = build_data_for_state(
        knobs,
        state="+z",
        model_name=model_name,
        drift_amp=drift_amp,
        drift_delta=drift_delta,
        tfinal=tfinal,
        n_time_points=n_time_points,
    )

    data_x = build_data_for_state(
        knobs,
        state="+x",
        model_name=model_name,
        drift_amp=drift_amp,
        drift_delta=drift_delta,
        tfinal=tfinal,
        n_time_points=n_time_points,
    )

    Tz , _ = measure_lifetime_observable(data_z, observable="sz")
    Tx , _ = measure_lifetime_observable(data_x, observable="sx")

    score = score_calibration(Tx=Tx, Tz=Tz, config=reward_config)

    return {
        "knobs": np.asarray(knobs, dtype=float),
        "model_name": model_name,
        "model_label": config["label"],
        "knob_names": config["knob_names"],
        "drift_amp": float(drift_amp),
        "drift_delta": float(drift_delta),
        **score,
    }
