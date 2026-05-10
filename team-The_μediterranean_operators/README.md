# ETH Quantum Hackathon 2026 - Adaptive Calibration of Cat Qubits

## Team: The_μediterranean_operators

- David Ulloa Cañizares - ETH Zurich
- Iason Kazazis - ETH Zurich
- Gehad Ahmed - Friedrich Schiller University Jena
- Thanos Loukanaris - PSL University

## Overview

The challenge consisted in designing and benchmarking a robust optimization algorithm for cat qubit stabilization. This robustness was tested under different hardware drift, noise, and optimization models. To be more precise, we pursued simultaneously the following objectives:

- Obtaining a high target bias: $\eta = T_z/T_x$
- Maximizing the lifetimes $T_z$ and $T_x$

$T_z$ represents how long a system stays in a state, and $T_x$ how long a superposition remains coherent.

To do this, we built an adaptive calibration framework that automatically tunes system parameters to:

- The above mentioned properties
- Improve stability under drift
- Make parameter search more efficient

The overall goal is to combine physics-informed modeling with numerical optimization to support scalable quantum control.

## Methodology

We frame calibration as an iterative optimization loop:

1. Simulate the quantum system using physical models.
2. Evaluate performance with a reward/objective function.
3. Update control parameters with an optimizer.
4. Repeat until the system converges to a strong operating regime.

This allows us to discover stable operating points automatically instead of relying on manual tuning.

## Models, Drifts, and Objectives

We swept a combinatorial grid of **5 models × 4 drift types × 5 penalty functions = 100 configurations**, each optimized with CMA-ES over 30 epochs, and compiled into an interactive dashboard.

### 5 Models (`src/models.py`)

| Key | Label | Knobs |
|-----|-------|-------|
| `standard_cat` | Standard Cat | Re(g₂), Im(g₂), Re(ε_d), Im(ε_d) |
| `moon_cat` | Moon Cat | + Re(λ), Im(λ) |
| `drift_compensated_cat` | Drift-Compensated Cat | + Δ_d |
| `kerr_cat` | Kerr Cat | + K |
| `effective_cat` | Effective Single-Mode Cat | Re(ε₂), Im(ε₂) |

### 4 Drift Types (`src/drift_hamiltonians.py`)

- **none** – No drift
- **amplitude** – Amplitude drift on g₂
- **frequency** – Storage frequency drift Δ(t)
- **both** – Combined amplitude + frequency drift

### 5 Penalty Functions (`src/objectives.py`)

| Key | Label |
|-----|-------|
| `log_sum_penalty` | R = ln(T_Z) + ln(T_X) − λ·(η − η₀)² |
| `weighted_log` | R = w_Z·ln(T_Z) + w_X·ln(T_X) − λ·(η − η₀)² |
| `bias_prioritized` | R = ln(T_Z) − γ·max(\|η − η₀\| − ε, 0) |
| `bias_constrained` | R = ln(T_Z) if \|η − η₀\| < ε else −∞ |
| `log_relative_penalty` | R = ln(T_Z) + ln(T_X) − λ·(ln(η) − ln(η₀))² |

## Interactive Dashboard

**[https://gehadsalemfekry.github.io/cat-calibration-dashboard/](https://gehadsalemfekry.github.io/cat-calibration-dashboard/)**

Explore all 100 optimization traces interactively: switch between models, drift types, and objective functions; view T_Z, T_X, η, and reward convergence over epochs.

## Results and Insights

- Improved calibration performance:
    - Found more stable parameter configurations
    - Increased robustness against drift
- Drift-aware optimization:
    - Drift-aware models outperformed static approaches
    - Adaptive updates improved long-term stability
- Efficiency gains:
    - Structured optimization performed better than naive parameter sweeps

## Project Structure

- `src/` — Core physics models, Hamiltonians, lifetime extraction, objectives, optimizers, and utilities
  - `src/generate_plots_and_Wigner/` — Scripts for expectation-value decay plots and Wigner function GIFs
  - `src/experiment_automation/` — Reward, lifetime, and calibration evaluation modules
- `optimization.ipynb` — Main CMA-ES optimization loop and analysis
- `notebook.ipynb` — Supplementary exploration notebook
- `Presentation Slides.pptx` — Presentation Slides

## Tech Stack

- Python
- NumPy / JAX
- Dynamiqs (quantum simulation)
- CMA-ES (SepCMA from `cmaes`)
- Custom optimization routines

## Challenges

- Balancing accuracy and computational cost
- Modeling realistic drift dynamics
- Designing a meaningful reward function
- Keeping optimization stable across parameter regimes

## Future Work

- Integrate more advanced or different optimizers
- Improve drift modeling and real-time adaptation
- Extend to multi-qubit systems
- Connect to experimental calibration pipelines

## Conclusion

This project shows that adaptive optimization can significantly improve quantum system calibration, especially in the presence of drift and uncertainty.

These methods are important for making quantum hardware more reliable and scalable.

