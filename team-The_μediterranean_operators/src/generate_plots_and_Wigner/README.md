# Generate Plots and Wigner GIFs

This folder contains scripts to generate expectation value decay plots and animated Wigner function GIFs for various quantum Hamiltonian models.

## Scripts

Each script corresponds to a different Hamiltonian type:

1. **generate_plots_and_Wigner_standard.py** - Standard cat state Hamiltonian
2. **generate_plots_and_Wigner_Kerr_static.py** - Static Kerr nonlinearity
3. **generate_plots_and_Wigner_moon_cat.py** - Moon cat state (enhanced Kerr)
4. **generate_plots_and_Wigner_amplitude_drift.py** - Time-dependent amplitude drift
5. **generate_plots_and_Wigner_frequency_drift.py** - Storage frequency drift
6. **generate_plots_and_Wigner_Kerr_drift.py** - Time-dependent Kerr drift
7. **generate_plots_and_Wigner_TLS_drift.py** - TLS coupling drift

## Output Structure

For each Hamiltonian type, plots and GIFs are generated with filenames encoding key parameters:

### Expectation Value Plots (PNG)
```
Plots/{model}_{initial_state}_{parameters}_T{tfinal}.png
```

**Example:**
- `Plots/standard_cat_+z_g2=1.0_eps=4.0_T3.0.png`
- `Plots/frequency_drift_+x_A=0.3_T2.5.png`

### Wigner Function Evolution (GIF)
```
Wigner/{model}_{initial_state}_{parameters}_T{tfinal}.gif
```

**Example:**
- `Wigner/moon_cat_+z_lambda=0.2_T3.0.gif`
- `Wigner/kerr_drift_+x_A=0.2_T2.0.gif`

## Filename Convention

- **model**: Type of Hamiltonian (standard_cat, kerr_static, moon_cat, etc.)
- **initial_state**: Quantum state basis (+x, +z)
- **parameters**: Key parameters varied (e.g., g2=1.0, A=0.3)
- **tfinal**: Final evolution time

## Running the Generators

### Individual Script
```bash
python generate_plots_and_Wigner_standard.py
```

### All Generators at Once
```bash
python run_all_generators.py
```

## Output Location

All generated files are saved to the main `../Figures/` directory:
- PNG plots: `../Figures/Plots/`
- GIFs: `../Figures/Wigner/`

## Parameter Sweeps

Each script performs a parameter sweep to show how dynamics evolve with varying system parameters:

- Standard cat: sweeps `g2` and `eps_d`
- Kerr static: sweeps `kerr` coefficient
- Moon cat: sweeps `lambda` coupling
- Drifts: sweep amplitude or coupling strength

For each parameter combination, both +x and +z initial states are simulated.
