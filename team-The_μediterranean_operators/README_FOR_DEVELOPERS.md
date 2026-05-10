

#Hamiltonians

##Static Hamiltonians

In this seciton we will discribe how to use static Hamiltonians
Do:
import static_hamiltonians as statH

then with statH.build (it will pop up the build options), use any

data_build_H = {
    "Hilbert_space_large": 30,
    "Hilbert_space_cutted_for_solution": 10,
    "knobs": [1, 0, 4, 0],
}
H = statH.build_standard_cat(data_build_H)
data_halflife = {
    "initial_state": "+x",
    "Hamiltonian": H,
    "kappa_a": 1.0,
    "kappa_b": 10.0,
    "tfinal": 10.0,
    "plot": True}
data = {**data_halflife, **data_build_H}
halflife , _ = measure_lifetime(data)

##Dynamic Hamiltonians

Again, import drift_hamiltonians as driftH

then with driftH.build (it will pop up the build options), use any

data_build_H = {
    "Hilbert_space_large": 30,
    "Hilbert_space_cutted_for_solution": 10,
    "knobs": [1, 0, 4, 0, 1, 0],
    "f": lambda t: jnp.sin(2.0 * jnp.pi * t),
}
H = driftH.build_amplitude_drift(data_build_H)
data_halflife = {
    "initial_state": "+z",
    "Hamiltonian": H,
    "kappa_a": 1.0,
    "kappa_b": 8.0,
    "tfinal": 10.0,
    "plot": True}
data = {**data_halflife, **data_build_H}
halflife , _ = measure_lifetime(data)


At this point Look at IasonassPlayground file to see how it works (dont look 5)
