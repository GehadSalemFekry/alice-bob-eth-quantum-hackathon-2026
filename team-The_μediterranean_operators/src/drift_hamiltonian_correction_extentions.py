"""Corrected drift Hamiltonian builders with detuning.

These functions mirror `drift_hamiltonians.py` but add a real detuning
term `Delta * a† a` to the storage mode. The knobs convention is:

- knobs[0] = g2_Re
- knobs[1] = g2_Im
- knobs[2] = eps_d_Re
- knobs[3] = eps_d_Im
- knobs[4] = Delta (real detuning, optional; defaults to 0.0)

If only four knobs are provided, the detuning is treated as zero.
"""

import jax.numpy as jnp
import dynamiqs as dq


def _parse_knobs(data):
	knobs = data["knobs"]
	g2 = knobs[0] + 1j * knobs[1]
	eps_d = knobs[2] + 1j * knobs[3]
	Delta = knobs[4] if len(knobs) > 4 else data.get("Delta", 0.0)
	return g2, eps_d, Delta


def amplitude_drift(epoch, amplitude=0.5, frequency=0.01):
	return amplitude * jnp.sin(2 * jnp.pi * frequency * epoch)


def frequency_drift(epoch, amplitude=0.5, frequency=0.01):
	return amplitude * jnp.sin(2 * jnp.pi * frequency * epoch)


def _build_common_ops(data):
	na = data["Hilbert_space_large"]
	nb = data["Hilbert_space_cutted_for_solution"]
	a = dq.tensor(dq.destroy(na), dq.eye(nb))
	b = dq.tensor(dq.eye(na), dq.destroy(nb))
	return a, b


def _static_two_photon_hamiltonian(data):
	g2, eps_d, Delta = _parse_knobs(data)
	a, b = _build_common_ops(data)

	H_static = (
		jnp.conj(g2) * a @ a @ b.dag()
		+ g2 * a.dag() @ a.dag() @ b
		- eps_d * b.dag()
		- jnp.conj(eps_d) * b
		+ Delta * a.dag() @ a
	)
	return H_static, a, b


def build_amplitude_drift(data) -> dq.QArray:
	"""Amplitude drift in the buffer drive with detuning on the storage mode.

	The envelope is `eps_d(t) = A * exp(a*t)` and the Hamiltonian includes
	`Delta * a† a` with `Delta` taken from `knobs[4]` or `data["Delta"]`.
	"""
	g2, _, Delta = _parse_knobs(data)
	na = data["Hilbert_space_large"]
	nb = data["Hilbert_space_cutted_for_solution"]
	knobs = data["knobs"]

	eps_fn = lambda t: knobs[2] * jnp.exp(knobs[3] * t)
	a = dq.tensor(dq.destroy(na), dq.eye(nb))
	b = dq.tensor(dq.eye(na), dq.destroy(nb))

	H_static = (
		jnp.conj(g2) * a @ a @ b.dag()
		+ g2 * a.dag() @ a.dag() @ b
		+ Delta * a.dag() @ a
	)
	H_drive = (
		-dq.modulated(eps_fn, b.dag())
		- dq.modulated(lambda t: jnp.conj(eps_fn(t)), b)
	)

	return dq.constant(H_static) + H_drive


def build_frequency_drift(data) -> dq.QArray:
	"""Frequency drift in the storage resonator with detuning term.

	The drift is `Delta(t) = A * sin(omega * t + phi)`, and the static
	detuning `Delta * a†a` is added from the last knob.
	"""
	g2, eps_d, Delta = _parse_knobs(data)
	a, b = _build_common_ops(data)

	A, omega, phi = data["drift_params"]
	Delta_fn = lambda t: A * jnp.sin(omega * t + phi)

	H_static = (
		jnp.conj(g2) * a @ a @ b.dag()
		+ g2 * a.dag() @ a.dag() @ b
		- eps_d * b.dag()
		- jnp.conj(eps_d) * b
		+ Delta * a.dag() @ a
	)
	H_drift = dq.modulated(Delta_fn, a.dag() @ a)

	return dq.constant(H_static) + H_drift


def build_kerr_drift(data) -> dq.QArray:
	"""Kerr drift in the storage resonator with detuning term."""
	g2, eps_d, Delta = _parse_knobs(data)
	a, b = _build_common_ops(data)

	A, omega, phi = data["drift_params"]
	K_fn = lambda t: A * jnp.sin(omega * t + phi)

	H_static = (
		jnp.conj(g2) * a @ a @ b.dag()
		+ g2 * a.dag() @ a.dag() @ b
		- eps_d * b.dag()
		- jnp.conj(eps_d) * b
		+ Delta * a.dag() @ a
	)
	H_kerr = dq.modulated(K_fn, a.dag() @ a @ a.dag() @ a)

	return dq.constant(H_static) + H_kerr


def build_tls_drift(data) -> dq.QArray:
	"""TLS coupling drift with detuning term on the storage mode."""
	g2, eps_d, Delta = _parse_knobs(data)
	na = data["Hilbert_space_large"]
	nb = data["Hilbert_space_cutted_for_solution"]
	tls_dim = data.get("tls_dim", 2)

	a = dq.tensor(dq.destroy(na), dq.eye(nb), dq.eye(tls_dim))
	b = dq.tensor(dq.eye(na), dq.destroy(nb), dq.eye(tls_dim))
	tls = dq.tensor(dq.eye(na), dq.eye(nb), dq.destroy(tls_dim))

	H_static = (
		jnp.conj(g2) * a @ a @ b.dag()
		+ g2 * a.dag() @ a.dag() @ b
		- eps_d * b.dag()
		- jnp.conj(eps_d) * b
		+ Delta * a.dag() @ a
	)

	A, omega, phi = data.get("drift_params", [0.1, 0.15, 0.0])
	g_tls_fn = lambda t: A * jnp.sin(omega * t + phi)
	sigma_plus = tls.dag()
	sigma_minus = tls

	H_tls = dq.modulated(g_tls_fn, a @ sigma_plus + a.dag() @ sigma_minus)
	return dq.constant(H_static) + H_tls
