"""
tests/test_dynamics.py — Unit tests for dynamics.py (Milestone 1)

Validates EOM against known analytical benchmarks:
  1. Upright equilibrium: zero acceleration at θ₁=θ₂=0
  2. Hanging equilibrium: zero acceleration at θ₁=π, θ₂=0
  3. Energy conservation: RK4 drift < 0.1% over 5 s free swing
  4. Single pendulum period: simulated period within 1% of 2π√(l/g)
  5. Mass matrix is symmetric positive definite
  6. Cart x-axis forces during LQR stabilization (tabulated + convergence check)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics import state_dot, mass_matrix, simulate, total_energy, load_params


# ---------------------------------------------------------------------------
# Default parameters for tests — loaded from config.yaml
# ---------------------------------------------------------------------------

PARAMS = load_params()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_upright_equilibrium():
    """At θ₁=θ₂=0, ṡtate=0, u=0: all accelerations must be zero."""
    state = np.zeros(6)
    sd = state_dot(0.0, state, u=0.0, params=PARAMS)
    qddot = sd[3:]  # [ẍ, θ̈₁, θ̈₂]
    assert np.allclose(qddot, 0.0, atol=1e-10), (
        f"Upright equilibrium: expected zero accelerations, got {qddot}"
    )


def test_hanging_equilibrium():
    """At θ₁=π, θ₂=0 (fully hanging), ṡtate=0, u=0: accelerations must be zero."""
    state = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
    sd = state_dot(0.0, state, u=0.0, params=PARAMS)
    qddot = sd[3:]
    assert np.allclose(qddot, 0.0, atol=1e-10), (
        f"Hanging equilibrium: expected zero accelerations, got {qddot}"
    )


def test_energy_conservation():
    """
    Free swing from a moderate initial angle (u=0) over 5 s.
    RK4 energy drift must stay below 0.1% of initial energy.
    """
    state0 = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    t_span = (0.0, 5.0)
    dt = 0.005

    times, states, _ = simulate(state0, t_span, dt, control_fn=lambda t, s: 0.0, params=PARAMS)

    E0 = total_energy(states[0], PARAMS)
    E_final = total_energy(states[-1], PARAMS)

    rel_drift = abs(E_final - E0) / abs(E0)
    assert rel_drift < 0.001, (
        f"Energy conservation: relative drift = {rel_drift:.6f} (limit 0.001)"
    )


def test_single_pendulum_period():
    """
    With a very heavy cart (M >> 0) and negligible link 2 (m2→0),
    the system reduces to a single uniform-rod pendulum hanging from the cart.

    Analytical period for a uniform rod swinging from one end: T = 2π√(2l/(3g)).
    Simulated period must match within 1%.
    """
    heavy_cart_params = {**PARAMS, 'M': 1e6, 'm2': 1e-9, 'l2': 1e-9}

    l1 = heavy_cart_params['l1']
    g  = heavy_cart_params['g']
    # Uniform rod (COM at l/2, I = ml²/12): effective length = 2l/3
    T_analytical = 2 * np.pi * np.sqrt(2 * l1 / (3 * g))  # ~1.16 s for l1=0.5

    # Small displacement from hanging equilibrium (θ₁ = π)
    eps = 0.05  # 0.05 rad ≈ 2.9° — well within small-angle regime
    state0 = np.array([0.0, np.pi + eps, 0.0, 0.0, 0.0, 0.0])

    # Simulate for ~4 full periods
    t_end = 4.5 * T_analytical
    times, states, _ = simulate(
        state0, (0.0, t_end), dt=0.001,
        control_fn=lambda t, s: 0.0,
        params=heavy_cart_params,
    )

    # Detect period via same-direction zero-crossings of (θ₁ - π).
    # Consecutive positive→negative crossings are exactly one full period apart.
    th1 = states[:, 1] - np.pi  # deviation from π
    crossings = []
    for i in range(1, len(th1)):
        if th1[i - 1] > 0 and th1[i] <= 0:
            frac = th1[i - 1] / (th1[i - 1] - th1[i])
            crossings.append(times[i - 1] + frac * (times[i] - times[i - 1]))

    assert len(crossings) >= 2, (
        f"Could not detect two same-direction zero-crossings (got {len(crossings)}). "
        "Check simulation length or initial conditions."
    )

    # Consecutive same-direction crossings are already one full period apart
    T_measured = np.mean(np.diff(crossings))

    rel_error = abs(T_measured - T_analytical) / T_analytical
    assert rel_error < 0.01, (
        f"Period: measured={T_measured:.4f} s, analytical={T_analytical:.4f} s, "
        f"relative error={rel_error:.4f} (limit 0.01)"
    )


def test_cart_force_stabilization():
    """
    LQR upright-balance: cart (M) x-axis force profile during stabilization.

    An LQR controller is derived by numerically linearizing around the upright
    equilibrium [x=0, θ₁=0, θ₂=0, ẋ=0, θ̇₁=0, θ̇₂=0].  Starting from a small
    angular perturbation (θ₁=0.1 rad, θ₂=0.05 rad) the controller drives both
    links back to vertical while the forces u(t) applied to the cart are logged.

    Note: l2 is overridden to 0.5 m (config holds a placeholder value of 200 m).

    Asserts:
      - Non-negligible initial force |u(t=0)| > 0.1 N  (controller is active)
      - |θ₁|, |θ₂|, |θ̇₁|, |θ̇₂| < 0.01  at t = 10 s  (system stabilized)
    """
    from scipy.linalg import solve_continuous_are

    # Override the config placeholder: l2 = 200 m → physically sensible 0.5 m
    params = {**PARAMS, 'l2': 0.5}

    # ------------------------------------------------------------------
    # 1. Numerical linearization around upright equilibrium (u = 0)
    # ------------------------------------------------------------------
    eq = np.zeros(6)
    eps_fd = 1e-6
    f0 = state_dot(0.0, eq, u=0.0, params=params)

    A = np.zeros((6, 6))
    for j in range(6):
        s = eq.copy()
        s[j] += eps_fd
        A[:, j] = (state_dot(0.0, s, u=0.0, params=params) - f0) / eps_fd

    B = np.zeros((6, 1))
    B[:, 0] = (state_dot(0.0, eq, u=eps_fd, params=params) - f0) / eps_fd

    # ------------------------------------------------------------------
    # 2. LQR design  (continuous-time Riccati)
    # ------------------------------------------------------------------
    # Penalise angular deviations and rates; x-position left unpenalised
    # (cart is free to translate while balancing).
    Q = np.diag([0.0, 100.0, 100.0, 1.0, 10.0, 10.0])
    R = np.array([[0.1]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)          # shape (1, 6)

    # ------------------------------------------------------------------
    # 3. Simulate with LQR controller from small perturbation
    # ------------------------------------------------------------------
    state0 = np.array([0.0, 0.1, 0.05, 0.0, 0.0, 0.0])
    t_span = (0.0, 10.0)
    dt     = params['dt']

    force_log: list[tuple[float, float]] = []

    def lqr_control(t, state):
        u = float(-np.dot(K.flatten(), state))
        force_log.append((t, u))
        return u

    times, states, u_log = simulate(state0, t_span, dt, lqr_control, params)

    # ------------------------------------------------------------------
    # 4. Print tabulated forces at key time points
    # ------------------------------------------------------------------
    sample_times = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    print("\nCart x-axis force u(t) during LQR stabilization:")
    print(f"  {'t (s)':>7}  {'u (N)':>12}  {'θ₁ (rad)':>12}  {'θ₂ (rad)':>12}")
    print("  " + "-" * 50)
    n_log = len(force_log)
    for t_s in sample_times:
        s_idx = min(int(round(t_s / dt)), len(times) - 1)
        f_idx = min(int(round(t_s / dt)), n_log - 1)
        u_val  = force_log[f_idx][1]
        th1_val = states[s_idx, 1]
        th2_val = states[s_idx, 2]
        print(f"  {t_s:>7.1f}  {u_val:>12.4f}  {th1_val:>12.6f}  {th2_val:>12.6f}")

    # ------------------------------------------------------------------
    # 5. Assertions
    # ------------------------------------------------------------------
    u0 = float(-np.dot(K.flatten(), state0))
    assert abs(u0) > 0.1, (
        f"Expected non-negligible initial force; got u(0) = {u0:.4f} N"
    )

    fin = states[-1]
    assert abs(fin[1]) < 0.01, f"θ₁ not stabilized: {fin[1]:.6f} rad at t=10 s"
    assert abs(fin[2]) < 0.01, f"θ₂ not stabilized: {fin[2]:.6f} rad at t=10 s"
    assert abs(fin[4]) < 0.01, f"θ̇₁ not stabilized: {fin[4]:.6f} rad/s at t=10 s"
    assert abs(fin[5]) < 0.01, f"θ̇₂ not stabilized: {fin[5]:.6f} rad/s at t=10 s"


def test_mass_matrix_symmetric_pd():
    """Mass matrix M(q) must be symmetric and positive definite for any configuration."""
    test_states = [
        np.zeros(6),
        np.array([0.0, 0.5, -0.3, 0.0, 0.0, 0.0]),
        np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.2, -0.8, 0.5, -0.3, 0.1]),
    ]

    for state in test_states:
        M = mass_matrix(state, PARAMS)

        assert np.allclose(M, M.T, atol=1e-12), (
            f"Mass matrix not symmetric at state={state}\nM=\n{M}"
        )

        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0), (
            f"Mass matrix not positive definite at state={state}\n"
            f"Eigenvalues: {eigvals}"
        )
