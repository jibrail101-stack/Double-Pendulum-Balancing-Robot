"""
dynamics.py — Cart + Double Pendulum Dynamics (Milestone 1)

System configuration:
  - Cart (mass M) slides along x-axis, actuated by force u
  - Link 1 (mass m1, length l1): angle θ₁ from upright vertical
  - Link 2 (mass m2, length l2): angle θ₂ from link 1 (relative)

Uniform rod model: COM at lᵢ/2, rotational inertia Iᵢ = mᵢlᵢ²/12

State vector: [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
"""

import numpy as np
import sympy as sp
from functools import lru_cache
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Symbolic EOM derivation (runs once at import, cached)
# ---------------------------------------------------------------------------

def _derive_eom():
    """
    Derive equations of motion symbolically via Lagrangian mechanics.
    Returns lambdified (M_func, rhs_func) where:
      M_func(q, params)        -> 3x3 mass matrix numpy array
      rhs_func(q, qdot, u, params) -> 3-vector numpy array
    such that M @ q̈ = rhs.
    """
    # Symbolic variables
    t = sp.Symbol('t')
    M_s, m1_s, m2_s = sp.symbols('M m1 m2', positive=True)
    l1_s, l2_s, g_s = sp.symbols('l1 l2 g', positive=True)
    u_s = sp.Symbol('u')

    # Generalised coordinates as functions of time
    x   = sp.Function('x')(t)
    th1 = sp.Function('th1')(t)
    th2 = sp.Function('th2')(t)

    q    = [x,   th1,   th2]
    qdot = [x.diff(t), th1.diff(t), th2.diff(t)]

    # Rotational inertia (uniform rod about COM)
    I1 = m1_s * l1_s**2 / 12
    I2 = m2_s * l2_s**2 / 12

    # Absolute angle of link 2 from vertical
    phi2 = th1 + th2

    # COM positions
    x_c1 = x + (l1_s / 2) * sp.sin(th1)
    y_c1 =     (l1_s / 2) * sp.cos(th1)
    x_c2 = x + l1_s * sp.sin(th1) + (l2_s / 2) * sp.sin(phi2)
    y_c2 =     l1_s * sp.cos(th1) + (l2_s / 2) * sp.cos(phi2)

    # COM velocities
    xd_c1 = x_c1.diff(t)
    yd_c1 = y_c1.diff(t)
    xd_c2 = x_c2.diff(t)
    yd_c2 = y_c2.diff(t)

    # Kinetic energy
    T = (sp.Rational(1, 2) * M_s  * qdot[0]**2
       + sp.Rational(1, 2) * m1_s * (xd_c1**2 + yd_c1**2) + sp.Rational(1, 2) * I1 * qdot[1]**2
       + sp.Rational(1, 2) * m2_s * (xd_c2**2 + yd_c2**2) + sp.Rational(1, 2) * I2 * (qdot[1] + qdot[2])**2)

    # Potential energy (y measured upward from cart pivot height)
    V = (m1_s * g_s * y_c1 + m2_s * g_s * y_c2)

    L = T - V

    # Euler-Lagrange: d/dt(∂L/∂q̇ᵢ) - ∂L/∂qᵢ = Qᵢ
    # Generalised forces: Qₓ = u, Qθ₁ = Qθ₂ = 0
    Q = [u_s, sp.Integer(0), sp.Integer(0)]

    eom = []
    for i, (qi, qdi) in enumerate(zip(q, qdot)):
        dL_dqdot = sp.diff(L, qdi)
        ddt_dL_dqdot = dL_dqdot.diff(t)
        dL_dq = sp.diff(L, qi)
        eom.append(sp.simplify(ddt_dL_dqdot - dL_dq - Q[i]))

    # Extract mass matrix M and RHS (everything not involving q̈)
    qddot_syms = [sp.Symbol('xdd'), sp.Symbol('th1dd'), sp.Symbol('th2dd')]

    # Substitute q̈ symbols for the second derivatives
    subs_qdd = {qi.diff(t, 2): qdd for qi, qdd in zip(q, qddot_syms)}
    eom_sub = [e.subs(subs_qdd) for e in eom]

    M_mat = sp.zeros(3, 3)
    rhs_vec = sp.zeros(3, 1)
    for i, eq in enumerate(eom_sub):
        for j, qdd in enumerate(qddot_syms):
            M_mat[i, j] = sp.diff(eq, qdd)
        # rhs = -(eq without q̈ terms)
        rhs_vec[i] = -eq.subs({qdd: 0 for qdd in qddot_syms})

    # Simplify
    M_mat = sp.simplify(M_mat)
    rhs_vec = sp.simplify(rhs_vec)

    # Replace Function(t) with plain symbols for lambdify
    x_s, th1_s, th2_s = sp.symbols('x th1 th2')
    xd_s, th1d_s, th2d_s = sp.symbols('xd th1d th2d')

    sym_map = {
        x: x_s, th1: th1_s, th2: th2_s,
        x.diff(t): xd_s, th1.diff(t): th1d_s, th2.diff(t): th2d_s,
    }

    M_mat_s  = M_mat.subs(sym_map)
    rhs_vec_s = rhs_vec.subs(sym_map)

    # Arguments for lambdify: (state vars, param vars)
    state_args = [x_s, th1_s, th2_s, xd_s, th1d_s, th2d_s]
    param_args = [M_s, m1_s, m2_s, l1_s, l2_s, g_s, u_s]
    all_args   = state_args + param_args

    M_func   = sp.lambdify(all_args, M_mat_s,  modules='numpy')
    rhs_func = sp.lambdify(all_args, rhs_vec_s, modules='numpy')

    return M_func, rhs_func


# Derive once at module load
_M_FUNC, _RHS_FUNC = _derive_eom()


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def load_params(config_path=None):
    """Load parameters from YAML config. Falls back to defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {
        'M':  cfg['cart']['mass'],
        'm1': cfg['link1']['mass'],
        'm2': cfg['link2']['mass'],
        'l1': cfg['link1']['length'],
        'l2': cfg['link2']['length'],
        'g':  cfg['physics']['g'],
        'dt': cfg['integrator']['dt'],
        'b1': cfg.get('damping', {}).get('b1', 0.0),
        'b2': cfg.get('damping', {}).get('b2', 0.0),
    }


def _unpack(params):
    return (params['M'], params['m1'], params['m2'],
            params['l1'], params['l2'], params['g'])


# ---------------------------------------------------------------------------
# Dynamics functions
# ---------------------------------------------------------------------------

def mass_matrix(state, params):
    """Return 3×3 mass matrix M(q) as numpy array."""
    args = list(state) + list(_unpack(params)) + [0.0]  # u not needed for M
    return np.array(_M_FUNC(*args), dtype=float)


def cart_accel_to_force(state, a_des, params):
    """
    Compute the cart force u (N) required to produce a desired cart
    acceleration a_des (m/s²) given the current pendulum state.

    Uses exact inverse dynamics: solves the full 3×3 EOM for u, correctly
    accounting for the coupling reaction forces the pendulum exerts on the cart.

    From  M(q) @ q̈ = f(q, q̇) + [u, 0, 0]ᵀ  with  ẍ = a_des prescribed:
      - Rows 1,2 (no u): solve 2×2 system for [θ̈₁, θ̈₂]
      - Row  0         : u = M[0,:] @ q̈ − f[0]

    Parameters
    ----------
    state : array-like  [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    a_des : float       desired cart acceleration (m/s²)
    params: dict        system parameters

    Returns
    -------
    float  — required force on cart (N)
    """
    state = np.asarray(state, dtype=float)
    M_num = mass_matrix(state, params)

    # f = gravity + Coriolis terms only (u = 0)
    args = list(state) + list(_unpack(params)) + [0.0]
    f = np.array(_RHS_FUNC(*args), dtype=float).flatten()

    # Rows 1,2: M[1:,1:] @ [θ̈₁, θ̈₂] = f[1:] - M[1:,0] * a_des
    theta_ddot = np.linalg.solve(M_num[1:, 1:], f[1:] - M_num[1:, 0] * a_des)

    # Row 0: u = M[0,:] @ q̈ - f[0]
    qddot = np.array([a_des, theta_ddot[0], theta_ddot[1]])
    return float(M_num[0, :] @ qddot - f[0])


def state_dot(t, state, u, params):
    """
    Compute state derivative [ẋ, θ̇₁, θ̇₂, ẍ, θ̈₁, θ̈₂].

    Parameters
    ----------
    t     : float       (time, unused — system is autonomous except for u)
    state : array-like  [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    u     : float       control force on cart (N)
    params: dict        system parameters

    Returns
    -------
    np.ndarray shape (6,)
    """
    state = np.asarray(state, dtype=float)
    M_s, m1, m2, l1, l2, g = _unpack(params)
    args = list(state) + [M_s, m1, m2, l1, l2, g, float(u)]
    M_num   = np.array(_M_FUNC(*args),   dtype=float)
    rhs_num = np.array(_RHS_FUNC(*args), dtype=float).flatten()

    # Optional viscous joint damping (default 0 — backwards compatible)
    b1 = params.get('b1', 0.0)
    b2 = params.get('b2', 0.0)
    if b1 or b2:
        rhs_num[1] -= b1 * state[4]   # −b1·θ̇₁
        rhs_num[2] -= b2 * state[5]   # −b2·θ̇₂

    qddot   = np.linalg.solve(M_num, rhs_num)
    return np.concatenate([state[3:], qddot])


def rk4_step(state, t, dt, u, params):
    """Single RK4 step. Returns next state."""
    k1 = state_dot(t,          state,             u, params)
    k2 = state_dot(t + dt/2,   state + dt/2 * k1, u, params)
    k3 = state_dot(t + dt/2,   state + dt/2 * k2, u, params)
    k4 = state_dot(t + dt,     state + dt   * k3, u, params)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(state0, t_span, dt, control_fn, params):
    """
    Simulate the system over t_span using RK4.

    Parameters
    ----------
    state0     : array-like  initial state [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    t_span     : (t0, tf)    integration interval
    dt         : float       timestep (s)
    control_fn : callable    u = control_fn(t, state) -> float
    params     : dict        system parameters

    Returns
    -------
    times  : np.ndarray shape (N,)
    states : np.ndarray shape (N, 6)
    u_log  : np.ndarray shape (N,)  — control force applied at each step
    """
    t0, tf = t_span
    times  = np.arange(t0, tf + dt, dt)
    states = np.empty((len(times), 6))
    u_log  = np.empty(len(times))
    states[0] = np.asarray(state0, dtype=float)
    u_log[0]  = control_fn(t0, states[0])

    for i in range(1, len(times)):
        t       = times[i - 1]
        u       = control_fn(t, states[i - 1])
        u_log[i - 1] = u
        states[i]    = rk4_step(states[i - 1], t, dt, u, params)

    u_log[-1] = control_fn(times[-1], states[-1])
    return times, states, u_log


def total_energy(state, params):
    """
    Compute total mechanical energy T + V.

    Potential energy is defined with the cart pivot as the reference height
    (y = 0). Upright position has maximum PE.

    Parameters
    ----------
    state  : array-like [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
    params : dict

    Returns
    -------
    float
    """
    state = np.asarray(state, dtype=float)
    x, th1, th2, xd, th1d, th2d = state
    M, m1, m2, l1, l2, g = _unpack(params)
    I1 = m1 * l1**2 / 12
    I2 = m2 * l2**2 / 12

    phi2 = th1 + th2

    # COM velocities (link 1)
    xd_c1 = xd + (l1/2) * np.cos(th1) * th1d
    yd_c1 =    - (l1/2) * np.sin(th1) * th1d

    # COM velocities (link 2)
    xd_c2 = xd + l1 * np.cos(th1) * th1d + (l2/2) * np.cos(phi2) * (th1d + th2d)
    yd_c2 =    - l1 * np.sin(th1) * th1d - (l2/2) * np.sin(phi2) * (th1d + th2d)

    T = (0.5 * M  * xd**2
       + 0.5 * m1 * (xd_c1**2 + yd_c1**2) + 0.5 * I1 * th1d**2
       + 0.5 * m2 * (xd_c2**2 + yd_c2**2) + 0.5 * I2 * (th1d + th2d)**2)

    V = (m1 * g * (l1/2) * np.cos(th1)
       + m2 * g * (l1 * np.cos(th1) + (l2/2) * np.cos(phi2)))

    return T + V
