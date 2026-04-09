"""
controller_lqr.py — LQR Stabilizing Controller (Milestone 3)

Linearizes the cart-double pendulum about the upright equilibrium
  x_eq = [x=0, θ₁=0, θ₂=0, ẋ=0, θ̇₁=0, θ̇₂=0]
using numerical finite differences, then solves the continuous-time
algebraic Riccati equation (CARE) for an optimal feedback gain K.

Control law:  u(t) = −K @ state

Usage
-----
    from dynamics import load_params, simulate
    from controller_lqr import LQRController

    params = load_params()
    ctrl = LQRController(params)
    ctrl.print_info()                       # eigenvalues, gains

    state0 = [0.0, 0.15, 0.10, 0.0, 0.0, 0.0]   # ≈ ±9° perturbation
    times, states, u_log = simulate(
        state0, (0.0, 30.0), params['dt'],
        ctrl.control_fn, params,
    )

    # or just:
    from controller_lqr import demo
    demo()
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from dynamics import state_dot, load_params, total_energy


class LQRController:
    """
    Linear-Quadratic Regulator for the cart-double pendulum upright equilibrium.

    Design procedure
    ----------------
    1. Numerically linearise state_dot() around x_eq = 0 via forward
       finite differences → A (6×6), B (6×1).
    2. Solve continuous CARE:
           Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0
    3. Compute gain:  K = R⁻¹ Bᵀ P   (shape 1×6)
    4. Control law:   u = −K @ state

    The equilibrium is x_eq = [0, 0, 0, 0, 0, 0] (fully upright, at rest).
    Cart position x is left unpenalized in Q — the cart is free to translate
    while balancing the links.

    Parameters
    ----------
    params : dict         system parameters (from load_params())
    Q      : (6,6) ndarray or None   state cost matrix; defaults to tuned values
    R      : (1,1) ndarray or None   control cost matrix
    eps    : float        finite-difference step for Jacobian (default 1e-6)
    """

    # Default gains — validated to meet FR-05 (±5° for 30 s from ±10° perturbation)
    # Penalise angles strongly; leave cart x unpenalized (free translation).
    _Q_DEFAULT = np.diag([0.0, 100.0, 100.0, 1.0, 10.0, 10.0])
    _R_DEFAULT = np.array([[0.1]])

    def __init__(self, params: dict, Q=None, R=None, eps: float = 1e-6):
        self.params = params
        self.Q   = np.asarray(Q,   dtype=float) if Q is not None else self._Q_DEFAULT.copy()
        self.R   = np.asarray(R,   dtype=float) if R is not None else self._R_DEFAULT.copy()
        self._eps = eps

        self.A, self.B = self._linearise()
        self.K, self.P = self._solve_lqr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def control_fn(self, t: float, state) -> float:
        """
        Compute LQR control force u = −K @ state.

        Compatible with dynamics.simulate() control_fn signature.

        Parameters
        ----------
        t     : float       (unused — controller is time-invariant)
        state : array-like  [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]

        Returns
        -------
        float — cart force u (N)
        """
        return float(-self.K.flatten() @ np.asarray(state, dtype=float))

    def print_info(self) -> None:
        """Print linearisation matrices, gain vector K, and closed-loop eigenvalues."""
        eig_ol = np.linalg.eigvals(self.A)
        eig_cl = np.linalg.eigvals(self.A - self.B @ self.K)
        # Cart position (x) is unpenalized → one eigenvalue is exactly 0.
        # This is marginal (not unstable) — x drifts to a constant, not infinity.
        # Treat Re(λ) ≤ 1e-8 as acceptable.
        stable  = bool(np.all(eig_cl.real <= 1e-8))

        print()
        print("=" * 60)
        print("  LQR CONTROLLER — UPRIGHT STABILISER")
        print("=" * 60)

        print(f"\n  Linearisation  (eps = {self._eps})")
        print(f"  Equilibrium    x_eq = [0, 0, 0, 0, 0, 0]")

        print(f"\n  Open-loop eigenvalues of A:")
        for ev in sorted(eig_ol, key=lambda e: -abs(e.real)):
            print(f"    {ev.real:>+10.5f}  {ev.imag:>+10.5f}j")

        print(f"\n  Gain vector K  (u = −K @ state):")
        labels = ['x ', 'θ₁', 'θ₂', 'ẋ ', 'θ̇₁', 'θ̇₂']
        for lbl, k in zip(labels, self.K.flatten()):
            print(f"    K[{lbl}] = {k:>+12.5f}")

        print(f"\n  Closed-loop eigenvalues of (A − BK):")
        for ev in sorted(eig_cl, key=lambda e: e.real):
            print(f"    {ev.real:>+10.5f}  {ev.imag:>+10.5f}j")

        print(f"\n  Angles asymptotically stable:  {'YES ✓' if stable else 'NO  ✗'}")
        print(f"  (Cart x is marginally stable — drifts to a constant, not infinity)")
        print("=" * 60)
        print()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _linearise(self):
        """
        Forward finite-difference Jacobian of state_dot() at x_eq = 0.

        Returns
        -------
        A : np.ndarray (6, 6)   ∂f/∂state  |_eq
        B : np.ndarray (6, 1)   ∂f/∂u      |_eq
        """
        eq  = np.zeros(6)
        f0  = np.array(state_dot(0.0, eq, u=0.0, params=self.params), dtype=float)
        eps = self._eps

        A = np.zeros((6, 6))
        for j in range(6):
            s_p = eq.copy()
            s_p[j] += eps
            A[:, j] = (np.array(state_dot(0.0, s_p, u=0.0, params=self.params), dtype=float) - f0) / eps

        B = np.zeros((6, 1))
        B[:, 0] = (np.array(state_dot(0.0, eq, u=eps, params=self.params), dtype=float) - f0) / eps

        return A, B

    def _solve_lqr(self):
        """
        Solve the continuous-time algebraic Riccati equation:
            Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0

        Returns
        -------
        K : np.ndarray (1, 6)   optimal gain matrix
        P : np.ndarray (6, 6)   cost-to-go matrix
        """
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.solve(self.R, self.B.T @ P)   # R⁻¹ Bᵀ P
        return K, P


# ---------------------------------------------------------------------------
# Swing-up + LQR combined controller
# ---------------------------------------------------------------------------

class SwingUpLQRController:
    """
    Two-phase controller: energy-pumping swing-up → LQR stabilization.

    All parameters default to None and are derived automatically from the
    physical system (masses, lengths, gravity) when not supplied.  This means
    the controller self-configures correctly after any change to config.yaml.

    Phase 1 — swing-up (triggered by trigger())
    ---------------------------------------------
    Short initial kick breaks the hanging equilibrium, then an Åström-Furuta
    pendulum-only energy pump drives both links toward the upright:

        u = k_sw · ẋ · (E_target − E_pend)

    where E_pend = KE_links + PE (cart KE excluded).

    Auto-derived swing-up parameters:
      u_max    = 4 · (M + m1 + m2) · g
      kick_dur = 0.08 · T_nat   (T_nat = 2π·√(2·l1/3g), clamped 0.05–0.5 s)
      k_sw     = 4 · u_max / (√(2·g·l1) · E_target)

    Phase 2 — LQR capture
    ----------------------
    Switches to the LQR when both |θ₁| and |θ₂| are within capture_deg AND
    the predicted LQR force is ≤ capture_force_factor · u_max.  The force gate
    prevents premature capture at high-velocity passes (common for non-standard
    geometries) that would otherwise cause immediate divergence.

    Auto-derived LQR weights (Bryson's rule + open-loop eigenvalue scaling):
      Q[θ]  = 1 / (5°)²
      Q[θ̇] = 1 / (λ_max · 5° · 0.5)²   λ_max = most unstable open-loop eigenvalue
      R     = Q[θ] / 2000

    Geometry warning
    ----------------
    If l2/l1 < 0.2 the second link is too short/light for the energy pump to
    control reliably.  The LQR stabiliser (small perturbations) remains valid;
    only the large-angle swing-up may fail.

    Usage
    -----
        ctrl = SwingUpLQRController(params)   # all params auto-derived
        ctrl.trigger()                        # start swing-up (or press G)
        ctrl.reset()                          # return to idle

        u = ctrl.control_fn(t, state)   # compatible with simulate()
    """

    def __init__(self, params: dict,
                 k_sw: float = None,
                 u_max: float = None,
                 capture_deg: float = 30.0,
                 kick_frac: float = 0.4,
                 kick_dur: float = None,
                 Q_capture=None,
                 R_capture=None,
                 capture_force_factor: float = 3.0):
        # Use undamped params for both energy computation and LQR design.
        # Adding physical joint damping changes the linearised A matrix and
        # inflates K[θ₂] to >1500 N/rad, causing immediate LQR divergence.
        self.params = {**params, 'b1': 0.0, 'b2': 0.0}
        p = self.params
        _M, _m1, _m2, _l1, _l2, _g = (p['M'], p['m1'], p['m2'],
                                        p['l1'], p['l2'], p['g'])

        # ── Geometry warning ──────────────────────────────────────────
        if _l2 / _l1 < 0.2:
            print(f"[SwingUpLQR] WARNING: l2/l1 = {_l2/_l1:.3f} < 0.2")
            print( "[SwingUpLQR]   The second link is very short relative to l1.")
            print( "[SwingUpLQR]   The energy pump may not converge for this geometry.")
            print( "[SwingUpLQR]   LQR stabiliser remains valid for small perturbations.")
            print( "[SwingUpLQR]   Run tune.py for simulation-validated parameters.")

        # ── Auto-derive swing-up parameters ───────────────────────────
        _m_total  = _M + _m1 + _m2
        _E_target = _m1*_g*(_l1/2) + _m2*_g*(_l1 + _l2/2)   # upright PE
        _T_nat    = 2.0 * np.pi * np.sqrt(2.0 * _l1 / (3.0 * _g))
        _v_char   = np.sqrt(2.0 * _g * _l1)

        if u_max is None:
            u_max = 4.0 * _m_total * _g
        if kick_dur is None:
            kick_dur = float(np.clip(0.08 * _T_nat, 0.05, 0.5))
        if k_sw is None:
            # Pump saturates at u_max when ẋ = v_char and ΔE = E_target/4
            k_sw = 4.0 * u_max / (_v_char * _E_target)

        # ── Auto-derive LQR Q/R (Bryson's rule + eigenvalue scaling) ─
        if Q_capture is None or R_capture is None:
            _lqr_probe  = LQRController(self.params)
            _lambda_max = float(np.max(np.linalg.eigvals(_lqr_probe.A).real))
            _th_acc     = np.radians(5.0)
            _Q_th       = 1.0 / _th_acc ** 2
            _Q_thd      = 1.0 / (_lambda_max * _th_acc * 0.5) ** 2
            if Q_capture is None:
                Q_capture = np.diag([0.0, 4*_Q_th, 4*_Q_th, 1.0,
                                     4*_Q_thd, 4*_Q_thd])
            if R_capture is None:
                R_capture = np.array([[_Q_th / 2000.0]])

        self.k_sw       = k_sw
        self.u_max      = u_max
        self._cap_rad   = np.radians(capture_deg)
        self._kick_frac = kick_frac
        self._kick_dur  = kick_dur
        self._phase     = 'idle'   # 'idle' | 'swingup' | 'lqr'
        self._t0        = None     # time when trigger() was called

        # Force gate: only switch to LQR when |u_lqr(state)| ≤ limit.
        # Prevents divergence when the pendulum passes through the angle
        # zone at high velocity requiring forces far beyond u_max.
        self._cap_force_limit = capture_force_factor * u_max

        self.lqr = LQRController(self.params,
                                  Q=np.asarray(Q_capture, dtype=float),
                                  R=np.asarray(R_capture, dtype=float))

        # E_target stored for _pend_energy pump
        self.E_target = _E_target

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------

    def _pend_energy(self, state) -> float:
        """Mechanical energy of links only (excludes cart KE)."""
        p = self.params
        m1, m2, l1, l2, g = p['m1'], p['m2'], p['l1'], p['l2'], p['g']
        x, th1, th2, xd, td1, td2 = state
        v1x = xd + (l1/2)*(-np.sin(th1))*td1
        v1y =       (l1/2)* np.cos(th1) *td1
        v2x = xd + l1*(-np.sin(th1))*td1 + (l2/2)*(-np.sin(th1+th2))*(td1+td2)
        v2y =       l1* np.cos(th1) *td1 + (l2/2)* np.cos(th1+th2) *(td1+td2)
        I1  = m1*l1**2/12
        I2  = m2*l2**2/12
        KE1 = 0.5*m1*(v1x**2 + v1y**2) + 0.5*I1*td1**2
        KE2 = 0.5*m2*(v2x**2 + v2y**2) + 0.5*I2*(td1+td2)**2
        PE  = m1*g*(l1/2)*np.cos(th1) + m2*g*(l1*np.cos(th1) + (l2/2)*np.cos(th1+th2))
        return float(KE1 + KE2 + PE)

    def control_fn(self, t: float, state) -> float:
        """u = swing-up force or LQR force depending on phase."""
        state = np.asarray(state, dtype=float)

        if self._phase == 'idle':
            return 0.0

        # Record trigger time on first real call
        if self._t0 is None:
            self._t0 = t

        th1 = float(np.arctan2(np.sin(state[1]), np.cos(state[1])))  # wrap to (−π, π]
        th2 = float(np.arctan2(np.sin(state[2]), np.cos(state[2])))

        # Switch to LQR when both links are within the capture angle AND the
        # predicted LQR force is within the force limit (prevents capture at
        # high-velocity passes that would immediately diverge).
        if abs(th1) < self._cap_rad and abs(th2) < self._cap_rad:
            if abs(self.lqr.control_fn(t, state)) <= self._cap_force_limit:
                self._phase = 'lqr'

        if self._phase == 'lqr':
            return self.lqr.control_fn(t, state)

        # ── Swing-up ──────────────────────────────────────────────────
        t_elapsed = t - self._t0

        # Short initial kick to break the perfectly stable hanging equilibrium.
        # Without this, u=k·ẋ·(E_up−E) is zero because ẋ=0 at rest.
        if t_elapsed < self._kick_dur:
            return self.u_max * self._kick_frac

        # Pendulum-only energy pump (Åström-Furuta variant):
        # u = k · ẋ · (E_pend_target − E_pend)
        # Adding energy when cart motion can do positive work on the links.
        E  = self._pend_energy(state)
        xd = float(state[3])
        u  = self.k_sw * xd * (self.E_target - E)
        return float(np.clip(u, -self.u_max, self.u_max))

    def __call__(self, t: float, state) -> float:
        return self.control_fn(t, state)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        return self._phase

    def trigger(self) -> None:
        """Start the swing-up sequence (called when G is pressed)."""
        if self._phase == 'idle':
            self._phase = 'swingup'
            self._t0    = None   # set on first control_fn call

    def reset(self) -> None:
        """Return to idle — call this whenever the simulation resets."""
        self._phase = 'idle'
        self._t0    = None


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def demo() -> None:
    """
    Interactive swing-up + LQR stabilization demo.

    Starts with the pendulum hanging at rest. Press G to trigger the
    swing-up maneuver — the cart moves to pump energy into the links
    until both reach the upright position, then LQR takes over to balance.

    Controls
    --------
    G          — trigger swing-up and stabilization
    R / R-click — reset to hanging at rest
    Space      — pause / resume
    """
    from visualizer import CartPendulumVisualizer

    params = load_params()
    ctrl   = SwingUpLQRController(params)

    print("[demo] Pendulum hanging at rest.")
    print("[demo] Press G to swing up and stabilize.")
    print("[demo] Press R to reset.")

    # Tiny initial perturbation: 0.5° off exact hanging.
    # Prevents u=k·ẋ·(E−E_up)=0 at the very first step (exact equilibrium).
    # The initial kick in control_fn() handles any remaining energy.
    state0 = np.array([0.0, np.pi - 0.01, 0.0, 0.0, 0.0, 0.0])
    viz = CartPendulumVisualizer(params)
    viz.run_interactive(state0=state0, swingup_ctrl=ctrl)


if __name__ == '__main__':
    demo()
