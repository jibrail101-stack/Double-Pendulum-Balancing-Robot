"""
tune.py — Physics-based auto-tuner for SwingUpLQRController

Derives all controller parameters analytically from the physical system
(masses, link lengths, gravity) so the controller works correctly after
any change to config.yaml.  If the initial parameters don't produce a
stable swing-up, the tuner automatically backs off k_sw until they do.

Usage
-----
    python tune.py                        # tune + validate current config.yaml
    python tune.py --config my.yaml       # use a different config file
    python tune.py --no-adaptive          # skip adaptive k_sw refinement
    python tune.py --theta-acc 3          # target 3° steady-state accuracy

From Python
-----------
    from tune import tune_all, build_controller
    from dynamics import load_params

    params = load_params()
    ctrl   = build_controller(params)     # drop-in for SwingUpLQRController(params)
    ctrl.trigger()

Theory — analytical parameter derivation
-----------------------------------------
u_max    = 4 · (M + m1 + m2) · g
             → enough to accelerate total mass at ~4g equivalent

kick_dur = 0.08 · T_nat   (clamped to [0.05, 0.5] s)
           T_nat = 2π·√(2·l1 / 3·g)   (uniform rod, small oscillation)
             → kicks for ~8% of one natural period

k_sw     = 4 · u_max / (v_char · E_target)
           v_char   = √(2·g·l1)       (characteristic cart speed)
           E_target = PE at perfect upright
             → pump saturates at u_max when ẋ = v_char and ΔE = E_target / 4

LQR (Bryson's rule + eigenvalue scaling):
  Q[θ]   = 1 / θ_acc²
  Q[θ̇]  = 1 / (λ_max · θ_acc · 0.5)²   λ_max = largest open-loop eigenvalue
  R_stab  = Q[θ] / 1000
  Q_cap   = 4 · Q_stab  (more aggressive capture phase)
  R_cap   = Q[θ] / 2000

Force-gate capture condition:
  LQR takes over only when |u_lqr(state)| ≤ 3 · u_max.
  Prevents catastrophic divergence when the pendulum passes through the
  capture angle zone at high angular velocity (common for short l2).

Adaptive k_sw:
  If the first simulation diverges or never stabilises, k_sw is reduced by
  40% on each retry (up to 4 times).  Reports which value succeeded.
"""

import argparse
import numpy as np

from dynamics import load_params, simulate
from controller_lqr import LQRController, SwingUpLQRController

# Factor applied to u_max for the LQR force-gate (3× gives a safe margin
# for the original geometry while blocking dangerous captures elsewhere).
_CAPTURE_FORCE_FACTOR = 3.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unpack(params):
    return (params['M'], params['m1'], params['m2'],
            params['l1'], params['l2'], params['g'])


# ---------------------------------------------------------------------------
# Public API — analytical tuning
# ---------------------------------------------------------------------------

def compute_scales(params: dict) -> dict:
    """
    Compute characteristic physical scales that drive all tuning formulas.

    Returns
    -------
    dict with keys:
      m_total    — total system mass (kg)
      F_grav     — gravitational force on total mass (N)
      E_target   — upright potential energy / swing-up target (J)
      T_nat      — natural period of link 1 as a free-swinging rod (s)
      v_char     — characteristic cart velocity (m/s)
      lambda_max — largest real open-loop eigenvalue (1/s)
      A, B       — linearised system matrices at upright
    """
    M, m1, m2, l1, l2, g = _unpack(params)
    m_total = M + m1 + m2

    E_target = m1 * g * (l1 / 2) + m2 * g * (l1 + l2 / 2)

    # Uniform rod pivoted at top end, small-oscillation period
    T_nat = 2.0 * np.pi * np.sqrt(2.0 * l1 / (3.0 * g))

    # Characteristic cart speed: if all upright PE converted to cart KE
    v_char = np.sqrt(2.0 * g * l1)

    # Linearise at upright (undamped) to get open-loop eigenvalues
    undamped = {**params, 'b1': 0.0, 'b2': 0.0}
    lqr_probe = LQRController(undamped)
    eigs = np.linalg.eigvals(lqr_probe.A)
    lambda_max = float(np.max(eigs.real))

    return dict(
        m_total=m_total,
        F_grav=m_total * g,
        E_target=E_target,
        T_nat=T_nat,
        v_char=v_char,
        lambda_max=lambda_max,
        A=lqr_probe.A,
        B=lqr_probe.B,
    )


def tune_swingup(params: dict, scales: dict = None) -> dict:
    """
    Derive swing-up parameters analytically.

    Returns
    -------
    dict: k_sw, u_max, kick_frac, kick_dur
    """
    if scales is None:
        scales = compute_scales(params)

    F_grav   = scales['F_grav']
    E_target = scales['E_target']
    T_nat    = scales['T_nat']
    v_char   = scales['v_char']

    u_max     = 4.0 * F_grav
    kick_frac = 0.4
    kick_dur  = float(np.clip(0.08 * T_nat, 0.05, 0.5))

    # Pump saturates at u_max when ẋ = v_char and ΔE = E_target/4
    k_sw = 4.0 * u_max / (v_char * E_target)

    return dict(k_sw=k_sw, u_max=u_max, kick_frac=kick_frac, kick_dur=kick_dur)


def tune_lqr(params: dict, scales: dict = None,
             theta_acc_deg: float = 5.0) -> tuple:
    """
    Derive Q/R matrices using Bryson's rule scaled by open-loop eigenvalues.

    Returns
    -------
    Q_stab, R_stab, Q_capture, R_capture   — all as numpy arrays
    """
    if scales is None:
        scales = compute_scales(params)

    lambda_max = scales['lambda_max']
    theta_acc  = np.radians(theta_acc_deg)

    Q_theta       = 1.0 / theta_acc ** 2
    theta_dot_max = lambda_max * theta_acc * 0.5
    Q_thetadot    = 1.0 / theta_dot_max ** 2

    Q_stab = np.diag([0.0, Q_theta, Q_theta, 0.0, Q_thetadot, Q_thetadot])
    R_stab = np.array([[Q_theta / 1000.0]])

    Q_cap = np.diag([0.0, 4*Q_theta, 4*Q_theta, 1.0, 4*Q_thetadot, 4*Q_thetadot])
    R_cap = np.array([[Q_theta / 2000.0]])

    return Q_stab, R_stab, Q_cap, R_cap


# ---------------------------------------------------------------------------
# Build controller
# ---------------------------------------------------------------------------

def build_controller(params: dict, tuned: dict = None) -> SwingUpLQRController:
    """
    Construct a SwingUpLQRController with auto-tuned parameters.

    Always uses the force-gate capture condition (factor = 3.0) so LQR only
    takes over when the initial control force is manageable.

    Parameters
    ----------
    params : system parameters (from load_params())
    tuned  : result of tune_all(); computed automatically if None

    Returns
    -------
    SwingUpLQRController — call .trigger() to start the maneuver
    """
    if tuned is None:
        tuned = tune_all(params)
    return SwingUpLQRController(
        params,
        k_sw                = tuned['k_sw'],
        u_max               = tuned['u_max'],
        kick_frac           = tuned['kick_frac'],
        kick_dur            = tuned['kick_dur'],
        Q_capture           = tuned['Q_cap'],
        R_capture           = tuned['R_cap'],
        capture_force_factor = _CAPTURE_FORCE_FACTOR,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(params: dict, tuned: dict, t_sim: float = 60.0,
             verbose: bool = True) -> dict:
    """
    Run a swing-up simulation and report FR-04 / FR-05 metrics.

    Parameters
    ----------
    params   : system parameters
    tuned    : result of tune_all()
    t_sim    : simulation duration (s)
    verbose  : print results to stdout

    Returns
    -------
    dict with keys:
      diverged, captured, capture_time_s,
      fr04_pass, fr05_pass,
      ss_theta1_deg, ss_theta2_deg, peak_force_N
    """
    ctrl = build_controller(params, tuned)
    ctrl.trigger()

    state0 = np.array([0.0, np.pi - 0.01, 0.0, 0.0, 0.0, 0.0])
    times, states, u_log = simulate(
        state0, (0.0, t_sim), params['dt'], ctrl.control_fn, params,
    )

    if np.any(np.isnan(states)):
        first_nan = int(np.argmax(np.any(np.isnan(states), axis=1)))
        result = dict(diverged=True, diverged_at_s=float(times[first_nan]),
                      captured=False, capture_time_s=None,
                      fr04_pass=False, fr05_pass=False,
                      ss_theta1_deg=None, ss_theta2_deg=None, peak_force_N=None)
        if verbose:
            print(f"\n  DIVERGED at t = {times[first_nan]:.2f} s\n")
        return result

    # Capture: first time both |θ₁| AND |θ₂| are within 30°
    cap_rad = np.radians(30.0)
    in_cap  = (np.abs(states[:, 1]) < cap_rad) & (np.abs(states[:, 2]) < cap_rad)
    # Check for sustained stability (angles stay within 5° for at least 2 s)
    stable_5deg = (np.abs(states[:, 1]) < np.radians(5.0)) & \
                  (np.abs(states[:, 2]) < np.radians(5.0))
    # Find first time stable_5deg holds for 400 consecutive steps (2 s at 200 Hz)
    window = 400
    captured, capture_time = False, None
    for i in range(len(times) - window):
        if np.all(stable_5deg[i:i+window]):
            captured, capture_time = True, float(times[i])
            break

    fr04_pass = bool(captured and capture_time <= 15.0)

    mask_last = times >= (t_sim - 30.0)
    th1_last  = np.degrees(np.abs(states[mask_last, 1]))
    th2_last  = np.degrees(np.abs(states[mask_last, 2]))
    fr05_pass = bool(captured and np.all(th1_last <= 5.0) and np.all(th2_last <= 5.0))

    ss_theta1  = float(np.degrees(np.abs(states[-1, 1])))
    ss_theta2  = float(np.degrees(np.abs(states[-1, 2])))
    peak_force = float(np.nanmax(np.abs(u_log)))

    result = dict(
        diverged=False,
        captured=captured,
        capture_time_s=capture_time,
        fr04_pass=fr04_pass,
        fr05_pass=fr05_pass,
        ss_theta1_deg=ss_theta1,
        ss_theta2_deg=ss_theta2,
        peak_force_N=peak_force,
    )

    if verbose:
        ok = lambda b: 'PASS ✓' if b else 'FAIL ✗'
        cap_str = f"{capture_time:.2f} s" if captured else "never"
        print()
        print("=" * 55)
        print("  VALIDATION RESULTS")
        print("=" * 55)
        print(f"  Simulation duration       : {t_sim:.0f} s")
        print(f"  Stabilised (both < 5°)    : {cap_str}")
        print(f"  FR-04  (stable ≤ 15 s)    : {ok(fr04_pass)}")
        print(f"  FR-05  (±5° for last 30 s): {ok(fr05_pass)}")
        print(f"  θ₁ at t = {t_sim:.0f} s         : {ss_theta1:.4f}°")
        print(f"  θ₂ at t = {t_sim:.0f} s         : {ss_theta2:.4f}°")
        print(f"  Peak control force        : {peak_force:.1f} N")
        print("=" * 55)
        print()

    return result


# ---------------------------------------------------------------------------
# Main tuning pipeline (with optional adaptive k_sw refinement)
# ---------------------------------------------------------------------------

def tune_all(params: dict, theta_acc_deg: float = 5.0,
             adaptive: bool = True) -> dict:
    """
    Run the full tuning pipeline.

    Parameters
    ----------
    params        : system parameters (from load_params())
    theta_acc_deg : desired steady-state angle accuracy (°)
    adaptive      : if True, validate the result and reduce k_sw until
                    the simulation is stable (up to 4 retries at 0.6× each)

    Returns
    -------
    dict with keys:
      k_sw, u_max, kick_frac, kick_dur,
      Q_stab, R_stab, Q_cap, R_cap,
      scales,
      adapted   — True if k_sw was reduced from the analytical value
    """
    scales = compute_scales(params)
    sw     = tune_swingup(params, scales)
    Q_stab, R_stab, Q_cap, R_cap = tune_lqr(params, scales, theta_acc_deg)

    tuned = dict(
        **sw,
        Q_stab=Q_stab, R_stab=R_stab,
        Q_cap=Q_cap,   R_cap=R_cap,
        scales=scales,
        adapted=False,
    )

    if not adaptive:
        return tuned

    # ── Geometry sanity check ──────────────────────────────────────────
    _, _, _, l1, l2, _ = _unpack(params)
    if l2 / l1 < 0.2:
        print(f"  [tune] WARNING: l2/l1 = {l2/l1:.3f} < 0.2 — second link is very short")
        print(f"         relative to l1. The Åström-Furuta energy pump relies on both")
        print(f"         links having comparable inertia. Swing-up may not converge.")
        print(f"         LQR stabiliser remains valid (±10° → 0° in ~30 s).")

    # ── Adaptive loop ─────────────────────────────────────────────────
    # Try up to 4 reductions of k_sw (0.6× each ≈ a ~9× range overall).
    k_sw_initial = tuned['k_sw']
    succeeded    = False
    best         = None   # best non-diverging fallback if none succeed

    for attempt in range(5):
        scale   = 0.6 ** attempt
        current = {**tuned, 'k_sw': k_sw_initial * scale,
                   'adapted': attempt > 0}

        result = validate(params, current, t_sim=60.0, verbose=False)

        if result['fr05_pass'] and not result['diverged']:
            best      = current
            succeeded = True
            break

        # Save first non-diverging attempt as fallback
        if not result['diverged'] and best is None:
            best = current

        if attempt == 0:
            print(f"  [tune] k_sw={current['k_sw']:.3f} → "
                  f"diverged={result['diverged']}, stable={result['fr05_pass']} — reducing k_sw…")
        else:
            print(f"  [tune] k_sw={current['k_sw']:.3f} (×{scale:.2f}) → "
                  f"diverged={result['diverged']}, stable={result['fr05_pass']}")

    if succeeded and best['adapted']:
        print(f"  [tune] Adapted k_sw: {k_sw_initial:.4f} → {best['k_sw']:.4f}")
    elif not succeeded:
        if best is None:
            best = tuned  # fall back to analytical (all attempts diverged)
        print()
        print("  [tune] WARNING: swing-up could not be stabilised for this geometry.")
        print(f"         Tried k_sw ∈ [{k_sw_initial*0.6**4:.3f}, {k_sw_initial:.3f}].")
        print( "         The LQR stabiliser (±10° → 0°) is still valid.")
        print( "         Returning best non-diverging parameters as starting point.")
        print( "         Suggestions: increase l2, or reduce m1/m2 ratio.")
        print()

    return best


# ---------------------------------------------------------------------------
# Pretty-printer
# ---------------------------------------------------------------------------

def print_tuned(params: dict, tuned: dict) -> None:
    sc = tuned['scales']

    print()
    print("=" * 60)
    print("  AUTO-TUNER — PARAMETER REPORT")
    print("=" * 60)

    M, m1, m2, l1, l2, g = _unpack(params)
    print(f"\n  System")
    print(f"    M  = {M:.3f} kg   m1 = {m1:.3f} kg   m2 = {m2:.3f} kg")
    print(f"    l1 = {l1:.3f} m    l2 = {l2:.3f} m")

    print(f"\n  Characteristic scales")
    print(f"    m_total    = {sc['m_total']:.3f} kg")
    print(f"    E_target   = {sc['E_target']:.3f} J   (upright PE)")
    print(f"    T_nat      = {sc['T_nat']:.3f} s   (link 1 natural period)")
    print(f"    v_char     = {sc['v_char']:.3f} m/s (characteristic cart speed)")
    print(f"    λ_max      = {sc['lambda_max']:.3f} 1/s (largest open-loop eigenvalue)")

    adapted_tag = "  ← adapted" if tuned['adapted'] else ""
    print(f"\n  Swing-up parameters")
    print(f"    u_max      = {tuned['u_max']:.2f} N")
    print(f"    k_sw       = {tuned['k_sw']:.4f}{adapted_tag}")
    print(f"    kick_frac  = {tuned['kick_frac']:.2f}  (kick = {tuned['kick_frac']*tuned['u_max']:.1f} N)")
    print(f"    kick_dur   = {tuned['kick_dur']:.3f} s")
    print(f"    force gate = {_CAPTURE_FORCE_FACTOR}× u_max = {_CAPTURE_FORCE_FACTOR * tuned['u_max']:.1f} N")

    labels = ['x ', 'θ₁', 'θ₂', 'ẋ ', 'θ̇₁', 'θ̇₂']
    print(f"\n  LQR — stabiliser  Q_stab / R_stab")
    for lbl, q in zip(labels, np.diag(tuned['Q_stab'])):
        print(f"    Q[{lbl}] = {q:>10.3f}")
    print(f"    R      = {float(tuned['R_stab'].flat[0]):>10.5f}")

    print(f"\n  LQR — capture  Q_cap / R_cap")
    for lbl, q in zip(labels, np.diag(tuned['Q_cap'])):
        print(f"    Q[{lbl}] = {q:>10.3f}")
    print(f"    R      = {float(tuned['R_cap'].flat[0]):>10.5f}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Auto-tune SwingUpLQRController for the current config.yaml',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config',       default=None, metavar='PATH',
                        help='Path to config.yaml (default: next to this file)')
    parser.add_argument('--no-adaptive',  action='store_true',
                        help='Skip adaptive k_sw refinement (analytical values only)')
    parser.add_argument('--theta-acc',    type=float, default=5.0, metavar='DEG',
                        help='Desired steady-state accuracy in degrees (default: 5)')
    parser.add_argument('--t-sim',        type=float, default=60.0, metavar='SECS',
                        help='Final validation simulation duration in seconds (default: 60)')
    args = parser.parse_args()

    params = load_params(args.config)
    tuned  = tune_all(params,
                      theta_acc_deg=args.theta_acc,
                      adaptive=not args.no_adaptive)

    print_tuned(params, tuned)
    validate(params, tuned, t_sim=args.t_sim)


if __name__ == '__main__':
    main()
