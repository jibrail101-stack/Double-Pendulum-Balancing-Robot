"""
tests/test_maneuver.py — Acceleration-schedule maneuver test

Starting from the pendulum hanging perfectly vertically downward (θ₁ = π),
apply a sequence of prescribed cart accelerations and evaluate how close the
double pendulum gets to the upright position (θ₁ = θ₂ = 0).

──────────────────────────────────────────────────
HOW TO USE
──────────────────────────────────────────────────
1. Edit MANEUVER below — each tuple is (cart_acceleration m/s², duration s).
   The total duration must not exceed 10 seconds.

2. Run:
       python3 -m pytest tests/test_maneuver.py -v -s

   The -s flag shows the printed report in the terminal.

3. Read the report, tweak MANEUVER, re-run.

──────────────────────────────────────────────────
PHYSICS NOTE
──────────────────────────────────────────────────
The cart acceleration is converted to the exact required force using inverse
dynamics — it accounts for how the swinging pendulum pushes back on the cart,
not just F = M·a of the cart mass alone.

The double pendulum is a notoriously chaotic system. Swinging it to upright by
hand-tuning accelerations is genuinely difficult. Use this test to build
intuition, then the energy-based swing-up controller in Milestone 4 will do
it systematically.
──────────────────────────────────────────────────
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dynamics import load_params, simulate, cart_accel_to_force, total_energy

# ===========================================================================
# EDIT THESE  ↓
# ===========================================================================

# Each tuple: (cart acceleration m/s²,  duration s)
# Total duration must be ≤ 10 s
MANEUVER = [
    ( 3.0, 1.5),   # push right hard — starts swinging link 1
    (-2.0, 1.0),   # pull back — pumps energy into swing
    ( 3.0, 1.0),   # push right again on the upswing
    (-2.5, 1.0),   # reverse — links should be rising
    ( 2.0, 1.0),   # continue pumping
    (-1.5, 1.0),   # refine
    ( 1.0, 1.5),   # gentle push toward top
]

# How close to upright counts as "reached upright" (degrees)
# θ₁ = 0° is perfect upright. Loosen this if you're just exploring.
UPRIGHT_THRESHOLD_DEG = 30.0

# ===========================================================================
# Helpers
# ===========================================================================

PARAMS = load_params()


def _build_control_fn(maneuver, params):
    """
    Convert a list of (accel, duration) segments into a control_fn(t, state).
    Computes the exact force needed at each timestep via inverse dynamics.
    After the schedule ends the cart is unforced (u = 0).
    """
    # Pre-compute cumulative time breakpoints
    t_ends = []
    t_acc = 0.0
    for accel, dur in maneuver:
        t_acc += dur
        t_ends.append((t_acc, accel))

    def control_fn(t_now, state):
        for t_end, accel in t_ends:
            if t_now < t_end:
                return cart_accel_to_force(state, accel, params)
        return 0.0

    return control_fn


def _angular_distance_to_upright(th1_array):
    """
    Return the angular distance (rad) of each θ₁ value from upright (θ₁ = 0),
    correctly wrapped to [0, π] regardless of how many times the link has
    rotated past ±π.
    """
    return np.abs(np.arctan2(np.sin(th1_array), np.cos(th1_array)))


def _run_maneuver():
    """Simulate from hanging and return (times, states)."""
    state0 = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])  # hanging, at rest
    total_time = sum(d for _, d in MANEUVER)
    control_fn  = _build_control_fn(MANEUVER, PARAMS)
    return simulate(state0, (0.0, total_time), PARAMS['dt'], control_fn, PARAMS)  # (times, states, u_log)


# ===========================================================================
# Tests
# ===========================================================================

def test_maneuver_duration():
    """Total maneuver duration must not exceed 10 s (FR-04 context)."""
    total = sum(d for _, d in MANEUVER)
    assert total <= 10.0, (
        f"Maneuver is {total:.1f} s — exceeds the 10 s limit. "
        "Shorten one or more segments."
    )


def test_maneuver_numerically_stable():
    """
    The simulation must not produce NaN or Inf at any timestep.
    This confirms the EOM and integrator stay well-conditioned for your
    input schedule. If this fails, the accelerations are likely too large
    (ill-conditioned mass matrix) or the timestep is too big.
    """
    times, states, u_log = _run_maneuver()
    assert not np.any(np.isnan(states)), (
        "Simulation produced NaN — check for extreme accelerations or reduce dt."
    )
    assert not np.any(np.isinf(states)), (
        "Simulation produced Inf — cart force may have become singular."
    )


def test_maneuver_report():
    """
    Always passes. Prints a detailed trajectory report so you can read the
    outcome and tune MANEUVER accordingly.

    Run with:  python3 -m pytest tests/test_maneuver.py::test_maneuver_report -v -s
    """
    times, states, u_log = _run_maneuver()
    total_time = times[-1]

    th1 = states[:, 1]
    th2 = states[:, 2]
    th1d = states[:, 4]
    th2d = states[:, 5]

    dist = _angular_distance_to_upright(th1)   # angular distance from upright (rad)
    best_idx = int(np.argmin(dist))

    E0 = total_energy(states[0], PARAMS)
    E_arr = np.array([total_energy(s, PARAMS) for s in states[::20]])  # sample every 20 steps

    # Print maneuver schedule
    print("\n" + "="*55)
    print("  MANEUVER SCHEDULE")
    print("="*55)
    t_acc = 0.0
    for i, (accel, dur) in enumerate(MANEUVER):
        print(f"  Segment {i+1}: {accel:+.2f} m/s²  for {dur:.1f} s"
              f"  [{t_acc:.1f} → {t_acc+dur:.1f} s]")
        t_acc += dur
    print(f"  Total duration: {total_time:.2f} s")

    # Print trajectory summary
    print("\n" + "="*55)
    print("  TRAJECTORY REPORT")
    print("="*55)
    print(f"  Start   t = 0.000 s")
    print(f"    θ₁ = {np.degrees(th1[0]):+7.2f}°   (180° = hanging)")
    print(f"    θ₂ = {np.degrees(th2[0]):+7.2f}°")
    print(f"    E₀ = {E0:.4f} J")

    print(f"\n  Closest to upright  t = {times[best_idx]:.3f} s")
    print(f"    θ₁ = {np.degrees(th1[best_idx]):+7.2f}°   (0° = upright ✓)")
    print(f"    θ₂ = {np.degrees(th2[best_idx]):+7.2f}°   (0° = upright ✓)")
    print(f"    θ̇₁ = {th1d[best_idx]:+7.3f} rad/s")
    print(f"    θ̇₂ = {th2d[best_idx]:+7.3f} rad/s")
    print(f"    Δ from upright = {np.degrees(dist[best_idx]):.2f}°")

    print(f"\n  End     t = {total_time:.3f} s")
    print(f"    θ₁ = {np.degrees(th1[-1]):+7.2f}°")
    print(f"    θ₂ = {np.degrees(th2[-1]):+7.2f}°")
    print(f"    θ̇₁ = {th1d[-1]:+7.3f} rad/s")
    print(f"    θ̇₂ = {th2d[-1]:+7.3f} rad/s")

    print(f"\n  Energy injected: {E_arr[-1] - E0:+.4f} J"
          f"  (start: {E0:.4f} J  →  end: {E_arr[-1]:.4f} J)")
    print("="*55 + "\n")

    assert True  # always passes — this test is a report, not a gate


def test_maneuver_reaches_upright():
    """
    Check that the pendulum passes within UPRIGHT_THRESHOLD_DEG of vertical
    at some point during the maneuver.

    If this fails, read the report from test_maneuver_report and adjust MANEUVER.
    You can also relax UPRIGHT_THRESHOLD_DEG at the top of this file while
    you're still exploring.
    """
    times, states, u_log = _run_maneuver()

    th1 = states[:, 1]
    dist = _angular_distance_to_upright(th1)
    best_idx = int(np.argmin(dist))
    best_deg = float(np.degrees(dist[best_idx]))

    assert best_deg < UPRIGHT_THRESHOLD_DEG, (
        f"Pendulum never came within {UPRIGHT_THRESHOLD_DEG:.0f}° of upright.\n"
        f"Best achieved: θ₁ = {best_deg:.1f}° at t = {times[best_idx]:.2f} s.\n"
        f"Tip: run test_maneuver_report to see the full trajectory, "
        f"then tune MANEUVER or increase UPRIGHT_THRESHOLD_DEG."
    )
