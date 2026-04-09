"""
report.py — Performance Report (Milestone 5)

Runs four quantitative tests and prints a structured report covering all
functional requirements:

  1. Swing-up performance  (FR-04) — settling time, capture time
  2. LQR stabilisation     (FR-05) — angle RMS/max over last 30 s
  3. Disturbance rejection          — peak deviation and recovery time
  4. Energy conservation   (NFR-02) — drift over 30 s free-swing

Usage
-----
  python report.py                    # print to stdout
  python report.py --save report.md   # also write a Markdown file
  python report.py --config alt.yaml  # custom config
"""

import argparse
from pathlib import Path

import numpy as np

from dynamics import load_params, simulate
from controller_lqr import LQRController, SwingUpLQRController


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _settling_time(times, states, threshold_deg=5.0, hold_s=10.0):
    """
    First time t* such that max(|θ₁|, |θ₂|) ≤ threshold_deg for all t ≥ t*
    within a hold window of hold_s seconds.  Returns None if never settled.
    """
    th_rad = np.radians(threshold_deg)
    in_band = (np.abs(states[:, 1]) <= th_rad) & (np.abs(states[:, 2]) <= th_rad)
    dt = float(times[1] - times[0])
    hold_steps = max(1, int(hold_s / dt))
    for i in range(len(times) - hold_steps):
        if np.all(in_band[i : i + hold_steps]):
            return float(times[i])
    return None


def _first_entry_time(times, states, threshold_deg=30.0):
    """First time both |θ₁| and |θ₂| are within threshold_deg (capture zone)."""
    th_rad = np.radians(threshold_deg)
    mask = (np.abs(states[:, 1]) <= th_rad) & (np.abs(states[:, 2]) <= th_rad)
    idx = np.argmax(mask)
    return float(times[idx]) if mask[idx] else None


def _control_effort(times, u_log):
    """Integral of |u| dt and RMS of u."""
    dt   = float(times[1] - times[0]) if len(times) > 1 else 0.0
    rms  = float(np.sqrt(np.mean(u_log ** 2)))
    integral = float(np.sum(np.abs(u_log)) * dt)
    return rms, integral


def _energy_drift(times, states, params):
    from dynamics import total_energy
    E = np.array([total_energy(s, params) for s in states])
    drift_abs = abs(float(E[-1]) - float(E[0]))
    drift_pct = drift_abs / (abs(float(E[0])) + 1e-12) * 100.0
    return drift_abs, drift_pct, E


# ---------------------------------------------------------------------------
# Test runners
# ---------------------------------------------------------------------------

def test_swingup(params, t_end=40.0):
    """
    FR-04: swing-up from hanging rest.
    Returns a dict of metrics.
    """
    ctrl = SwingUpLQRController(params)
    ctrl.trigger()
    state0 = np.array([0.0, np.pi - 0.01, 0.0, 0.0, 0.0, 0.0])
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    ctrl.control_fn, params)

    capture_t  = _first_entry_time(times, states, threshold_deg=30.0)
    settling_t = _settling_time(times, states, threshold_deg=5.0, hold_s=10.0)
    u_rms, u_int = _control_effort(times, u_log)
    peak_u = float(np.max(np.abs(u_log)))

    return dict(
        times=times, states=states, u_log=u_log,
        capture_time=capture_t,
        settling_time=settling_t,
        u_rms=u_rms,
        u_integral=u_int,
        peak_u=peak_u,
        fr04_pass=(settling_t is not None and settling_t <= 15.0),
    )


def test_lqr(params, t_end=35.0):
    """
    FR-05: LQR from ±10° perturbation — must hold ±5° for last 30 s.
    Returns a dict of metrics.
    """
    ctrl   = LQRController(params)
    state0 = np.array([0.0, np.radians(10.0), np.radians(-8.0), 0.0, 0.0, 0.0])
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    ctrl.control_fn, params)

    window = times >= (times[-1] - 30.0)
    th1_max = float(np.max(np.abs(np.degrees(states[window, 1]))))
    th2_max = float(np.max(np.abs(np.degrees(states[window, 2]))))
    th1_rms = float(np.sqrt(np.mean(np.degrees(states[window, 1])**2)))
    th2_rms = float(np.sqrt(np.mean(np.degrees(states[window, 2])**2)))
    u_rms, u_int = _control_effort(times, u_log)
    peak_u = float(np.max(np.abs(u_log)))

    return dict(
        times=times, states=states, u_log=u_log,
        th1_max=th1_max, th2_max=th2_max,
        th1_rms=th1_rms, th2_rms=th2_rms,
        u_rms=u_rms, u_integral=u_int, peak_u=peak_u,
        fr05_pass=(th1_max <= 5.0 and th2_max <= 5.0),
    )


def test_disturbance(params, t_settle=8.0, impulse_N=50.0, impulse_dur=0.05,
                     t_after=20.0):
    """
    Disturbance rejection test.

    Protocol
    --------
    1. Start LQR from [0, 0.05, 0.03, 0, 0, 0] — small perturbation
    2. Let it settle for t_settle seconds
    3. Apply a +impulse_N force for impulse_dur seconds
    4. Continue for t_after seconds; measure peak deviation and recovery time

    Returns a dict of metrics.
    """
    ctrl = LQRController(params)
    t_impulse_start = t_settle
    t_impulse_end   = t_settle + impulse_dur
    t_total         = t_settle + impulse_dur + t_after

    def ctrl_disturbed(t, state):
        u = ctrl.control_fn(t, state)
        if t_impulse_start <= t < t_impulse_end:
            u += impulse_N
        return u

    state0 = np.array([0.0, np.radians(5.0), np.radians(3.0), 0.0, 0.0, 0.0])
    times, states, u_log = simulate(state0, (0.0, t_total), params['dt'],
                                    ctrl_disturbed, params)

    # Analyse only the post-impulse window
    post = times >= t_impulse_end
    if post.sum() == 0:
        return dict(peak_th1=None, peak_th2=None, recovery_time=None,
                    impulse_N=impulse_N, impulse_dur=impulse_dur)

    peak_th1 = float(np.max(np.abs(np.degrees(states[post, 1]))))
    peak_th2 = float(np.max(np.abs(np.degrees(states[post, 2]))))

    # Recovery: first time after impulse both angles re-enter ±5° and hold for 5 s
    post_times  = times[post]
    post_states = states[post]
    rec_t = _settling_time(post_times, post_states, threshold_deg=5.0, hold_s=5.0)
    if rec_t is not None:
        recovery_time = rec_t - float(t_impulse_end)
    else:
        recovery_time = None

    return dict(
        times=times, states=states, u_log=u_log,
        peak_th1=peak_th1, peak_th2=peak_th2,
        recovery_time=recovery_time,
        impulse_N=impulse_N, impulse_dur=impulse_dur,
    )


def test_energy(params, t_end=30.0):
    """
    NFR-02: energy conservation over t_end seconds of free-swing.
    Returns drift metrics.
    """
    state0 = np.array([0.0, 2.5, -1.2, 0.0, 0.0, 0.0])
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    lambda t, s: 0.0, params)
    drift_abs, drift_pct, E = _energy_drift(times, states, params)
    return dict(
        drift_abs=drift_abs, drift_pct=drift_pct,
        E0=float(E[0]), E_end=float(E[-1]),
        nfr02_pass=(drift_pct < 1.0),
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def _fmt(value, unit='', width=8, precision=3):
    if value is None:
        return 'N/A'.rjust(width)
    return f'{value:>{width}.{precision}f} {unit}'.rstrip()


def build_report(params, r_su, r_lqr, r_dist, r_en):
    lines = []
    A = lines.append

    M, m1, m2 = params['M'], params['m1'], params['m2']
    l1, l2, g = params['l1'], params['l2'], params['g']
    dt        = params['dt']

    A("=" * 66)
    A("  DOUBLE PENDULUM ROBOT — PERFORMANCE REPORT")
    A("=" * 66)
    A("")
    A("  System parameters")
    A(f"  {'Cart':12s}: M  = {M} kg")
    A(f"  {'Link 1':12s}: m1 = {m1} kg,  l1 = {l1} m")
    A(f"  {'Link 2':12s}: m2 = {m2} kg,  l2 = {l2} m")
    A(f"  {'Gravity':12s}: g  = {g} m/s²")
    A(f"  {'Timestep':12s}: dt = {dt} s  ({round(1/dt)} Hz)")
    A("")

    # ── 1. Swing-up (FR-04) ─────────────────────────────────────────────────
    A("-" * 66)
    A("  TEST 1 — Swing-Up Performance  (FR-04)")
    A("-" * 66)
    A(f"  Capture time (enter ±30°)   : {_fmt(r_su['capture_time'], 's')}")
    A(f"  Settling time (±5°, hold 10s): {_fmt(r_su['settling_time'], 's')}")
    A(f"  Peak control force          : {_fmt(r_su['peak_u'], 'N')}")
    A(f"  Control effort ∫|u| dt      : {_fmt(r_su['u_integral'], 'N·s')}")
    A(f"  Control effort RMS          : {_fmt(r_su['u_rms'], 'N')}")
    fr04 = r_su['fr04_pass']
    req  = 15.0
    A(f"  FR-04 (settle ≤ {req:.0f} s)        : {'PASS ✓' if fr04 else 'FAIL ✗'}")
    A("")

    # ── 2. LQR stabilisation (FR-05) ────────────────────────────────────────
    A("-" * 66)
    A("  TEST 2 — LQR Stabilisation  (FR-05)")
    A("  Initial state: θ₁ = +10°, θ₂ = −8°  |  Measured over last 30 s")
    A("-" * 66)
    A(f"  θ₁  max  : {_fmt(r_lqr['th1_max'], '°')}")
    A(f"  θ₁  RMS  : {_fmt(r_lqr['th1_rms'], '°')}")
    A(f"  θ₂  max  : {_fmt(r_lqr['th2_max'], '°')}")
    A(f"  θ₂  RMS  : {_fmt(r_lqr['th2_rms'], '°')}")
    A(f"  Peak |u| : {_fmt(r_lqr['peak_u'], 'N')}")
    A(f"  RMS  |u| : {_fmt(r_lqr['u_rms'], 'N')}")
    fr05 = r_lqr['fr05_pass']
    A(f"  FR-05 (|θ₁|, |θ₂| ≤ 5° last 30 s): {'PASS ✓' if fr05 else 'FAIL ✗'}")
    A("")

    # ── 3. Disturbance rejection ─────────────────────────────────────────────
    A("-" * 66)
    A("  TEST 3 — Disturbance Rejection")
    A(f"  Impulse: {r_dist['impulse_N']:.0f} N for {r_dist['impulse_dur']*1000:.0f} ms  "
      "(applied after settling)")
    A("-" * 66)
    A(f"  Peak θ₁ deviation post-impulse: {_fmt(r_dist['peak_th1'], '°')}")
    A(f"  Peak θ₂ deviation post-impulse: {_fmt(r_dist['peak_th2'], '°')}")
    A(f"  Recovery time (back to ±5°)   : {_fmt(r_dist['recovery_time'], 's')}")
    recovered = r_dist['recovery_time'] is not None
    A(f"  Recovered                     : {'YES ✓' if recovered else 'NO  ✗'}")
    A("")

    # ── 4. Energy conservation (NFR-02) ─────────────────────────────────────
    A("-" * 66)
    A("  TEST 4 — Energy Conservation  (NFR-02, 30 s free-swing)")
    A("-" * 66)
    A(f"  E(t=0)       : {_fmt(r_en['E0'], 'J')}")
    A(f"  E(t=30 s)    : {_fmt(r_en['E_end'], 'J')}")
    A(f"  Drift (abs)  : {_fmt(r_en['drift_abs'], 'J')}")
    A(f"  Drift (%)    : {r_en['drift_pct']:.6f} %")
    A(f"  NFR-02 (< 1%): {'PASS ✓' if r_en['nfr02_pass'] else 'FAIL ✗'}")
    A("")

    # ── Summary ──────────────────────────────────────────────────────────────
    A("=" * 66)
    A("  SUMMARY")
    A("=" * 66)
    all_pass = fr04 and fr05 and r_en['nfr02_pass']
    rows = [
        ("FR-04", "Swing-up settle ≤ 15 s",          "PASS ✓" if fr04       else "FAIL ✗"),
        ("FR-05", "Balance ±5° for 30 s",             "PASS ✓" if fr05       else "FAIL ✗"),
        ("NFR-02","Energy drift < 1 % (30 s)",        "PASS ✓" if r_en['nfr02_pass'] else "FAIL ✗"),
        ("—",     "Disturbance recovery (50 N pulse)", "PASS ✓" if recovered  else "FAIL ✗"),
    ]
    for req_id, desc, result in rows:
        A(f"  [{req_id:6s}]  {desc:<35s}  {result}")
    A("")
    A(f"  Overall: {'ALL PASS ✓' if all_pass and recovered else 'SOME FAILURES — see above'}")
    A("=" * 66)
    A("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description='Double Pendulum Robot — performance report')
    p.add_argument('--config', type=str, default=None,
                   help='Path to YAML config (default: config.yaml)')
    p.add_argument('--save', type=str, default=None, metavar='PATH',
                   help='Write report to a Markdown file as well as stdout')
    return p.parse_args(argv)


def main(argv=None):
    args   = _parse_args(argv)
    config = Path(args.config) if args.config else None
    params = load_params(config)

    print("\nRunning performance tests — this may take a few seconds …\n")

    print("  [1/4] Swing-up simulation …", end='', flush=True)
    r_su = test_swingup(params)
    print(" done")

    print("  [2/4] LQR stabilisation  …", end='', flush=True)
    r_lqr = test_lqr(params)
    print(" done")

    print("  [3/4] Disturbance test   …", end='', flush=True)
    r_dist = test_disturbance(params)
    print(" done")

    print("  [4/4] Energy conservation …", end='', flush=True)
    r_en = test_energy(params)
    print(" done\n")

    report = build_report(params, r_su, r_lqr, r_dist, r_en)
    print(report)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"[report] Saved → {out}")


if __name__ == '__main__':
    main()
