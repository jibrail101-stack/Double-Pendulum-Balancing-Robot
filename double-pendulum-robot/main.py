"""
main.py — Double Pendulum Robot — Integration Pipeline (Milestone 5)

Ties together every module into a single runnable entry point:
  dynamics.py       — Lagrangian EOM, RK4 integrator
  controller_lqr.py — LQR stabiliser + swing-up (Åström-Furuta energy pump)
  visualizer.py     — real-time 2D Matplotlib animation
  logger.py         — CSV / JSON state logging
  tune.py           — analytical auto-tuner (optional pre-run report)

Modes
-----
  interactive  (default) — live simulation; drag cart with mouse, press G to swing up
  swingup                — batch simulate a full swing-up from hanging rest, then replay
  lqr                    — batch simulate LQR from a small perturbation (±10°), replay
  free                   — batch simulate free-swing (no control), replay

Usage
-----
  python main.py                             # interactive swing-up (default)
  python main.py --mode swingup              # batch simulate + replay
  python main.py --mode lqr                  # LQR stabiliser batch run
  python main.py --mode free                 # free-swing batch run
  python main.py --config path/to/cfg.yaml   # use a custom config file
  python main.py --t-end 30                  # simulation duration for batch modes (s)
  python main.py --save results/swing.gif    # save animation to .gif or .mp4
  python main.py --no-viz                    # headless: log only, no display
  python main.py --tune                      # print auto-tuner parameter report first
  python main.py --log-dir results           # directory for CSV/JSON output
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from dynamics import load_params, simulate
from controller_lqr import LQRController, SwingUpLQRController
from visualizer import CartPendulumVisualizer
from logger import StateLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner(mode: str, params: dict) -> None:
    M, m1, m2 = params['M'], params['m1'], params['m2']
    l1, l2    = params['l1'], params['l2']
    g, dt     = params['g'], params['dt']
    print()
    print("=" * 60)
    print("  DOUBLE PENDULUM ROBOT")
    print("=" * 60)
    print(f"  Mode      : {mode.upper()}")
    print(f"  Cart      : M  = {M} kg")
    print(f"  Link 1    : m1 = {m1} kg,  l1 = {l1} m")
    print(f"  Link 2    : m2 = {m2} kg,  l2 = {l2} m")
    print(f"  Gravity   : g  = {g} m/s²")
    print(f"  Timestep  : dt = {dt} s  ({round(1/dt)} Hz)")
    print("=" * 60)
    print()


def _run_tuner(params: dict) -> None:
    """Print auto-tuner report without modifying the controller."""
    try:
        from tune import tune_all
        print("[tune] Running auto-tuner …")
        tune_all(params)
    except Exception as exc:
        print(f"[tune] Auto-tuner failed: {exc}")


def _save_logs(logger: StateLogger, log_dir: Path, prefix: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = log_dir / f"{prefix}_trajectory.csv"
    json_path = log_dir / f"{prefix}_trajectory.json"
    logger.to_csv(csv_path)
    logger.to_json(json_path)
    print(f"[log] CSV  → {csv_path}")
    print(f"[log] JSON → {json_path}")


# ---------------------------------------------------------------------------
# Simulation modes
# ---------------------------------------------------------------------------

def run_interactive(params: dict) -> None:
    """
    Live interactive simulation.

    Controls
    --------
    Left-drag        — move cart with mouse
    G / Enter        — trigger swing-up + LQR stabilisation sequence
    R / Right-click  — reset to hanging at rest
    Space            — pause / resume
    """
    ctrl   = SwingUpLQRController(params)
    state0 = np.array([0.0, np.pi - 0.01, 0.0, 0.0, 0.0, 0.0])

    print("[main] Interactive mode.")
    print("[main]   G / Enter  → swing up and stabilise")
    print("[main]   R          → reset to hanging rest")
    print("[main]   Space      → pause / resume")
    print()

    viz  = CartPendulumVisualizer(params)
    _anim = viz.run_interactive(state0=state0, swingup_ctrl=ctrl)   # noqa: F841 — keep reference


def run_swingup(params: dict, t_end: float, log_dir: Path,
                visualize: bool, save_path: str | None) -> None:
    """
    Batch simulation of a full swing-up + LQR stabilisation from hanging rest.
    Logs the trajectory and (optionally) replays or saves the animation.
    """
    ctrl = SwingUpLQRController(params)
    ctrl.trigger()   # start energy pump immediately

    state0 = np.array([0.0, np.pi - 0.01, 0.0, 0.0, 0.0, 0.0])
    print(f"[main] Simulating swing-up for {t_end} s …")
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    ctrl.control_fn, params)

    logger = StateLogger(params)
    logger.log_trajectory(times, states, u_log)
    logger.summary()
    _save_logs(logger, log_dir, "swingup")

    if visualize:
        viz = CartPendulumVisualizer(params)
        if save_path:
            viz.save(times, states, u_log, filepath=save_path)
        else:
            viz.replay(times, states, u_log)


def run_lqr(params: dict, t_end: float, log_dir: Path,
            visualize: bool, save_path: str | None) -> None:
    """
    Batch simulation of LQR stabilisation from a ±10° perturbation.
    Validates FR-05: must maintain |θ₁|, |θ₂| ≤ 5° for the last 30 s.
    """
    ctrl   = LQRController(params)
    ctrl.print_info()

    state0 = np.array([0.0, np.radians(10.0), np.radians(-8.0), 0.0, 0.0, 0.0])
    print(f"[main] Simulating LQR stabilisation for {t_end} s …")
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    ctrl.control_fn, params)

    logger = StateLogger(params)
    logger.log_trajectory(times, states, u_log)
    logger.summary()

    # FR-05 check: ±5° during last 30 s
    window = times >= (times[-1] - 30.0)
    if window.sum() > 0:
        th1_rms = float(np.sqrt(np.mean(np.degrees(states[window, 1])**2)))
        th2_rms = float(np.sqrt(np.mean(np.degrees(states[window, 2])**2)))
        th1_max = float(np.max(np.abs(np.degrees(states[window, 1]))))
        th2_max = float(np.max(np.abs(np.degrees(states[window, 2]))))
        fr05 = th1_max <= 5.0 and th2_max <= 5.0
        print()
        print("  FR-05 check (|θ| ≤ 5° for last 30 s):")
        print(f"    θ₁  RMS = {th1_rms:.3f}°   max = {th1_max:.3f}°")
        print(f"    θ₂  RMS = {th2_rms:.3f}°   max = {th2_max:.3f}°")
        print(f"    Result  : {'PASS ✓' if fr05 else 'FAIL ✗'}")
        print()

    _save_logs(logger, log_dir, "lqr")

    if visualize:
        viz = CartPendulumVisualizer(params)
        if save_path:
            viz.save(times, states, u_log, filepath=save_path)
        else:
            viz.replay(times, states, u_log)


def run_free(params: dict, t_end: float, log_dir: Path,
             visualize: bool, save_path: str | None) -> None:
    """
    Batch simulation of free-swing from an offset position (no control).
    Useful for validating energy conservation in the dynamics model.
    """
    state0 = np.array([0.0, 2.5, -1.2, 0.0, 0.0, 0.0])
    print(f"[main] Simulating free-swing for {t_end} s …")
    times, states, u_log = simulate(state0, (0.0, t_end), params['dt'],
                                    lambda t, s: 0.0, params)

    logger = StateLogger(params)
    logger.log_trajectory(times, states, u_log)
    logger.summary()

    # Energy conservation check  (energy property → (N, 3): [KE, PE, E_total])
    E0  = float(logger.energy[0,  2])
    E_f = float(logger.energy[-1, 2])
    E_drift_pct = abs(E_f - E0) / (abs(E0) + 1e-12) * 100.0
    print(f"  Energy conservation:  E₀ = {E0:.4f} J  →  Ef = {E_f:.4f} J")
    print(f"  Drift = {E_drift_pct:.4f} %  {'(PASS ✓)' if E_drift_pct < 1.0 else '(FAIL ✗)'}")
    print()

    _save_logs(logger, log_dir, "free")

    if visualize:
        viz = CartPendulumVisualizer(params)
        if save_path:
            viz.save(times, states, u_log, filepath=save_path)
        else:
            viz.replay(times, states, u_log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Double Pendulum Robot — simulation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--mode', choices=['interactive', 'swingup', 'lqr', 'free'],
        default='interactive',
        help='Simulation mode (default: interactive)',
    )
    p.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file (default: config.yaml alongside main.py)',
    )
    p.add_argument(
        '--t-end', type=float, default=30.0,
        help='Simulation duration in seconds for batch modes (default: 30)',
    )
    p.add_argument(
        '--save', type=str, default=None, metavar='PATH',
        help='Save animation to .gif or .mp4 (batch modes only)',
    )
    p.add_argument(
        '--no-viz', action='store_true',
        help='Headless mode: log and print only, no animation window',
    )
    p.add_argument(
        '--tune', action='store_true',
        help='Print auto-tuner parameter report before running',
    )
    p.add_argument(
        '--log-dir', type=str, default='results',
        help='Directory for CSV/JSON output (default: results/)',
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args    = _parse_args(argv)
    config  = Path(args.config) if args.config else None
    params  = load_params(config)
    log_dir = Path(args.log_dir)

    _print_banner(args.mode, params)

    if args.tune:
        _run_tuner(params)

    visualize = not args.no_viz
    save_path = args.save

    if args.mode == 'interactive':
        if args.no_viz:
            print("[main] --no-viz has no effect in interactive mode — skipping.")
            sys.exit(0)
        run_interactive(params)

    elif args.mode == 'swingup':
        run_swingup(params, args.t_end, log_dir, visualize, save_path)

    elif args.mode == 'lqr':
        run_lqr(params, args.t_end, log_dir, visualize, save_path)

    elif args.mode == 'free':
        run_free(params, args.t_end, log_dir, visualize, save_path)


if __name__ == '__main__':
    main()
