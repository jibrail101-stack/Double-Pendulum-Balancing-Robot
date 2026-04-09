"""
logger.py — State trajectory and control input logger (Milestone 2)

Records every timestep of a simulation to memory, then exports as CSV or JSON.
Each row stores: time, full state vector, control force, kinetic energy,
potential energy, and total energy.

Usage
-----
    from dynamics import load_params, simulate
    from logger import StateLogger

    params = load_params()
    logger = StateLogger(params)

    times, states, u_log = simulate(state0, t_span, dt, ctrl, params)
    logger.log_trajectory(times, states, u_log)

    logger.summary()
    logger.to_csv('run_001.csv')
    logger.to_json('run_001.json')
"""

import csv
import json
import numpy as np
from pathlib import Path
from dynamics import total_energy


class StateLogger:
    """
    Logs state, control, and energy data for a cart-double pendulum simulation.

    Columns recorded per timestep
    ──────────────────────────────
    t        — simulation time (s)
    x        — cart position (m)
    th1      — link 1 angle from vertical (rad)
    th2      — link 2 angle relative to link 1 (rad)
    xd       — cart velocity (m/s)
    th1d     — link 1 angular velocity (rad/s)
    th2d     — link 2 angular velocity (rad/s)
    u        — cart force (N)
    KE       — kinetic energy (J)
    PE       — potential energy (J)
    E_total  — total mechanical energy (J)
    """

    _COLUMNS = ['t', 'x', 'th1', 'th2', 'xd', 'th1d', 'th2d',
                'u', 'KE', 'PE', 'E_total']

    def __init__(self, params: dict):
        self._params = params
        self._rows: list[dict] = []

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, t: float, state, u: float) -> None:
        """Record a single timestep."""
        state = np.asarray(state, dtype=float)
        x, th1, th2, xd, th1d, th2d = state

        E_tot = total_energy(state, self._params)
        PE    = self._potential(state)
        KE    = E_tot - PE

        self._rows.append({
            't':       float(t),
            'x':       float(x),
            'th1':     float(th1),
            'th2':     float(th2),
            'xd':      float(xd),
            'th1d':    float(th1d),
            'th2d':    float(th2d),
            'u':       float(u),
            'KE':      float(KE),
            'PE':      float(PE),
            'E_total': float(E_tot),
        })

    def log_trajectory(self, times, states, u_log) -> None:
        """
        Bulk-log the output of dynamics.simulate().

        Parameters
        ----------
        times  : np.ndarray (N,)
        states : np.ndarray (N, 6)
        u_log  : np.ndarray (N,)
        """
        for t, s, u in zip(times, states, u_log):
            self.log(t, s, u)

    def clear(self) -> None:
        """Discard all logged data."""
        self._rows.clear()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_csv(self, filepath) -> Path:
        """
        Write logged data to a CSV file.

        Returns the resolved Path written to.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"[logger] CSV saved → {filepath}  ({len(self._rows)} rows)")
        return filepath

    def to_json(self, filepath) -> Path:
        """
        Write logged data to a JSON file.

        Structure: { "params": {...}, "columns": [...], "data": [[...], ...] }
        Storing column-major arrays keeps the file compact and avoids
        repeating column names on every row.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            'params':  self._params,
            'columns': self._COLUMNS,
            'data':    [[row[c] for c in self._COLUMNS] for row in self._rows],
        }

        with filepath.open('w') as f:
            json.dump(payload, f, indent=2)

        print(f"[logger] JSON saved → {filepath}  ({len(self._rows)} rows)")
        return filepath

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the logged trajectory."""
        if not self._rows:
            print("[logger] No data logged.")
            return

        t_arr    = np.array([r['t']      for r in self._rows])
        x_arr    = np.array([r['x']      for r in self._rows])
        th1_arr  = np.array([r['th1']    for r in self._rows])
        th2_arr  = np.array([r['th2']    for r in self._rows])
        u_arr    = np.array([r['u']      for r in self._rows])
        E_arr    = np.array([r['E_total'] for r in self._rows])
        KE_arr   = np.array([r['KE']     for r in self._rows])

        E_drift = abs(E_arr[-1] - E_arr[0])
        E_drift_pct = E_drift / max(abs(E_arr[0]), 1e-12) * 100

        print()
        print("=" * 52)
        print("  SIMULATION LOG SUMMARY")
        print("=" * 52)
        print(f"  Steps logged   : {len(self._rows)}")
        print(f"  Duration       : {t_arr[0]:.3f} → {t_arr[-1]:.3f} s")
        print(f"  Timestep (avg) : {np.mean(np.diff(t_arr)):.5f} s")
        print()
        print(f"  Cart x         : [{x_arr.min():+.4f},  {x_arr.max():+.4f}] m")
        print(f"  θ₁             : [{np.degrees(th1_arr.min()):+7.2f}°, "
              f"{np.degrees(th1_arr.max()):+7.2f}°]")
        print(f"  θ₂             : [{np.degrees(th2_arr.min()):+7.2f}°, "
              f"{np.degrees(th2_arr.max()):+7.2f}°]")
        print()
        print(f"  Control force u: [{u_arr.min():+.3f},  {u_arr.max():+.3f}] N")
        print(f"  Peak |u|       : {np.abs(u_arr).max():.3f} N")
        print()
        print(f"  E(t=0)         : {E_arr[0]:+.4f} J")
        print(f"  E(t=end)       : {E_arr[-1]:+.4f} J")
        print(f"  Energy drift   : {E_drift:.4f} J  ({E_drift_pct:.4f}%)")
        print(f"  Peak KE        : {KE_arr.max():.4f} J")
        print("=" * 52)
        print()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _potential(self, state) -> float:
        """Compute PE from state (avoids re-importing dynamics internals)."""
        x, th1, th2, *_ = state
        p = self._params
        phi2 = th1 + th2
        return (p['m1'] * p['g'] * (p['l1'] / 2) * np.cos(th1)
              + p['m2'] * p['g'] * (p['l1'] * np.cos(th1)
                                  + (p['l2'] / 2) * np.cos(phi2)))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"StateLogger({len(self._rows)} steps logged)"

    @property
    def times(self) -> np.ndarray:
        return np.array([r['t'] for r in self._rows])

    @property
    def states(self) -> np.ndarray:
        cols = ['x', 'th1', 'th2', 'xd', 'th1d', 'th2d']
        return np.array([[r[c] for c in cols] for r in self._rows])

    @property
    def u_log(self) -> np.ndarray:
        return np.array([r['u'] for r in self._rows])

    @property
    def energy(self) -> np.ndarray:
        """Returns (N, 3) array of [KE, PE, E_total] per timestep."""
        return np.array([[r['KE'], r['PE'], r['E_total']] for r in self._rows])
