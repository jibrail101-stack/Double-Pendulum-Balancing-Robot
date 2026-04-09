# Double Pendulum Balancing Robot

A physics-accurate 2D simulation of a cart-mounted double pendulum with a two-phase controller: an Åström-Furuta energy-pumping swing-up followed by LQR stabilisation at the upright unstable equilibrium.

---

## System Overview

```
          ● tip2
         /
        / link 2  (m2, l2)
       ●  joint 1
      /
     /  link 1  (m1, l1)
    ●  pivot (attached to cart)
   [====]  cart (M)
───────────────────────  track
        → u (control force)
```

**State vector:** `[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]`
- `x` — cart position along the track
- `θ₁` — link 1 angle from the upright vertical
- `θ₂` — link 2 angle relative to link 1
- Actuation: horizontal force `u` on the cart only (underactuated, 2 passive joints)

**Default parameters** (configurable in `config.yaml`):

| Component | Mass | Length |
|-----------|------|--------|
| Cart      | 1.0 kg | — |
| Link 1    | 1.5 kg | 1.5 m |
| Link 2    | 0.25 kg | 0.5 m |

---

## Features

- Full Lagrangian equations of motion derived symbolically with SymPy, evaluated numerically at runtime
- RK4 integrator at 200 Hz — energy drift < 0.001% over 30 s
- LQR stabiliser — continuous CARE solved via `scipy.linalg.solve_continuous_are`
- Swing-up controller — Åström-Furuta pendulum-only energy pump with auto-derived gains
- Force-gate LQR capture — prevents premature handoff at high-velocity passes
- Auto-tuner (`tune.py`) — derives all controller parameters analytically from physics; validates against FR-04/FR-05 with adaptive refinement
- Real-time interactive visualiser — mouse-drag cart, press G to swing up
- CSV/JSON state logger with energy breakdown
- All parameters externally configurable via `config.yaml`

---

## Installation

**Requirements:** Python 3.11+

```bash
git clone <repo-url>
cd double-pendulum-robot
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.26
scipy>=1.12
sympy>=1.12
matplotlib>=3.8
pyyaml>=6.0
pytest>=8.0
```

---

## Quick Start

```bash
# Interactive swing-up simulation (default)
python main.py

# Batch simulate swing-up, log to results/, replay animation
python main.py --mode swingup --t-end 30

# LQR stabiliser from ±10° perturbation
python main.py --mode lqr --t-end 35

# Free-swing energy conservation check
python main.py --mode free --t-end 5

# Full performance report (settling time, control effort, disturbance rejection)
python report.py

# Run unit tests
pytest tests/
```

---

## Usage

```
python main.py [--mode MODE] [--config PATH] [--t-end SECS]
               [--save PATH] [--no-viz] [--tune] [--log-dir DIR]

Modes:
  interactive   Live simulation — drag cart, G to swing up (default)
  swingup       Batch simulate full swing-up sequence, then replay
  lqr           Batch simulate LQR from small perturbation, replay
  free          Batch simulate free-swing, energy conservation check

Options:
  --config      Path to YAML config (default: config.yaml)
  --t-end       Simulation duration in seconds for batch modes (default: 30)
  --save PATH   Save animation to .gif or .mp4 (batch modes)
  --no-viz      Headless: log and print only, no animation window
  --tune        Print auto-tuner parameter report before running
  --log-dir     Output directory for CSV/JSON logs (default: results/)
```

### Interactive controls

| Key / Action | Effect |
|---|---|
| Left-click + drag | Move cart to mouse position |
| **G** or Enter | Trigger swing-up → LQR stabilisation |
| **R** or right-click | Reset to hanging at rest |
| **Space** | Pause / resume |

---

## Physics Background

### Equations of Motion

The system has three degrees of freedom: cart position `x`, link 1 angle `θ₁`, and link 2 relative angle `θ₂`. The Lagrangian `L = T − V` gives:

```
M(q) q̈ + C(q, q̇) q̇ + G(q) = B u
```

where `M(q)` is the 3×3 configuration-dependent mass matrix, `C` captures Coriolis/centrifugal terms, `G` is the gravity vector, and `B = [1, 0, 0]ᵀ` (force acts only on cart). The EOM are derived symbolically using SymPy and lambdified for fast numerical evaluation.

### LQR Stabiliser

At the upright equilibrium `(θ₁, θ₂) = (0, 0)` the nonlinear system is linearised via numerical finite differences:

```
ẋ = A x + B u
```

The continuous-time algebraic Riccati equation is solved for the optimal cost-to-go matrix `P`, giving gain `K = R⁻¹ BᵀP` and control law `u = −K state`.

The cart position `x` is left unpenalised in `Q` — the cart translates freely while the links are balanced.

### Swing-Up Controller (Åström-Furuta)

The energy pump drives the pendulum-only mechanical energy toward the upright target value `E_target`:

```
u = k_sw · ẋ · (E_target − E_pend)
```

where `E_pend` is the kinetic + potential energy of both links (cart KE excluded). A short initial kick (`u = 0.4 · u_max` for ~0.09 s) breaks the hanging equilibrium before the pump takes over.

**Auto-derived parameters:**
```
u_max    = 4 · (M + m1 + m2) · g
k_sw     = 4 · u_max / (√(2g·l1) · E_target)
kick_dur = 0.08 · T_nat   (clamped 0.05 – 0.5 s)
```

**Force-gate capture:** LQR takes over only when both `|θ₁|, |θ₂| ≤ 30°` AND the predicted LQR force is `≤ 3 · u_max`. This prevents capture at high-velocity passes where the required restoring force would immediately diverge.

**Geometry constraint:** `l2/l1 ≥ 0.2` is required for reliable swing-up. Below this ratio the second link's rotational inertia is too small for the energy pump to control.

---

## Project Structure

```
double-pendulum-robot/
├── main.py            Entry point — all modes, CLI argument parser
├── dynamics.py        Symbolic EOM (SymPy), RK4 integrator, load_params()
├── controller_lqr.py  LQRController, SwingUpLQRController
├── visualizer.py      CartPendulumVisualizer — real-time 2D animation
├── logger.py          StateLogger — CSV/JSON trajectory logging
├── tune.py            Analytical auto-tuner with simulation validation
├── report.py          Performance report — FR metrics, disturbance tests
├── update_config.py   Recomputes derived fields in config.yaml
├── config.yaml        All physical parameters + parameter limits
├── requirements.txt
└── tests/
    ├── test_dynamics.py    Unit tests: EOM, energy conservation, RK4
    └── test_maneuver.py    Integration tests: swing-up, LQR FR-04/FR-05
```

---

## Configuration

Edit `config.yaml` to change physical parameters. Run `python update_config.py` after changes to refresh the derived fields (I₁, I₂, sample_hz).

```yaml
cart:
  mass: 1.0        # M (kg)  — must be ≥ m1+m2 for stable dynamics

link1:
  mass: 1.5        # m1 (kg)
  length: 1.5      # l1 (m)  — sets system scale

link2:
  mass: 0.25       # m2 (kg)
  length: 0.5      # l2 (m)  — CRITICAL: l2/l1 must be ≥ 0.2 for swing-up

physics:
  g: 9.81          # m/s²

integrator:
  dt: 0.005        # timestep (s) — keep ≤ 0.010 s (≥ 100 Hz, FR-02)
```

---

## Performance Results

| Metric | Result | Requirement |
|--------|--------|-------------|
| Swing-up + capture | < 2 s | FR-04: ≤ 15 s |
| Settled (±5°) | ~15 s | FR-04: ≤ 15 s |
| θ₁ max (last 30 s) | 0.056° | FR-05: ≤ 5° |
| θ₂ max (last 30 s) | 0.002° | FR-05: ≤ 5° |
| Energy drift (5 s) | 0.0000% | NFR-02: stable |
| Simulation rate | 200 Hz | NFR-02: ≥ 100 Hz |

See `python report.py` for full metrics including control effort and disturbance rejection.

---

## References

- Åström & Furuta (2000) — *Swinging up a pendulum by energy control*
- Tedrake (MIT) — *Underactuated Robotics*, Chapters 3, 7
- Spong (1995) — *The Swing Up Control Problem for the Acrobot*
- SciPy `solve_continuous_are` documentation
