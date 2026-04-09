"""
visualizer.py — Real-time 2D animation and energy plotting (Milestone 2)

Displays the cart-double pendulum system in a dark-themed figure with:
  • Main panel  — cart (rectangle) sliding on a track, two pendulum links,
                  joint markers, and a force indicator arrow
  • Energy panel — rolling KE / PE / Total energy time series
  • Info panel   — live state readout (x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂, u, energies)

Usage
-----
    from dynamics import load_params, simulate
    from visualizer import CartPendulumVisualizer, demo

    params = load_params()
    state0 = [0, 0.3, 0.1, 0, 0, 0]          # upright, slightly displaced

    times, states, u_log = simulate(state0, (0, 5), params['dt'],
                                    lambda t, s: 0.0, params)

    viz = CartPendulumVisualizer(params)
    viz.replay(times, states, u_log)

    # Or just call demo() for a quick test:
    demo()
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from dynamics import total_energy, load_params, simulate


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_BG_DARK   = '#0d1117'
_BG_PANEL  = '#161b22'
_COL_CART  = '#58a6ff'
_COL_LINK1 = '#f78166'
_COL_LINK2 = '#56d364'
_COL_FORCE_POS = '#ff7b72'   # rightward force
_COL_FORCE_NEG = '#79c0ff'   # leftward force
_COL_KE    = '#58a6ff'
_COL_PE    = '#f78166'
_COL_TOT   = '#56d364'
_COL_TEXT  = '#c9d1d9'
_COL_DIM   = '#8b949e'
_COL_TRACK = '#30363d'


class CartPendulumVisualizer:
    """
    Animates a pre-computed cart-double pendulum trajectory.

    Parameters
    ----------
    params       : dict   — system parameters (from load_params())
    playback_fps : int    — target display frame rate (default 50)
    """

    def __init__(self, params: dict, playback_fps: int = 50):
        self.params = params
        self.fps    = playback_fps
        self._reach = params['l1'] + params['l2']

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay(self, times, states, u_log=None) -> animation.FuncAnimation:
        """
        Animate a pre-computed trajectory from dynamics.simulate().

        Parameters
        ----------
        times  : np.ndarray (N,)
        states : np.ndarray (N, 6)  [x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]
        u_log  : np.ndarray (N,) or None — control force at each step

        Returns
        -------
        FuncAnimation object (keep a reference to prevent garbage collection)
        """
        times  = np.asarray(times)
        states = np.asarray(states)
        u_log  = np.zeros(len(times)) if u_log is None else np.asarray(u_log)

        # Subsample to target playback fps
        sim_dt = float(times[1] - times[0]) if len(times) > 1 else 0.005
        stride = max(1, round(1.0 / (self.fps * sim_dt)))
        idx    = np.arange(0, len(times), stride)
        t_p, s_p, u_p = times[idx], states[idx], u_log[idx]

        # Pre-compute energies for all displayed frames
        KE, PE, E_tot = self._compute_energies(s_p)

        return self._build_and_run(t_p, s_p, u_p, KE, PE, E_tot)

    def run_interactive(self, state0=None, lqr_ctrl=None, lqr_active_start=False,
                        swingup_ctrl=None) -> animation.FuncAnimation:
        """
        Live interactive simulation: click-drag the cart with the mouse.

        Controls (mouse drag / LQR toggle mode)
        ----------------------------------------
        Left-click + drag   — cart follows your mouse x-position
        Release             — cart holds at its current position
        Right-click / R     — reset pendulum to hanging-at-rest  [0, π, 0, 0, 0, 0]
        U                   — reset to near-upright test position [0, 0.15, 0.10, 0, 0, 0]
        Space               — pause / resume simulation
        L                   — toggle LQR controller on/off (only if lqr_ctrl provided)

        Additional controls (swing-up mode, when swingup_ctrl provided)
        -----------------------------------------------------------------
        G / Enter           — trigger swing-up + stabilization sequence
        R / Right-click     — reset to hanging at rest and cancel swing-up

        Parameters
        ----------
        state0        : array-like (6,) or None
        lqr_ctrl      : callable or None   — standalone LQR toggle (L key)
        lqr_active_start : bool            — start with LQR already active
        swingup_ctrl  : object or None     — controller with .trigger(), .reset(),
                                             .phase, and .control_fn(t, state)
        """
        from collections import deque
        from dynamics import rk4_step, cart_accel_to_force

        params = self.params
        dt     = params['dt']

        if state0 is None:
            state0 = np.array([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])

        # ── Mutable simulation state (shared between callbacks) ─────
        state     = [np.asarray(state0, dtype=float)]
        t_sim     = [0.0]
        x_target  = [float(state0[0])]   # where the cart is trying to go
        u_now      = [0.0]
        dragging   = [False]
        paused     = [False]
        lqr_active = [bool(lqr_active_start and lqr_ctrl is not None)]
        u_max_ref  = [1.0]                # auto-scales the force arrow

        # Rolling energy history — last E_WIN seconds at display fps
        E_WIN = 6.0
        HIST  = int(E_WIN * self.fps)
        h_t   = deque(maxlen=HIST)
        h_KE  = deque(maxlen=HIST)
        h_PE  = deque(maxlen=HIST)
        h_E   = deque(maxlen=HIST)

        # Physics steps per display frame (keeps simulation real-time)
        spf = max(1, round(1.0 / (self.fps * dt)))

        # ── PD position controller (cart tracks x_target) ───────────
        # Uses inverse dynamics so pendulum coupling forces are accounted for.
        Kp    = 200.0   # N/m  — stiff enough to track mouse closely
        Kd    = 40.0    # N·s/m
        A_MAX = 80.0    # m/s² acceleration clip before converting to force

        def compute_u(s):
            # Swing-up controller takes priority when active
            if swingup_ctrl is not None and swingup_ctrl.phase != 'idle':
                return float(swingup_ctrl.control_fn(t_sim[0], s))
            if lqr_active[0] and lqr_ctrl is not None:
                return float(lqr_ctrl(t_sim[0], s))
            x, _, _, xd, _, _ = s
            a_des = float(np.clip(
                Kp * (x_target[0] - x) + Kd * (0.0 - xd),
                -A_MAX, A_MAX,
            ))
            return cart_accel_to_force(s, a_des, params)

        # ── Figure ───────────────────────────────────────────────────
        r  = self._reach
        l1 = params['l1']
        cw = max(0.20 * l1, 0.04)
        ch = max(0.10 * l1, 0.02)

        fig = plt.figure(figsize=(14, 7), facecolor=_BG_DARK)
        gs  = gridspec.GridSpec(
            2, 2,
            width_ratios=[3, 1], height_ratios=[3, 1],
            hspace=0.40, wspace=0.30,
            left=0.05, right=0.97, top=0.94, bottom=0.08,
        )
        ax_anim   = fig.add_subplot(gs[0, 0])
        ax_energy = fig.add_subplot(gs[1, 0])
        ax_info   = fig.add_subplot(gs[:, 1])

        for ax in (ax_anim, ax_energy, ax_info):
            ax.set_facecolor(_BG_PANEL)
            ax.tick_params(colors=_COL_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(_COL_TRACK)

        if swingup_ctrl is not None:
            _title = 'Cart — Double Pendulum   [G: swing up  |  R: reset  |  Space: pause]'
        elif lqr_ctrl is not None:
            _title = 'Cart — Double Pendulum   [drag: move  |  R: hang  |  U: upright  |  L: LQR  |  Space: pause]'
        else:
            _title = 'Cart — Double Pendulum   [drag: move  |  R: hang  |  U: upright  |  Space: pause]'
        fig.suptitle(_title, color=_COL_TEXT, fontsize=11, fontweight='bold')

        x_half = r * 1.6 + cw
        y_lo   = -(r + ch) * 1.1
        y_hi   =  (r + ch) * 1.1

        ax_anim.set_ylim(y_lo, y_hi)
        ax_anim.set_xlabel('x  (m)', color=_COL_DIM, fontsize=8)
        ax_anim.set_ylabel('y  (m)', color=_COL_DIM, fontsize=8)
        ax_anim.tick_params(colors=_COL_DIM)
        ax_anim.axhline(0, color=_COL_TRACK, lw=1.5, zorder=0)

        # ── Artists ──────────────────────────────────────────────────
        cart_patch = mpatches.FancyBboxPatch(
            (-cw/2, -ch/2), cw, ch,
            boxstyle='round,pad=0.005',
            linewidth=1.5, edgecolor=_COL_CART, facecolor='#1f3a5f', zorder=4,
        )
        ax_anim.add_patch(cart_patch)

        # Dotted vertical line showing where the mouse is targeting
        target_vline = ax_anim.axvline(
            x_target[0], color='#ffffff28', lw=1.5, ls=':', zorder=1,
        )

        link1_line, = ax_anim.plot([], [], '-', lw=4, color=_COL_LINK1, zorder=5, solid_capstyle='round')
        link2_line, = ax_anim.plot([], [], '-', lw=3, color=_COL_LINK2, zorder=5, solid_capstyle='round')
        joint0_dot, = ax_anim.plot([], [], 'o', ms=8, color=_COL_CART,  zorder=6)
        joint1_dot, = ax_anim.plot([], [], 'o', ms=7, color=_COL_LINK1, zorder=6)
        tip2_dot,   = ax_anim.plot([], [], 'o', ms=5, color=_COL_LINK2, zorder=6)
        force_line, = ax_anim.plot([], [], '-', lw=4, color=_COL_FORCE_POS,
                                   zorder=3, alpha=0.85, solid_capstyle='butt')
        time_label  = ax_anim.text(
            0.02, 0.97, '', transform=ax_anim.transAxes,
            color=_COL_TEXT, fontsize=9, va='top', fontfamily='monospace',
        )
        mode_label = ax_anim.text(
            0.98, 0.97, 'DRAG', transform=ax_anim.transAxes,
            color=_COL_TEXT, fontsize=9, va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=_BG_PANEL, edgecolor=_COL_TRACK),
        )

        # Energy axes (live rolling window)
        ax_energy.set_xlabel('Time  (s)', color=_COL_DIM, fontsize=8)
        ax_energy.set_ylabel('Energy  (J)', color=_COL_DIM, fontsize=8)
        ax_energy.set_title('Mechanical Energy', color=_COL_TEXT, fontsize=9, pad=4)
        ke_line,  = ax_energy.plot([], [], color=_COL_KE,  lw=1.5, label='KE')
        pe_line,  = ax_energy.plot([], [], color=_COL_PE,  lw=1.5, label='PE')
        tot_line, = ax_energy.plot([], [], color=_COL_TOT, lw=2.0, label='Total')
        ax_energy.legend(loc='upper right', fontsize=7,
                         facecolor=_BG_DARK, edgecolor=_COL_TRACK, labelcolor=_COL_TEXT)

        # Info panel
        ax_info.axis('off')
        ax_info.set_title('State', color=_COL_TEXT, fontsize=10, pad=6)
        info_text = ax_info.text(
            0.08, 0.94, '', transform=ax_info.transAxes,
            color=_COL_TEXT, fontsize=9.5, va='top', fontfamily='monospace',
            linespacing=1.7,
        )
        if swingup_ctrl is not None:
            _help = 'G / Enter → swing up\nR / R-click → reset\nSpace → pause/resume'
        elif lqr_ctrl is not None:
            _help = 'Drag   → move cart\nR / R-click → hang\nU      → near-upright\nSpace  → pause/resume\nL      → toggle LQR'
        else:
            _help = 'Drag   → move cart\nR / R-click → hang\nU      → near-upright\nSpace  → pause/resume'
        ax_info.text(
            0.08, 0.05, _help,
            transform=ax_info.transAxes,
            color=_COL_DIM, fontsize=8, va='bottom', fontstyle='italic',
        )
        legend_patches = [
            mpatches.Patch(color=_COL_LINK1, label=f'Link 1  (l={params["l1"]} m)'),
            mpatches.Patch(color=_COL_LINK2, label=f'Link 2  (l={params["l2"]} m)'),
            mpatches.Patch(color=_COL_FORCE_POS, label='Force (+)'),
            mpatches.Patch(color=_COL_FORCE_NEG, label='Force (−)'),
        ]
        ax_info.legend(handles=legend_patches, loc='lower left', fontsize=8,
                       facecolor=_BG_DARK, edgecolor=_COL_TRACK,
                       labelcolor=_COL_TEXT, framealpha=0.9)

        # ── Shared reset helper ──────────────────────────────────────
        def _reset(target_state):
            state[0]     = np.array(target_state, dtype=float)
            x_target[0]  = float(target_state[0])
            t_sim[0]     = 0.0
            u_now[0]     = 0.0
            u_max_ref[0] = 1.0
            h_t.clear(); h_KE.clear(); h_PE.clear(); h_E.clear()

        # ── Keyboard event handler ───────────────────────────────────
        def on_key_press(event):
            if event.key == ' ':
                paused[0] = not paused[0]
            elif event.key in ('r', 'R'):
                paused[0]     = False
                lqr_active[0] = False
                if swingup_ctrl is not None:
                    swingup_ctrl.reset()
                _reset([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
            elif event.key in ('u', 'U') and swingup_ctrl is None:
                paused[0] = False
                _reset([0.0, 0.15, 0.10, 0.0, 0.0, 0.0])
            elif event.key in ('l', 'L') and lqr_ctrl is not None:
                lqr_active[0] = not lqr_active[0]
                if lqr_active[0]:
                    dragging[0] = False
            elif event.key in ('g', 'G', 'return') and swingup_ctrl is not None:
                swingup_ctrl.trigger()

        # ── Mouse event handlers ─────────────────────────────────────
        def on_motion(event):
            if event.inaxes is ax_anim and event.xdata is not None:
                target_vline.set_xdata([event.xdata, event.xdata])
                if dragging[0]:
                    x_target[0] = float(event.xdata)

        def on_press(event):
            if event.inaxes is not ax_anim or event.xdata is None:
                return
            if event.button == 3:                          # right-click → reset to hanging
                paused[0]     = False
                lqr_active[0] = False
                if swingup_ctrl is not None:
                    swingup_ctrl.reset()
                _reset([0.0, np.pi, 0.0, 0.0, 0.0, 0.0])
            else:                                          # left-click → drag
                dragging[0] = True
                x_target[0] = float(event.xdata)

        def on_release(event):
            dragging[0] = False
            # Hold cart at wherever it ended up
            x_target[0] = float(state[0][0])

        fig.canvas.mpl_connect('motion_notify_event',  on_motion)
        fig.canvas.mpl_connect('button_press_event',   on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('key_press_event',      on_key_press)

        # ── Animation update (runs every display frame) ──────────────
        def update(_frame):
            # Step physics forward — skip when paused
            if not paused[0]:
                for _ in range(spf):
                    u         = compute_u(state[0])
                    state[0]  = rk4_step(state[0], t_sim[0], dt, u, params)
                    t_sim[0] += dt
                u_now[0] = u

            # Mode badge
            if paused[0]:
                mode_label.set_text('PAUSED')
                mode_label.set_color('#f0c040')
            elif swingup_ctrl is not None and swingup_ctrl.phase == 'swingup':
                mode_label.set_text('SWING')
                mode_label.set_color('#ff9500')
            elif ((swingup_ctrl is not None and swingup_ctrl.phase == 'lqr')
                  or lqr_active[0]):
                mode_label.set_text(' LQR ')
                mode_label.set_color(_COL_LINK2)
            elif swingup_ctrl is not None:
                mode_label.set_text('IDLE')
                mode_label.set_color(_COL_DIM)
            else:
                mode_label.set_text(' DRAG ')
                mode_label.set_color(_COL_TEXT)

            # Energy
            E_val  = total_energy(state[0], params)
            PE_val = self._pe(state[0])
            KE_val = E_val - PE_val
            h_t.append(t_sim[0])
            h_KE.append(KE_val)
            h_PE.append(PE_val)
            h_E.append(E_val)

            s  = state[0]
            xc = float(s[0])
            pivot, tip1, tip2 = self._link_endpoints(s)

            # Scroll viewport
            ax_anim.set_xlim(xc - x_half, xc + x_half)

            cart_patch.set_x(xc - cw / 2)

            # Links and joints
            link1_line.set_data([pivot[0], tip1[0]], [pivot[1], tip1[1]])
            link2_line.set_data([tip1[0],  tip2[0]], [tip1[1],  tip2[1]])
            joint0_dot.set_data([pivot[0]], [pivot[1]])
            joint1_dot.set_data([tip1[0]],  [tip1[1]])
            tip2_dot.set_data([tip2[0]],    [tip2[1]])

            # Force arrow (drawn at mid car-body height)
            u_max_ref[0] = max(u_max_ref[0], abs(u_now[0]))
            f_len = (u_now[0] / u_max_ref[0]) * r * 0.45
            force_line.set_data([xc, xc + f_len], [-ch * 0.7, -ch * 0.7])
            force_line.set_color(_COL_FORCE_POS if u_now[0] >= 0 else _COL_FORCE_NEG)

            time_label.set_text(f't = {t_sim[0]:.2f} s')

            # Rolling energy plot
            t_arr = np.array(h_t)
            if len(t_arr) > 1:
                ax_energy.set_xlim(t_arr[0], t_arr[-1] + 0.1)
                all_e = list(h_KE) + list(h_PE) + list(h_E)
                e_lo_v, e_hi_v = min(all_e), max(all_e)
                pad = max((e_hi_v - e_lo_v) * 0.15, 0.5)
                ax_energy.set_ylim(e_lo_v - pad, e_hi_v + pad)
            ke_line.set_data(t_arr, np.array(h_KE))
            pe_line.set_data(t_arr, np.array(h_PE))
            tot_line.set_data(t_arr, np.array(h_E))

            # Info readout
            x_s, th1, th2, xd, th1d, th2d = s
            info = (
                f"t    = {t_sim[0]:>8.2f} s\n"
                f"\n"
                f"x    = {x_s:>+8.4f} m\n"
                f"ẋ    = {xd:>+8.4f} m/s\n"
                f"\n"
                f"θ₁   = {np.degrees(th1) % 360:>8.2f} °\n"
                f"θ̇₁   = {th1d:>+8.4f} r/s\n"
                f"\n"
                f"θ₂   = {np.degrees(th2) % 360:>8.2f} °\n"
                f"θ̇₂   = {th2d:>+8.4f} r/s\n"
                f"\n"
                f"u    = {u_now[0]:>+8.2f} N\n"
                f"x_t  = {x_target[0]:>+8.4f} m\n"
                f"\n"
                f"KE   = {KE_val:>+8.4f} J\n"
                f"PE   = {PE_val:>+8.4f} J\n"
                f"E    = {E_val:>+8.4f} J"
            )
            info_text.set_text(info)

        anim = animation.FuncAnimation(
            fig, update,
            interval=int(1000 / self.fps),
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
        return anim

    def save(self, times, states, u_log=None,
             filepath: str = 'animation.gif', fps: int = 20) -> None:
        """
        Save animation to a GIF or MP4 file.

        Parameters
        ----------
        filepath : str   — output path, extension determines format (.gif or .mp4)
        fps      : int   — frames per second in the saved file
        """
        anim = self.replay(times, states, u_log)
        writer = 'pillow' if str(filepath).endswith('.gif') else 'ffmpeg'
        anim.save(filepath, writer=writer, fps=fps,
                  savefig_kwargs={'facecolor': _BG_DARK})
        print(f"[visualizer] Saved → {filepath}")

    # ------------------------------------------------------------------
    # Internal — geometry
    # ------------------------------------------------------------------

    def _link_endpoints(self, state):
        """
        Compute Cartesian positions of cart pivot, link 1 tip, link 2 tip.

        Returns
        -------
        pivot : (2,) — cart pivot position [x, 0]
        tip1  : (2,) — end of link 1
        tip2  : (2,) — end of link 2
        """
        x, th1, th2 = state[0], state[1], state[2]
        l1, l2 = self.params['l1'], self.params['l2']
        pivot = np.array([x,   0.0])
        tip1  = np.array([x + l1 * np.sin(th1),   l1 * np.cos(th1)])
        tip2  = tip1 + np.array([l2 * np.sin(th1 + th2), l2 * np.cos(th1 + th2)])
        return pivot, tip1, tip2

    def _compute_energies(self, states):
        """Return (KE, PE, E_total) arrays, one value per frame."""
        E_tot = np.array([total_energy(s, self.params) for s in states])
        PE    = np.array([self._pe(s) for s in states])
        KE    = E_tot - PE
        return KE, PE, E_tot

    def _pe(self, state):
        x, th1, th2 = state[0], state[1], state[2]
        p = self.params
        phi2 = th1 + th2
        return (p['m1'] * p['g'] * (p['l1'] / 2) * np.cos(th1)
              + p['m2'] * p['g'] * (p['l1'] * np.cos(th1)
                                  + (p['l2'] / 2) * np.cos(phi2)))

    # ------------------------------------------------------------------
    # Internal — figure construction
    # ------------------------------------------------------------------

    def _build_and_run(self, times, states, u_log, KE, PE, E_tot):
        r      = self._reach
        l1, l2 = self.params['l1'], self.params['l2']
        cw = max(0.20 * l1, 0.04)
        ch = max(0.10 * l1, 0.02)

        # ── figure & axes ───────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 7), facecolor=_BG_DARK)
        gs  = gridspec.GridSpec(
            2, 2,
            width_ratios=[3, 1],
            height_ratios=[3, 1],
            hspace=0.40, wspace=0.30,
            left=0.05, right=0.97, top=0.94, bottom=0.08,
        )
        ax_anim   = fig.add_subplot(gs[0, 0])   # main animation
        ax_energy = fig.add_subplot(gs[1, 0])   # energy time series
        ax_info   = fig.add_subplot(gs[:, 1])   # state info panel

        for ax in (ax_anim, ax_energy, ax_info):
            ax.set_facecolor(_BG_PANEL)
            ax.tick_params(colors=_COL_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor(_COL_TRACK)

        fig.suptitle('Cart — Double Pendulum', color=_COL_TEXT,
                     fontsize=13, fontweight='bold')

        # ── animation axes ──────────────────────────────────────────────
        x_half = r * 1.6 + cw
        y_lo   = -(r + ch) * 1.1
        y_hi   =  (r + ch) * 1.1

        ax_anim.set_ylim(y_lo, y_hi)
        ax_anim.set_xlabel('x  (m)', color=_COL_DIM, fontsize=8)
        ax_anim.set_ylabel('y  (m)', color=_COL_DIM, fontsize=8)
        ax_anim.tick_params(colors=_COL_DIM)
        ax_anim.axhline(0, color=_COL_TRACK, lw=1.5, zorder=0)

        # Dashed vertical upright reference
        ax_anim.axvline(0, color='#ffffff14', lw=0.8, ls='--', zorder=0)

        # Animated artists ------------------------------------------------
        cart_patch = mpatches.FancyBboxPatch(
            (-cw / 2, -ch / 2), cw, ch,
            boxstyle='round,pad=0.005',
            linewidth=1.5, edgecolor=_COL_CART, facecolor='#1f3a5f', zorder=4,
        )
        ax_anim.add_patch(cart_patch)

        link1_line, = ax_anim.plot([], [], '-', lw=4, color=_COL_LINK1, zorder=5, solid_capstyle='round')
        link2_line, = ax_anim.plot([], [], '-', lw=3, color=_COL_LINK2, zorder=5, solid_capstyle='round')
        joint0_dot, = ax_anim.plot([], [], 'o', ms=8, color=_COL_CART,  zorder=6)
        joint1_dot, = ax_anim.plot([], [], 'o', ms=7, color=_COL_LINK1, zorder=6)
        tip2_dot,   = ax_anim.plot([], [], 'o', ms=5, color=_COL_LINK2, zorder=6)
        force_line, = ax_anim.plot([], [], '-', lw=4, color=_COL_FORCE_POS, zorder=3, alpha=0.85, solid_capstyle='butt')

        time_label = ax_anim.text(
            0.02, 0.97, '', transform=ax_anim.transAxes,
            color=_COL_TEXT, fontsize=9, va='top', fontfamily='monospace',
        )

        # ── energy axes ─────────────────────────────────────────────────
        ax_energy.set_xlim(times[0], times[-1])
        e_vals  = np.concatenate([KE, PE, E_tot])
        e_lo, e_hi = e_vals.min(), e_vals.max()
        e_pad   = max((e_hi - e_lo) * 0.15, 0.5)
        ax_energy.set_ylim(e_lo - e_pad, e_hi + e_pad)
        ax_energy.set_xlabel('Time  (s)', color=_COL_DIM, fontsize=8)
        ax_energy.set_ylabel('Energy  (J)', color=_COL_DIM, fontsize=8)
        ax_energy.set_title('Mechanical Energy', color=_COL_TEXT, fontsize=9, pad=4)

        ke_line,  = ax_energy.plot([], [], color=_COL_KE,  lw=1.5, label='KE')
        pe_line,  = ax_energy.plot([], [], color=_COL_PE,  lw=1.5, label='PE')
        tot_line, = ax_energy.plot([], [], color=_COL_TOT, lw=2.0, label='Total')
        t_cursor  = ax_energy.axvline(times[0], color='#ffffff40', lw=1)
        ax_energy.legend(
            loc='upper right', fontsize=7,
            facecolor=_BG_DARK, edgecolor=_COL_TRACK,
            labelcolor=_COL_TEXT,
        )

        # ── info panel ──────────────────────────────────────────────────
        ax_info.axis('off')
        ax_info.set_title('State', color=_COL_TEXT, fontsize=10, pad=6)
        info_text = ax_info.text(
            0.08, 0.94, '', transform=ax_info.transAxes,
            color=_COL_TEXT, fontsize=9.5, va='top', fontfamily='monospace',
            linespacing=1.7,
        )

        # Legend for link colours
        legend_patches = [
            mpatches.Patch(color=_COL_LINK1, label=f'Link 1  (l={l1} m)'),
            mpatches.Patch(color=_COL_LINK2, label=f'Link 2  (l={l2} m)'),
            mpatches.Patch(color=_COL_FORCE_POS, label='Force (+)'),
            mpatches.Patch(color=_COL_FORCE_NEG, label='Force (−)'),
        ]
        ax_info.legend(
            handles=legend_patches, loc='lower left',
            fontsize=8, facecolor=_BG_DARK, edgecolor=_COL_TRACK,
            labelcolor=_COL_TEXT, framealpha=0.9,
        )

        # Max force magnitude for arrow scaling
        u_max = max(float(np.abs(u_log).max()), 1e-6)
        arrow_max = r * 0.45   # maximum arrow length in metres

        # ── animation callbacks ─────────────────────────────────────────
        def init():
            link1_line.set_data([], [])
            link2_line.set_data([], [])
            joint0_dot.set_data([], [])
            joint1_dot.set_data([], [])
            tip2_dot.set_data([], [])
            force_line.set_data([], [])
            ke_line.set_data([], [])
            pe_line.set_data([], [])
            tot_line.set_data([], [])
            return (cart_patch, link1_line, link2_line,
                    joint0_dot, joint1_dot, tip2_dot,
                    force_line, ke_line, pe_line, tot_line,
                    t_cursor, time_label, info_text)

        def update(i):
            state = states[i]
            t     = float(times[i])
            u     = float(u_log[i])

            pivot, tip1, tip2 = self._link_endpoints(state)
            xc = float(pivot[0])

            # Scroll viewport to follow cart
            ax_anim.set_xlim(xc - x_half, xc + x_half)

            cart_patch.set_x(xc - cw / 2)

            # Links
            link1_line.set_data([pivot[0], tip1[0]], [pivot[1], tip1[1]])
            link2_line.set_data([tip1[0],  tip2[0]], [tip1[1],  tip2[1]])

            # Joints
            joint0_dot.set_data([pivot[0]], [pivot[1]])
            joint1_dot.set_data([tip1[0]],  [tip1[1]])
            tip2_dot.set_data([tip2[0]],   [tip2[1]])

            # Force arrow at mid car-body height
            f_len = (u / u_max) * arrow_max
            force_line.set_data([xc, xc + f_len], [-ch * 0.7, -ch * 0.7])
            force_line.set_color(_COL_FORCE_POS if u >= 0 else _COL_FORCE_NEG)

            # Time label
            time_label.set_text(f't = {t:.3f} s')

            # Energy traces
            ke_line.set_data(times[:i + 1], KE[:i + 1])
            pe_line.set_data(times[:i + 1], PE[:i + 1])
            tot_line.set_data(times[:i + 1], E_tot[:i + 1])
            t_cursor.set_xdata([t, t])

            # Info readout
            x_s, th1, th2, xd, th1d, th2d = state
            info = (
                f"t    = {t:>8.3f} s\n"
                f"\n"
                f"x    = {x_s:>+8.4f} m\n"
                f"ẋ    = {xd:>+8.4f} m/s\n"
                f"\n"
                f"θ₁   = {np.degrees(th1):>+8.2f} °\n"
                f"θ̇₁   = {th1d:>+8.4f} r/s\n"
                f"\n"
                f"θ₂   = {np.degrees(th2):>+8.2f} °\n"
                f"θ̇₂   = {th2d:>+8.4f} r/s\n"
                f"\n"
                f"u    = {u:>+8.3f} N\n"
                f"\n"
                f"KE   = {KE[i]:>+8.4f} J\n"
                f"PE   = {PE[i]:>+8.4f} J\n"
                f"E    = {E_tot[i]:>+8.4f} J"
            )
            info_text.set_text(info)

            return (cart_patch, link1_line, link2_line,
                    joint0_dot, joint1_dot, tip2_dot,
                    force_line, ke_line, pe_line, tot_line,
                    t_cursor, time_label, info_text)

        anim = animation.FuncAnimation(
            fig, update,
            frames=len(times),
            init_func=init,
            interval=int(1000 / self.fps),
            blit=False,
            repeat=True,
        )

        plt.show()
        return anim


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def demo_interactive():
    """Launch the interactive mouse-driven simulation (default entry point)."""
    params = load_params()
    viz = CartPendulumVisualizer(params)
    viz.run_interactive()


def demo(t_end: float = 6.0, th1_0: float = 2.5, th2_0: float = -1.2):
    """
    Quick demo: free-swing from a position where both links are clearly visible.

    Default angles — link 1 past horizontal (2.5 rad ≈ 143°), link 2 bent back
    (-1.2 rad) — so the two segments are visually distinct from frame 0.

    Parameters
    ----------
    t_end  : float  — simulation duration (s)
    th1_0  : float  — initial θ₁ from upright (rad); default 2.5 (≈143°)
    th2_0  : float  — initial θ₂ relative to link 1 (rad); default -1.2 (≈-69°)
    """
    params = load_params()
    state0 = np.array([0.0, th1_0, th2_0, 0.0, 0.0, 0.0])
    times, states, u_log = simulate(
        state0, (0.0, t_end), params['dt'],
        lambda t, s: 0.0,
        params,
    )
    viz = CartPendulumVisualizer(params)
    viz.replay(times, states, u_log)


if __name__ == '__main__':
    demo_interactive()
