"""
avatar_viz.py
Animated walking avatar with real-time plantar pressure visualization.
Saves: data/gait_avatar.gif  (seamlessly looping)

Gait cycle convention (LEFT leg):
  t = 0.00 → left heel strike  (hip maximally forward)
  t = 0.30 → midstance
  t = 0.60 → toe-off           (hip maximally backward)
  t = 0.60–1.0 → swing phase   (knee flexes, then extends before next strike)
Right leg is phase-offset by 0.5.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks, resample as sp_resample
from pathlib import Path

# ─── Visual style ──────────────────────────────────────────────────────────────
BG     = '#0d1117'
SKIN   = '#c8d8e8'
DARK   = '#7a9ab0'
WHITE  = '#ffffff'
L_COL  = '#29b6f6'   # left  — cyan-blue
R_COL  = '#ef5350'   # right — coral-red

# ─── Body dimensions (normalised units) ───────────────────────────────────────
HEAD_R      = 0.090
TORSO_H     = 0.340
TORSO_W     = 0.105
UPPER_ARM   = 0.215
LOWER_ARM   = 0.175
UPPER_LEG   = 0.360
LOWER_LEG   = 0.325
FOOT_LEN    = 0.155
HIP_HALF    = 0.055
SHOULDER_X  = 0.075
PELVIS_Y    = 0.82      # height of pelvis above ground
GROUND_Y    = 0.00

# ─── Gait kinematics ──────────────────────────────────────────────────────────
# All angles in degrees; positive = forward (anterior) direction.
# Functions of phase t ∈ [0, 1).

def _hip(t):
    """Hip flexion/extension.  +forward at t=0 (heel strike)."""
    return 22.0 * np.cos(2 * np.pi * t)

def _knee(t):
    """Knee flexion — always ≥ 0 (never hyper-extends).
    Two peaks: small loading response (~t=0.07), large swing (~t=0.79)."""
    return np.maximum(0.0,
        9.0  * np.exp(-((t - 0.07) / 0.06) ** 2) +   # loading response
        58.0 * np.exp(-((t - 0.79) / 0.09) ** 2))     # mid-swing

def _ankle(t):
    """Ankle: plantarflexion at push-off (+), dorsiflexion at midstance (−)."""
    return (17.0 * np.exp(-((t - 0.57) / 0.10) ** 2) -
             8.0 * np.exp(-((t - 0.33) / 0.14) ** 2))

def _arm_angle(t):
    """Arm swing angle from vertical; arms swing opposite to ipsilateral leg.
    Pass t = contralateral leg phase (= ipsilateral + 0.5)."""
    return 17.0 * np.cos(2 * np.pi * t)   # mirrors hip


# ─── Forward kinematics ────────────────────────────────────────────────────────

def leg_fk(hip_xy, phase):
    """Compute (hip, knee, ankle, heel, toe) for one leg at gait phase."""
    ha = np.radians(_hip(phase))
    kf = np.radians(_knee(phase))
    af = np.radians(_ankle(phase))

    hip   = np.asarray(hip_xy, float)
    knee  = hip  + UPPER_LEG * np.array([ np.sin(ha), -np.cos(ha)])
    la    = ha - kf                                 # lower-leg angle from vertical
    ankle = knee + LOWER_LEG * np.array([ np.sin(la), -np.cos(la)])
    fa    = la - np.pi / 7.5 + af                  # foot angle
    toe   = ankle + FOOT_LEN       * np.array([ np.cos(fa), -np.sin(fa)])
    heel  = ankle + FOOT_LEN * 0.3 * np.array([-np.cos(fa),  np.sin(fa)])
    return dict(hip=hip, knee=knee, ankle=ankle, heel=heel, toe=toe)


def arm_fk(sh_xy, phase):
    """Compute (shoulder, elbow, wrist) for one arm at given phase."""
    aa    = np.radians(_arm_angle(phase))
    sh    = np.asarray(sh_xy, float)
    elbow = sh    + UPPER_ARM * np.array([np.sin(aa),       -np.cos(aa)])
    wrist = elbow + LOWER_ARM * np.array([np.sin(aa * 0.55), -np.cos(aa * 0.55)])
    return dict(shoulder=sh, elbow=elbow, wrist=wrist)


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def _seg(ax, a, b, color, lw, zo=2, alpha=1.0):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
            solid_capstyle='round', solid_joinstyle='round',
            alpha=alpha, zorder=zo)


def _dot(ax, pos, r, color, zo=3):
    ax.add_patch(mpatches.Circle(pos, r, color=color, zorder=zo))


def _dk(hex_color, f=0.50):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * f, g * f, b * f)


def draw_pressure_glow(ax, foot, fnorm, color):
    """Multi-layer elliptical glow on the ground proportional to force."""
    if fnorm < 0.02:
        return
    cx = (foot['heel'][0] + foot['toe'][0]) * 0.5
    cy = GROUND_Y - 0.012
    for alpha, scale in [(0.07, 3.8), (0.13, 2.6), (0.24, 1.7), (0.40, 1.1)]:
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 0.30 * scale * (0.6 + 0.4 * fnorm),
            0.072 * scale * fnorm,
            color=color, alpha=alpha * fnorm, zorder=1))


def draw_force_bar(ax, x, fnorm, color, label):
    """Vertical force indicator bar (VU-meter style)."""
    track_h = 0.68
    bar_w   = 0.075
    bar_h   = fnorm * track_h

    # Track
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - bar_w / 2, 0.06), bar_w, track_h,
        boxstyle='round,pad=0.01',
        facecolor='#131d28', edgecolor='#243040', lw=0.8, zorder=3))

    # Gradient-like fill: stack thin rects from green→yellow→red
    if bar_h > 0.005:
        n_segs = 30
        seg_h  = bar_h / n_segs
        for k in range(n_segs):
            frac = k / n_segs
            c    = mcolors.to_rgb(color)
            alpha = 0.55 + 0.45 * frac
            ax.add_patch(mpatches.Rectangle(
                (x - bar_w / 2 + 0.004, 0.06 + k * seg_h),
                bar_w - 0.008, seg_h,
                facecolor=c, alpha=alpha, zorder=4))

    # Force value label
    ax.text(x, 0.06 + bar_h + 0.035, f'{fnorm * 700:.0f} N',
            ha='center', va='bottom', fontsize=7.5,
            color=color, fontweight='bold', zorder=5)
    ax.text(x, 0.025, label,
            ha='center', va='center', fontsize=7.5,
            color=color, fontweight='bold', zorder=5)


def draw_avatar(ax, phase, lf_fn, rf_fn):
    """Draw one complete avatar frame."""
    l_phase = phase
    r_phase = (phase + 0.5) % 1.0

    # Key body anchor points
    pelvis   = np.array([0.0, PELVIS_Y])
    torso_t  = pelvis  + np.array([0.0, TORSO_H])
    neck_t   = torso_t + np.array([0.0, 0.055])
    head_c   = neck_t  + np.array([0.0, HEAD_R])
    lhip     = pelvis  + np.array([-HIP_HALF, 0.0])
    rhip     = pelvis  + np.array([ HIP_HALF, 0.0])
    lsh      = torso_t + np.array([-SHOULDER_X, -0.02])
    rsh      = torso_t + np.array([ SHOULDER_X, -0.02])

    # Joint positions
    lf = leg_fk(lhip, l_phase)
    rf = leg_fk(rhip, r_phase)
    # Arms: each arm swings with the CONTRALATERAL leg
    la = arm_fk(lsh, r_phase)   # left arm mirrors right leg
    ra = arm_fk(rsh, l_phase)   # right arm mirrors left leg

    # Determine which leg is more forward (higher hip angle = more forward)
    l_forward = _hip(l_phase) >= _hip(r_phase)
    back_leg, front_leg   = (rf, lf)   if l_forward else (lf, rf)
    back_col, front_col   = (_dk(R_COL), L_COL) if l_forward else (_dk(L_COL), R_COL)
    back_arm, front_arm   = (ra, la)   if l_forward else (la, ra)

    # ── Glows ──────────────────────────────────────────────────────────────
    draw_pressure_glow(ax, lf, lf_fn, L_COL)
    draw_pressure_glow(ax, rf, rf_fn, R_COL)

    # ── Ground shadow ───────────────────────────────────────────────────────
    ax.fill_between([-1.6, 1.6], GROUND_Y - 0.018, GROUND_Y,
                    color='#121c26', zorder=1)

    # ── Back leg ────────────────────────────────────────────────────────────
    _seg(ax, back_leg['hip'],   back_leg['knee'],  back_col, 9,  zo=2)
    _seg(ax, back_leg['knee'],  back_leg['ankle'], back_col, 8,  zo=2)
    _seg(ax, back_leg['heel'],  back_leg['toe'],   back_col, 7,  zo=2)
    _dot(ax, back_leg['knee'],  0.026, _dk(WHITE), zo=2)
    _dot(ax, back_leg['ankle'], 0.019, _dk(WHITE), zo=2)

    # ── Back arm ────────────────────────────────────────────────────────────
    _seg(ax, back_arm['shoulder'], back_arm['elbow'], _dk(SKIN), 7, zo=2, alpha=0.70)
    _seg(ax, back_arm['elbow'],    back_arm['wrist'],  _dk(SKIN), 5, zo=2, alpha=0.70)

    # ── Torso ───────────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (pelvis[0] - TORSO_W, pelvis[1] + 0.028), TORSO_W * 2, TORSO_H - 0.02,
        boxstyle='round,pad=0.032',
        facecolor=SKIN, edgecolor='none', zorder=4))

    # Pelvis connector
    ax.add_patch(mpatches.Ellipse(
        (pelvis[0], pelvis[1] + 0.01), HIP_HALF * 2 + 0.08, 0.10,
        color=SKIN, zorder=4))

    # ── Neck + head ─────────────────────────────────────────────────────────
    _seg(ax, torso_t, neck_t, SKIN, 8, zo=4)
    ax.add_patch(mpatches.Circle(head_c, HEAD_R, color=SKIN, zorder=5))
    # Eye
    ax.plot(head_c[0] + HEAD_R * 0.46, head_c[1] + HEAD_R * 0.12,
            'o', color=BG, markersize=3.8, zorder=6)
    # Ear
    ax.add_patch(mpatches.Ellipse(
        (head_c[0] - HEAD_R * 0.95, head_c[1] - HEAD_R * 0.05),
        0.022, 0.038, color=DARK, zorder=5))

    # ── Front arm ───────────────────────────────────────────────────────────
    _seg(ax, front_arm['shoulder'], front_arm['elbow'], SKIN, 8, zo=5)
    _seg(ax, front_arm['elbow'],    front_arm['wrist'],  SKIN, 6, zo=5)
    _dot(ax, front_arm['elbow'],    0.027, WHITE, zo=6)

    # ── Front leg ───────────────────────────────────────────────────────────
    _seg(ax, front_leg['hip'],   front_leg['knee'],  front_col, 10, zo=6)
    _seg(ax, front_leg['knee'],  front_leg['ankle'], front_col,  9, zo=6)
    _seg(ax, front_leg['heel'],  front_leg['toe'],   front_col,  8, zo=6)
    _dot(ax, front_leg['knee'],  0.030, WHITE, zo=7)
    _dot(ax, front_leg['ankle'], 0.022, WHITE, zo=7)

    # ── Force bars ──────────────────────────────────────────────────────────
    draw_force_bar(ax, -1.20, lf_fn, L_COL, 'LEFT')
    draw_force_bar(ax,  1.20, rf_fn, R_COL, 'RIGHT')

    # ── Phase indicator (small orbit dot) ───────────────────────────────────
    cx_ring = np.linspace(0, 2 * np.pi, 80)
    ax.plot(0.22 * np.cos(cx_ring), -0.14 + 0.048 * np.sin(cx_ring),
            color='#1e2d3d', lw=1.2, zorder=9)
    theta = 2 * np.pi * phase - np.pi / 2
    ax.plot(0.22 * np.cos(theta), -0.14 + 0.048 * np.sin(theta),
            'o', color=WHITE, markersize=5, zorder=10)
    ax.text(0, -0.195, 'GAIT CYCLE', ha='center', fontsize=6,
            color='#334455', zorder=9)


# ─── Main render ───────────────────────────────────────────────────────────────

def render(data_path='data/gait_data.csv', output_gif='data/gait_avatar.gif'):
    df = pd.read_csv(data_path)
    lf_cols = [c for c in df.columns if c.startswith('lf_')]
    rf_cols = [c for c in df.columns if c.startswith('rf_')]
    lf_total = df[lf_cols].sum(axis=1).values
    rf_total = df[rf_cols].sum(axis=1).values
    lf_heel  = (df['lf_heel_l_N'] + df['lf_heel_r_N']).values

    # ── Extract one representative gait cycle ──────────────────────────────
    strikes, _ = find_peaks(lf_heel, height=25, distance=40)
    if len(strikes) >= 3:
        s, e = int(strikes[1]), int(strikes[2])
    else:
        step = int(len(df) // 28)
        s, e = 0, step

    bw = max(float(lf_total.max()), float(rf_total.max()), 1.0)

    # ── Asymmetry metrics for title ────────────────────────────────────────
    cyc_lf = lf_total[s:e]
    cyc_rf = rf_total[s:e]
    l_on = cyc_lf[cyc_lf > 25].mean() if (cyc_lf > 25).any() else 1.0
    r_on = cyc_rf[cyc_rf > 25].mean() if (cyc_rf > 25).any() else 1.0
    asym = abs(l_on - r_on) / ((l_on + r_on) / 2) * 100

    # ── Resample cycle forces to N_FRAMES ─────────────────────────────────
    N = 72
    cyc_lf_r = np.clip(sp_resample(cyc_lf, N), 0, None)
    cyc_rf_r = np.clip(sp_resample(cyc_rf, N), 0, None)
    phases   = np.linspace(0, 1, N, endpoint=False)

    # ── Figure setup ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.22, 1.90)
    ax.set_aspect('equal')
    ax.axis('off')

    status_txt = ('LIMP DETECTED' if asym > 15 else
                  'MILD ASYMMETRY' if asym > 5 else 'NORMAL GAIT')
    status_col = ('#ef5350' if asym > 15 else
                  '#ffeb3b' if asym > 5 else '#69f0ae')

    fig.text(0.50, 0.97, 'Gait Analysis  |  Post-Surgical Rehabilitation',
             ha='center', color='white', fontsize=13, fontweight='bold')
    fig.text(0.50, 0.935, f'{status_txt}   –   Asymmetry Index: {asym:.1f}%',
             ha='center', color=status_col, fontsize=11, fontweight='bold')

    def _setup(a):
        a.clear()
        a.set_facecolor(BG)
        a.set_xlim(-1.55, 1.55)
        a.set_ylim(-0.22, 1.90)
        a.set_aspect('equal')
        a.axis('off')
        a.axhline(GROUND_Y, color='#1e2d3d', lw=1.8, zorder=0)

    def init():
        _setup(ax)
        return []

    def update(i):
        _setup(ax)
        idx = i % N
        draw_avatar(ax, phases[idx], cyc_lf_r[idx] / bw, cyc_rf_r[idx] / bw)
        return []

    anim = FuncAnimation(fig, update, frames=N,
                         init_func=init, interval=1000 / 28, blit=False)

    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    print('  Rendering animation frames (~20 s)...')
    anim.save(output_gif, writer=PillowWriter(fps=28))
    print(f'  Avatar animation saved → {output_gif}')
    plt.close(fig)


if __name__ == '__main__':
    render()
