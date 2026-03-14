"""
heatmap_viz.py
Proper plantar pressure heat map using:
  - Smooth spline foot silhouette (anatomically proportioned)
  - RBF interpolation between 7 sensor points
  - Masked pcolormesh (pressure only inside the foot outline)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import RBFInterpolator, splprep, splev
from pathlib import Path


# ── Foot outline — right foot, plantar view ────────────────────────────────────
# x ∈ [0,1]:   0 = medial (big-toe side),  1 = lateral (pinky side)
# y ∈ [0,2.5]: 0 = heel,                  2.5 = toe tips
# Points go clockwise starting from medial heel.
RIGHT_FOOT_CTRL = np.array([
    # Forefoot — wide, fully rounded top (no toes)
    [0.50, 2.28],
    [0.84, 2.10],
    [0.92, 1.76],
    # Lateral side — gently curved, barely any concavity
    [0.92, 1.42],   # widest lateral point
    [0.88, 1.05],   # subtle lateral waist
    [0.84, 0.65],
    [0.78, 0.25],
    # Heel — compact rounded oval
    [0.62, 0.03],
    [0.50, -0.02],  # heel posterior
    [0.38, 0.03],
    [0.22, 0.25],
    [0.16, 0.65],
    # Medial side — just slightly more curved than lateral (arch)
    [0.14, 1.05],   # subtle medial waist
    [0.08, 1.42],   # widest medial point
    [0.08, 1.76],
    [0.16, 2.10],
    [0.50, 2.28],   # close
])

# Sensor positions in same coordinate space (right foot)
# Sensor positions — symmetric around x=0.50, y positions matched to anatomy
# Foot length in coord space ≈ 2.33 units (y: -0.05 to 2.28)
#   Heel pad:          ~15% from heel  → y ≈ 0.35
#   Arch/midfoot:      ~43% from heel  → y ≈ 1.00
#   Metatarsal heads:  ~65-70% from heel → y ≈ 1.51–1.63
SENSOR_POS_RIGHT = {
    'heel_l': np.array([0.34, 0.35]),
    'heel_r': np.array([0.66, 0.35]),
    'mid_l':  np.array([0.26, 1.00]),
    'mid_r':  np.array([0.74, 1.00]),
    'toe_l':  np.array([0.23, 1.51]),
    'toe_c':  np.array([0.50, 1.63]),   # 3rd metatarsal head slightly more distal
    'toe_r':  np.array([0.77, 1.51]),
}

SENSOR_LABELS = {
    'heel_l': 'H·Med', 'heel_r': 'H·Lat',
    'mid_l':  'M·Med', 'mid_r':  'M·Lat',
    'toe_l':  'T·Med', 'toe_c':  'T·Ctr', 'toe_r': 'T·Lat',
}

SENSOR_KEYS = ['heel_l', 'heel_r', 'mid_l', 'mid_r', 'toe_l', 'toe_c', 'toe_r']


# ── Helpers ────────────────────────────────────────────────────────────────────

def smooth_outline(ctrl, n=600):
    """Fit a closed cubic spline through control points."""
    pts = ctrl[:-1]  # remove duplicate closing point
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([x, y])


def foot_mask(outline, gx, gy):
    """True where grid points lie inside the foot outline."""
    path = mpath.Path(np.vstack([outline, outline[0]]))
    pts  = np.column_stack([gx.ravel(), gy.ravel()])
    return path.contains_points(pts).reshape(gx.shape)


def rbf_pressure(sensor_pts, sensor_vals, boundary_pts, gx, gy):
    """Smooth RBF interpolation; boundary constrained to zero."""
    bv   = np.zeros(len(boundary_pts))
    pts  = np.vstack([sensor_pts, boundary_pts])
    vals = np.concatenate([sensor_vals, bv])
    rbf  = RBFInterpolator(pts, vals, kernel='thin_plate_spline', smoothing=0.5)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return np.clip(rbf(grid).reshape(gx.shape), 0, None)


def mirror_x(pts, x_max=1.0):
    m = pts.copy()
    m[:, 0] = x_max - m[:, 0]
    return m


# ── Main draw function ─────────────────────────────────────────────────────────

def draw_foot(ax, sensor_avgs, prefix, title, cmap, norm, is_left=False):
    sensor_vals = np.array([sensor_avgs.get(f'{prefix}_{k}_N', 0.0)
                            for k in SENSOR_KEYS])

    # Outline + sensor positions (mirror x for left foot)
    outline = smooth_outline(RIGHT_FOOT_CTRL)
    s_pos   = np.array([SENSOR_POS_RIGHT[k] for k in SENSOR_KEYS])
    if is_left:
        outline = mirror_x(outline)
        s_pos   = mirror_x(s_pos)

    # Fine grid
    res = 350
    gx, gy = np.meshgrid(np.linspace(-0.05, 1.05, res),
                         np.linspace(-0.10, 2.60, res))

    mask     = foot_mask(outline, gx, gy)
    boundary = smooth_outline(RIGHT_FOOT_CTRL if not is_left
                              else mirror_x(RIGHT_FOOT_CTRL[:-1], x_max=1.0), n=60)
    pressure = rbf_pressure(s_pos, sensor_vals, boundary, gx, gy)

    # Mask outside foot → NaN
    p_masked = np.where(mask, pressure, np.nan)

    # Draw background (dark foot silhouette so we don't see grid artifacts)
    ax.fill(outline[:, 0], outline[:, 1],
            color='#111', zorder=1)

    # Heat map
    ax.pcolormesh(gx, gy, p_masked, cmap=cmap, norm=norm,
                  shading='gouraud', rasterized=True, zorder=2)

    # Foot outline
    cl = np.vstack([outline, outline[0]])
    ax.plot(cl[:, 0], cl[:, 1], color='white', linewidth=1.2,
            alpha=0.6, zorder=4)

    # Sensor markers + force labels
    for i, key in enumerate(SENSOR_KEYS):
        px, py = s_pos[i]
        val = sensor_vals[i]
        ax.plot(px, py, 'o', color='white', markersize=5,
                markeredgecolor='#333', markeredgewidth=0.8, zorder=6)
        ax.text(px, py + 0.09, f'{val:.0f} N',
                ha='center', va='bottom', fontsize=6.5,
                color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.55, edgecolor='none'))

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.15, 2.55)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)

    # Anatomical direction labels
    ax.text(0.50, -0.12, 'HEEL', ha='center', fontsize=7.5,
            color='#888', style='italic', transform=ax.transData)
    ax.text(0.50,  2.48, 'TOES', ha='center', fontsize=7.5,
            color='#888', style='italic', transform=ax.transData)


# ── Entry point ────────────────────────────────────────────────────────────────

def render(data_path='data/gait_data.csv', output_png='data/heatmap.png'):
    df = pd.read_csv(data_path)

    sensor_cols  = [c for c in df.columns if c != 'timestamp_s']
    sensor_avgs  = {col: df[col].mean() for col in sensor_cols}

    all_vals = [sensor_avgs[c] for c in sensor_cols]
    norm = Normalize(vmin=0, vmax=max(all_vals) * 1.05)
    cmap = plt.get_cmap('turbo')   # blue→green→yellow→red — high visual impact

    BG = '#0d1117'
    fig, axes = plt.subplots(1, 2, figsize=(9, 11),
                             gridspec_kw={'wspace': 0.12})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)

    draw_foot(axes[0], sensor_avgs, 'lf',
              'Left Foot\nNormal Loading',
              cmap, norm, is_left=True)
    draw_foot(axes[1], sensor_avgs, 'rf',
              'Right Foot\nReduced Loading (Limp)',
              cmap, norm, is_left=False)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.025, pad=0.03, shrink=0.6)
    cbar.set_label('Avg Force (N)', color='white', fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    fig.suptitle('Plantar Pressure Distribution  (30 s Average)',
                 fontsize=14, fontweight='bold', color='white', y=0.97)
    for ax in axes:
        ax.title.set_color('white')

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches='tight',
                facecolor=BG)
    print(f'  Heat map saved → {output_png}')
    plt.show()


if __name__ == '__main__':
    render()
