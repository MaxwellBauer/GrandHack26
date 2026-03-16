"""
pressure_animation.py
=====================
Generates an MP4 animation of plantar pressure from a 5-sensor insole CSV.

CSV columns required:
    timestamp_ms, U1_g, M1_g, L1_g, L2_g, U2_g
    (any _raw columns are ignored)

Sensor layout (left foot, plantar view, toes at top):
    U1 = Medial forefoot  (under 1st metatarsal / big toe)
    U2 = Lateral forefoot (under 5th metatarsal / pinky)
    M1 = Midfoot / arch
    L1 = Medial heel
    L2 = Lateral heel

Usage:
    python pressure_animation.py data.csv
    python pressure_animation.py data.csv output.mp4
    python pressure_animation.py data.csv --fps 20 --smooth 5 --res 220
    python pressure_animation.py data.csv --vmax 500

Requirements:
    pip install numpy pandas matplotlib scipy
    ffmpeg must be installed and on PATH (brew install ffmpeg)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.path as mpath
import matplotlib.ticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RBFInterpolator, splprep, splev
from scipy.ndimage import uniform_filter1d


# ── Foot outline (identical to heatmap_viz.py) ────────────────────────────────
# x ∈ [0,1]:   0 = medial (big-toe side),  1 = lateral (pinky side)
# y ∈ [≈0, ≈2.28]: 0 = heel, 2.28 = toe tips
FOOT_CTRL = np.array([
    [0.50, 2.28],
    [0.84, 2.10],
    [0.92, 1.76],
    [0.92, 1.42],
    [0.88, 1.05],
    [0.84, 0.65],
    [0.78, 0.25],
    [0.62, 0.03],
    [0.50, -0.02],
    [0.38, 0.03],
    [0.22, 0.25],
    [0.16, 0.65],
    [0.14, 1.05],
    [0.08, 1.42],
    [0.08, 1.76],
    [0.16, 2.10],
    [0.50, 2.28],
])

# Sensor positions in foot coordinate space.
# User's normalized coords (origin top-left, y↓) → foot coords (y↑, heel=0):
#   foot_x = x_norm
#   foot_y = (1 - y_norm) * 2.30 - 0.02
SENSOR_KEYS = ['U1', 'M1', 'L1', 'L2']
SENSOR_POS = {
    'U1': np.array([0.40, 1.868]),   # Medial forefoot
    'M1': np.array([0.55, 1.176]),   # Midfoot / arch
    'L1': np.array([0.42, 0.486]),   # Medial heel
    'L2': np.array([0.58, 0.438]),   # Lateral heel
}
SENSOR_LABELS = {
    'U1': 'Medial\nForefoot',
    'M1': 'Midfoot',
    'L1': 'Medial\nHeel',
    'L2': 'Lateral\nHeel',
}

BG  = '#ffffff'   # white background
FG  = '#222222'   # primary text
FG2 = '#666666'   # secondary text


def make_cmap(cmap_end=1.0):
    """Return turbo truncated to [0, cmap_end] — e.g. 0.45 ends at green."""
    base = plt.get_cmap('turbo')
    if cmap_end >= 1.0:
        return base
    return LinearSegmentedColormap.from_list(
        f'turbo_trunc_{cmap_end:.2f}',
        base(np.linspace(0.0, cmap_end, 256))
    )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def smooth_outline(ctrl, n=600):
    pts = ctrl[:-1]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.column_stack([x, y])


def foot_mask(outline, gx, gy):
    path = mpath.Path(np.vstack([outline, outline[0]]))
    pts  = np.column_stack([gx.ravel(), gy.ravel()])
    return path.contains_points(pts).reshape(gx.shape)


def rbf_field(sensor_pts, sensor_vals, boundary_pts, gx, gy):
    """RBF interpolation with boundary constrained to zero."""
    bv   = np.zeros(len(boundary_pts))
    pts  = np.vstack([sensor_pts, boundary_pts])
    vals = np.concatenate([sensor_vals, bv])
    rbf  = RBFInterpolator(pts, vals, kernel='thin_plate_spline', smoothing=0)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return np.clip(rbf(grid).reshape(gx.shape), 0, None)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    needed = ['U1_g', 'M1_g', 'L1_g', 'L2_g']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: CSV missing columns: {missing}\n"
                 f"Found columns: {list(df.columns)}")

    # Build (n_frames × 4) array in SENSOR_KEYS order
    col_map = {'U1': 'U1_g', 'M1': 'M1_g', 'L1': 'L1_g', 'L2': 'L2_g'}
    sensor_data = df[[col_map[k] for k in SENSOR_KEYS]].values.astype(float)
    timestamps  = df['timestamp_ms'].values if 'timestamp_ms' in df.columns \
                  else np.arange(len(df)) * 50.0

    return sensor_data, timestamps


# ── Pre-computation ───────────────────────────────────────────────────────────

def precompute(res=220):
    """
    Build the geometry and RBF basis fields.

    The key insight: RBF interpolation is linear in the sensor values.
    So we precompute one 'basis pressure field' per sensor (unit impulse),
    then each animation frame = linear combination of bases × sensor values.
    This avoids re-solving the RBF system for every frame.
    """
    print("  Building foot outline and grid…")
    outline      = smooth_outline(FOOT_CTRL, n=600)
    boundary_pts = smooth_outline(FOOT_CTRL, n=80)   # boundary = 0 constraint

    gx, gy = np.meshgrid(np.linspace(-0.05, 1.05, res),
                         np.linspace(-0.10, 2.55, res))
    mask = foot_mask(outline, gx, gy)

    sensor_pts = np.array([SENSOR_POS[k] for k in SENSOR_KEYS])

    print(f"  Computing RBF basis fields ({len(SENSOR_KEYS)} sensors)…")
    bases = []
    for i, key in enumerate(SENSOR_KEYS):
        impulse = np.zeros(len(SENSOR_KEYS))
        impulse[i] = 1.0
        b = rbf_field(sensor_pts, impulse, boundary_pts, gx, gy)
        bases.append(b)
        print(f"    {key} ({i+1}/{len(SENSOR_KEYS)})")
    bases = np.array(bases)   # shape: (n_sensors, res, res)

    return outline, gx, gy, mask, bases


def build_pressure_frames(sensor_data, bases, smooth=3):
    """
    Compute per-frame pressure grids as fast matrix multiply, then
    optionally apply temporal smoothing.

    sensor_data : (n_frames, n_sensors)
    bases       : (n_sensors, res, res)
    returns     : (n_frames, res, res)
    """
    print("  Computing pressure frames via linear combination…")
    # (n_frames, n_sensors) × (n_sensors, res²) → (n_frames, res²)
    res = bases.shape[-1]
    pressures = np.tensordot(sensor_data, bases, axes=[[1], [0]])  # (n_frames, res, res)

    if smooth and smooth > 1:
        print(f"  Applying {smooth}-frame temporal smoothing…")
        pressures = uniform_filter1d(pressures.astype(float), size=smooth, axis=0)

    return np.clip(pressures, 0, None)


# ── Animation ─────────────────────────────────────────────────────────────────

def make_animation(sensor_data, timestamps, pressures,
                   outline, gx, gy, mask, vmax, fps, output_path,
                   peak_total, low_pct=15.0, high_pct=75.0, cmap=None):

    n_frames = len(pressures)
    sensor_pts = np.array([SENSOR_POS[k] for k in SENSOR_KEYS])
    if cmap is None:
        cmap = plt.get_cmap('turbo')
    norm = Normalize(vmin=0, vmax=vmax)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 12), facecolor=BG)
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[0.08, 1, 0.10],
                   hspace=0.06)

    ax_title = fig.add_subplot(gs[0])
    ax_foot  = fig.add_subplot(gs[1])
    ax_bar   = fig.add_subplot(gs[2])

    for ax in [ax_title, ax_foot, ax_bar]:
        ax.set_facecolor(BG)

    # Static foot silhouette background
    ax_foot.fill(outline[:, 0], outline[:, 1], color='#eeeeee', zorder=1)
    outline_closed = np.vstack([outline, outline[0]])
    ax_foot.plot(outline_closed[:, 0], outline_closed[:, 1],
                 color='#333333', linewidth=1.2, alpha=0.7, zorder=5)

    # Anatomical labels
    ax_foot.text(0.50, -0.08, 'HEEL', ha='center', va='top',
                 fontsize=12, color=FG2, style='italic',
                 transform=ax_foot.transData)
    ax_foot.text(0.50, 2.47, 'TOES', ha='center', va='bottom',
                 fontsize=12, color=FG2, style='italic',
                 transform=ax_foot.transData)
    ax_foot.text(-0.06, 1.10, 'MEDIAL', ha='left', va='center',
                 fontsize=10, color='#aaaaaa', rotation=90,
                 transform=ax_foot.transData)
    ax_foot.text(1.06, 1.10, 'LATERAL', ha='right', va='center',
                 fontsize=10, color='#aaaaaa', rotation=90,
                 transform=ax_foot.transData)

    ax_foot.set_xlim(-0.12, 1.12)
    ax_foot.set_ylim(-0.15, 2.55)
    ax_foot.set_aspect('equal')
    ax_foot.axis('off')

    # ── Colorbar (inset so foot stays centered) ────────────────────────────
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = ax_foot.inset_axes([1.03, 0.12, 0.04, 0.76])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Pressure (kg)', color=FG, fontsize=12, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=FG2, labelcolor=FG, labelsize=10)
    cbar.outline.set_edgecolor('#cccccc')
    cbar.formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f'{x/1000:.1f}')
    cbar.update_ticks()

    # ── Title panel ────────────────────────────────────────────────────────
    ax_title.axis('off')
    title_txt = ax_title.text(0.5, 0.65, 'Plantar Pressure - Right Foot - Partial Bearing',
                              ha='center', va='center', fontsize=18,
                              fontweight='bold', color=FG,
                              transform=ax_title.transAxes)
    time_txt = ax_title.text(0.5, 0.10, '',
                             ha='center', va='center', fontsize=12,
                             color=FG2, transform=ax_title.transAxes)

    # ── Total-force bar (% of global peak, with tolerance zones) ──────────
    ax_bar.set_facecolor(BG)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_ylim(-2.8, 1.2)
    ax_bar.axis('off')

    LOW, HIGH = low_pct, high_pct   # tolerance thresholds in %

    # Background track
    ax_bar.barh([0], [100], color='#e0e0e0', height=0.7, left=0, zorder=1)

    # Tolerance zone shading (green band between LOW and HIGH)
    ax_bar.barh([0], [HIGH - LOW], color='#c8e6c9', height=0.7,
                left=LOW, zorder=2, alpha=0.8)

    # Fixed tolerance lines (clipped to bar height) + offset labels
    bar_h = 0.35   # half-height of the bar (height=0.7)
    ax_bar.plot([LOW,  LOW],  [-bar_h, bar_h], color='#4fc3f7', linewidth=1.5, linestyle='--', alpha=0.85, zorder=5)
    ax_bar.plot([HIGH, HIGH], [-bar_h, bar_h], color='#ff6b6b', linewidth=1.5, linestyle='--', alpha=0.85, zorder=5)
    ax_bar.text(LOW  + 1.5, 0.42, f'MIN {LOW:.0f}%',  ha='left',  va='bottom', fontsize=10,
                color='#4fc3f7', fontweight='bold')
    ax_bar.text(HIGH + 1.5, 0.42, f'MAX {HIGH:.0f}%', ha='left',  va='bottom', fontsize=10,
                color='#ff6b6b', fontweight='bold')

    # Label axes
    ax_bar.text(0,   -0.55, '0%',   ha='left',   va='top', fontsize=10, color=FG2)
    ax_bar.text(100, -0.55, '100%', ha='right',  va='top', fontsize=10, color=FG2)
    ax_bar.text(50,  -0.55, '% Body Weight', ha='center', va='top',
                fontsize=11, color=FG2)

    # Dynamic bar (starts at 0)
    bar_fill = ax_bar.barh([0], [0], color='#4fc3f7', height=0.7, left=0, zorder=3)
    bar_label = ax_bar.text(50, -2.35, '0%',
                            ha='center', va='center', fontsize=22,
                            fontweight='bold', color=FG)

    # ── Initial heatmap (RGBA imshow — reliable for animation) ────────────
    # Build extent to match the gx/gy meshgrid
    x0, x1 = gx[0, 0],  gx[0, -1]
    y0, y1 = gy[0, 0],  gy[-1, 0]

    def frame_rgba(pressure_2d):
        """Convert a pressure grid + mask → RGBA array for imshow."""
        rgba = cmap(norm(pressure_2d)).copy()   # (res, res, 4)
        rgba[~mask] = (1.0, 1.0, 1.0, 1.0)  # white background outside foot
        return rgba

    im = ax_foot.imshow(frame_rgba(pressures[0]),
                        origin='lower',
                        extent=[x0, x1, y0, y1],
                        interpolation='bilinear',
                        zorder=2)

    # Sensor dot markers (static positions, values update)
    dot_artists = []
    val_artists = []
    for i, key in enumerate(SENSOR_KEYS):
        px, py = sensor_pts[i]
        dot, = ax_foot.plot(px, py, 'o', color='white', markersize=8,
                            markeredgecolor='#333', markeredgewidth=1.0, zorder=7)
        txt = ax_foot.text(px, py + 0.10, '0 kg',
                           ha='center', va='bottom', fontsize=10,
                           color=FG, fontweight='bold', zorder=8,
                           bbox=dict(boxstyle='round,pad=0.15',
                                     facecolor='white', alpha=0.75,
                                     edgecolor='none'))
        dot_artists.append(dot)
        val_artists.append(txt)

    # ── Update function ────────────────────────────────────────────────────
    def update(frame_idx):
        t_ms  = timestamps[frame_idx]
        vals  = sensor_data[frame_idx]
        total = vals.sum()

        # Heatmap
        im.set_data(frame_rgba(pressures[frame_idx]))

        # Sensor value labels
        for i, key in enumerate(SENSOR_KEYS):
            val_artists[i].set_text(f'{vals[i]/1000:.1f} kg')

        # Time label
        t_sec = t_ms / 1000.0
        time_txt.set_text(f't = {t_sec:.2f} s  |  frame {frame_idx+1}/{n_frames}')

        # Total force bar — percentage of global peak, color-coded
        pct = min(total / peak_total * 100, 100.0)
        if pct < LOW:
            color = '#4fc3f7'   # blue  — below lower tolerance
        elif pct <= HIGH:
            color = '#4caf50'   # green — within normal range
        else:
            color = '#f44336'   # red   — exceeds upper tolerance
        bar_fill[0].set_width(pct)
        bar_fill[0].set_facecolor(color)
        bar_label.set_text(f'{pct:.0f}%')
        bar_label.set_color(FG)

        return [im, time_txt, bar_label, bar_fill[0]] + val_artists

    # ── Build and save animation ───────────────────────────────────────────
    print(f"\n  Rendering {n_frames} frames at {fps} fps…")
    anim = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec='h264',
        bitrate=2500,
        extra_args=['-pix_fmt', 'yuv420p',   # broad player compatibility
                    '-preset', 'fast']
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing MP4 → {output_path}")
    with writer.saving(fig, output_path, dpi=150):
        for i in range(n_frames):
            update(i)
            writer.grab_frame()
            if (i + 1) % max(1, n_frames // 20) == 0:
                pct = (i + 1) / n_frames * 100
                bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:.0f}%  ({i+1}/{n_frames})", end='', flush=True)

    print(f"\n  Done → {output_path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Animate plantar pressure from 5-sensor insole CSV → MP4")
    parser.add_argument("csv",    help="Input CSV file")
    parser.add_argument("output", nargs='?', help="Output MP4 path (default: <csv_stem>.mp4)")
    parser.add_argument("--fps",    type=int,   default=20,
                        help="Output video frame rate (default 20, matching 50ms CSV interval)")
    parser.add_argument("--smooth", type=int,   default=3,
                        help="Temporal smoothing window in frames (default 3, 0=off)")
    parser.add_argument("--res",    type=int,   default=220,
                        help="Heatmap grid resolution (default 220)")
    parser.add_argument("--vmax",   type=float, default=None,
                        help="Colormap max pressure in grams (default: auto from data)")
    parser.add_argument("--start",  type=float, default=41.5,
                        help="Start time in seconds (default: 41.5 — walking section)")
    parser.add_argument("--end",    type=float, default=56.5,
                        help="End time in seconds (default: 56.5 — walking section)")
    parser.add_argument("--low",        type=float, default=10.0,
                        help="Lower tolerance threshold as %% of body weight (default: 10)")
    parser.add_argument("--high",       type=float, default=40.0,
                        help="Upper tolerance threshold as %% of body weight (default: 40)")
    parser.add_argument("--bodyweight", type=float, default=140.0,
                        help="Patient body weight in lbs used for %% normalization (default: 140)")
    parser.add_argument("--cmap-end",   type=float, default=1.0,
                        help="Truncate turbo colormap at this fraction (default: 1.0=full; 0.45=ends at green)")
    args = parser.parse_args()

    csv_path    = args.csv
    output_path = args.output or str(Path(csv_path).with_suffix('.mp4'))

    print(f"\n{'═'*54}")
    print(f"  Plantar Pressure Animator")
    print(f"  Input : {csv_path}")
    print(f"  Output: {output_path}")
    print(f"{'═'*54}\n")

    # Load data
    print("Loading CSV…")
    sensor_data_full, timestamps_full = load_data(csv_path)
    # Normalize bar to body weight (lbs → grams)
    peak_total = args.bodyweight * 453.592
    print(f"  {len(sensor_data_full)} frames  |  "
          f"{timestamps_full[-1]/1000:.2f} s  |  "
          f"sensors: {SENSOR_KEYS}")
    print(f"  Body weight: {args.bodyweight:.0f} lbs ({peak_total/1000:.1f} kg) — used for % normalization")

    # Time range filter
    t_ms = timestamps_full
    lo = args.start * 1000 if args.start is not None else t_ms[0]
    hi = args.end   * 1000 if args.end   is not None else t_ms[-1]
    mask_t = (t_ms >= lo) & (t_ms <= hi)
    sensor_data = sensor_data_full[mask_t]
    timestamps  = timestamps_full[mask_t]
    print(f"  Cropped to {len(sensor_data)} frames ({lo/1000:.1f}s – {hi/1000:.1f}s)")

    # Geometry + RBF basis
    print("\nBuilding heatmap geometry…")
    outline, gx, gy, mask, bases = precompute(res=args.res)

    # Pressure frames
    print("\nComputing pressure fields…")
    pressures = build_pressure_frames(sensor_data, bases, smooth=args.smooth)

    # Colormap range — use max individual sensor reading so the colorbar
    # reflects real sensor values (~24-30 kg), and with smoothing=0 the
    # RBF field passes through sensor points exactly, reaching full red
    vmax = args.vmax
    if vmax is None:
        vmax = max(float(sensor_data.max()), 1.0)
    print(f"  Colormap vmax: {vmax/1000:.1f} kg (sensor max)")

    # Animate + save
    print("\nAnimating…")
    cmap = make_cmap(args.cmap_end)
    make_animation(sensor_data, timestamps, pressures,
                   outline, gx, gy, mask, vmax, args.fps, output_path,
                   peak_total=peak_total,
                   low_pct=args.low, high_pct=args.high, cmap=cmap)

    print(f"\n{'═'*54}")
    print(f"  MP4 saved → {output_path}")
    print(f"  Duration: {len(sensor_data)/args.fps:.1f} s  |  "
          f"{len(sensor_data)} frames @ {args.fps} fps")
    print(f"{'═'*54}\n")


if __name__ == '__main__':
    main()
