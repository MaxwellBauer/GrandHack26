"""
gait_viz.py
Three-panel gait analysis dashboard:
  Panel 1 — Force traces over time with heel-strike markers
  Panel 2 — Step interval bar chart (cadence stability)
  Panel 3 — Symmetry summary with limp detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from pathlib import Path


FS = 100  # Hz


def load_foot_totals(df):
    lf_cols = [c for c in df.columns if c.startswith('lf_')]
    rf_cols = [c for c in df.columns if c.startswith('rf_')]
    lf_heel = df['lf_heel_l_N'] + df['lf_heel_r_N']
    rf_heel = df['rf_heel_l_N'] + df['rf_heel_r_N']
    lf_total = df[lf_cols].sum(axis=1)
    rf_total = df[rf_cols].sum(axis=1)
    return lf_total.values, rf_total.values, lf_heel.values, rf_heel.values


def detect_strikes(heel_signal, min_dist=40, threshold=30):
    peaks, _ = find_peaks(heel_signal, height=threshold, distance=min_dist)
    return peaks


def estimate_stance_duration(heel_signal, strike_idxs, threshold=15):
    durations = []
    for idx in strike_idxs:
        end = idx
        while end < len(heel_signal) - 1 and heel_signal[end] > threshold:
            end += 1
        dur = (end - idx) / FS
        if 0.1 < dur < 1.5:
            durations.append(dur)
    return np.array(durations)


def render(data_path='data/gait_data.csv', output_png='data/gait_analysis.png'):
    df = pd.read_csv(data_path)
    t = df['timestamp_s'].values

    lf_total, rf_total, lf_heel, rf_heel = load_foot_totals(df)

    lf_strikes = detect_strikes(lf_heel)
    rf_strikes = detect_strikes(rf_heel)

    lf_intervals = np.diff(t[lf_strikes]) if len(lf_strikes) > 1 else np.array([])
    rf_intervals = np.diff(t[rf_strikes]) if len(rf_strikes) > 1 else np.array([])

    lf_stance_durs = estimate_stance_duration(lf_heel, lf_strikes)
    rf_stance_durs = estimate_stance_duration(rf_heel, rf_strikes)

    lf_mean_stance = lf_stance_durs.mean() if len(lf_stance_durs) else 0
    rf_mean_stance = rf_stance_durs.mean() if len(rf_stance_durs) else 0

    asym = 0.0
    if lf_mean_stance + rf_mean_stance > 0:
        asym = abs(lf_mean_stance - rf_mean_stance) / ((lf_mean_stance + rf_mean_stance) / 2) * 100

    lf_cadence = 1 / lf_intervals.mean() if len(lf_intervals) else 0
    rf_cadence = 1 / rf_intervals.mean() if len(rf_intervals) else 0

    BG    = '#1a1a2e'
    PANEL = '#16213e'
    L_COL = '#4fc3f7'
    R_COL = '#ef5350'
    TEXT  = '#e0e0e0'

    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={'height_ratios': [2.5, 1.8, 1.5]})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Panel 1: Force traces
    ax1 = axes[0]
    ax1.plot(t, lf_total, color=L_COL, linewidth=1.2, label='Left Foot', alpha=0.9)
    ax1.plot(t, rf_total, color=R_COL, linewidth=1.2, label='Right Foot', alpha=0.9)

    for idx in lf_strikes:
        ax1.axvline(t[idx], color=L_COL, alpha=0.4, linewidth=0.8, linestyle='--')
    for idx in rf_strikes:
        ax1.axvline(t[idx], color=R_COL, alpha=0.4, linewidth=0.8, linestyle='--')

    for idx in lf_strikes[:5]:
        ax1.text(t[idx], lf_total[idx] + 15, 'L', color=L_COL,
                 fontsize=7, ha='center', fontweight='bold')
    for idx in rf_strikes[:5]:
        ax1.text(t[idx], rf_total[idx] + 15, 'R', color=R_COL,
                 fontsize=7, ha='center', fontweight='bold')

    ax1.set_ylabel('Total Force (N)', color=TEXT, fontsize=10)
    ax1.set_title('Foot Force Over Time', color=TEXT, fontsize=11, fontweight='bold')
    ax1.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    ax1.set_xlim(0, t[-1])
    ax1.yaxis.label.set_color(TEXT)

    # Panel 2: Step interval bar chart
    ax2 = axes[1]
    n_steps = max(len(lf_intervals), len(rf_intervals))
    w = 0.38

    if len(lf_intervals):
        xl = np.arange(1, len(lf_intervals) + 1)
        ax2.bar(xl - w/2, lf_intervals, width=w, color=L_COL, alpha=0.85, label='Left Interval')
    if len(rf_intervals):
        xr = np.arange(1, len(rf_intervals) + 1)
        ax2.bar(xr + w/2, rf_intervals, width=w, color=R_COL, alpha=0.85, label='Right Interval')

    ax2.axhline(1.0, color='white', linewidth=1.0, linestyle='--', alpha=0.5, label='Expected (1.0 s)')
    ax2.set_ylabel('Interval (s)', color=TEXT, fontsize=10)
    ax2.set_xlabel('Step #', color=TEXT, fontsize=10)
    ax2.set_title('Step Intervals (Cadence Stability)', color=TEXT, fontsize=11, fontweight='bold')
    ax2.legend(facecolor=PANEL, labelcolor=TEXT, framealpha=0.8, fontsize=9)
    ax2.set_xlim(0.4, n_steps + 0.6)
    ax2.yaxis.label.set_color(TEXT)

    # Panel 3: Symmetry summary
    ax3 = axes[2]
    ax3.axis('off')

    max_dur = max(lf_mean_stance, rf_mean_stance, 0.01)
    bar_y_l = 0.62
    bar_y_r = 0.30
    bar_h   = 0.18
    bar_max_w = 0.55

    ax3.add_patch(mpatches.FancyBboxPatch(
        (0.08, bar_y_l), bar_max_w * (lf_mean_stance / max_dur), bar_h,
        boxstyle='round,pad=0.01', facecolor=L_COL, alpha=0.85, transform=ax3.transAxes))
    ax3.text(0.06, bar_y_l + bar_h/2, 'Left', color=TEXT, fontsize=10,
             ha='right', va='center', transform=ax3.transAxes, fontweight='bold')
    ax3.text(0.08 + bar_max_w * (lf_mean_stance / max_dur) + 0.02,
             bar_y_l + bar_h/2,
             f'{lf_mean_stance:.2f} s  ({lf_cadence:.1f} steps/s)',
             color=TEXT, fontsize=9, va='center', transform=ax3.transAxes)

    ax3.add_patch(mpatches.FancyBboxPatch(
        (0.08, bar_y_r), bar_max_w * (rf_mean_stance / max_dur), bar_h,
        boxstyle='round,pad=0.01', facecolor=R_COL, alpha=0.85, transform=ax3.transAxes))
    ax3.text(0.06, bar_y_r + bar_h/2, 'Right', color=TEXT, fontsize=10,
             ha='right', va='center', transform=ax3.transAxes, fontweight='bold')
    ax3.text(0.08 + bar_max_w * (rf_mean_stance / max_dur) + 0.02,
             bar_y_r + bar_h/2,
             f'{rf_mean_stance:.2f} s  ({rf_cadence:.1f} steps/s)',
             color=TEXT, fontsize=9, va='center', transform=ax3.transAxes)

    if asym > 15:
        status_color = '#ff5252'
        status_text  = f'Asymmetry Index: {asym:.1f}%  \u2190 Limp Detected'
    elif asym > 5:
        status_color = '#ffeb3b'
        status_text  = f'Asymmetry Index: {asym:.1f}%  \u2190 Mild Asymmetry'
    else:
        status_color = '#69f0ae'
        status_text  = f'Asymmetry Index: {asym:.1f}%  \u2190 Normal'

    ax3.text(0.5, 0.04, status_text, color=status_color, fontsize=13,
             fontweight='bold', ha='center', va='bottom', transform=ax3.transAxes)
    ax3.set_title('Stance Duration Symmetry', color=TEXT, fontsize=11, fontweight='bold')

    fig.suptitle('Gait Analysis Dashboard', fontsize=15,
                 fontweight='bold', color=TEXT, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  Gait analysis saved → {output_png}")
    plt.show()


if __name__ == '__main__':
    render()
