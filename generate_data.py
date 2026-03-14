"""
generate_data.py
Generates synthetic shoe pressure sensor data simulating a walking gait,
including a mild right-foot limp (post-surgery rehabilitation scenario).

Sensor layout per foot (7 sensors):
    [toe_l]  [toe_c]  [toe_r]   <- 3 toe sensors
       [mid_l]    [mid_r]       <- 2 midsole sensors
       [heel_l]   [heel_r]      <- 2 heel sensors
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt


# ── Constants ──────────────────────────────────────────────────────────────────
FS = 100            # Hz
DURATION = 30       # seconds
BODY_WEIGHT = 700   # Newtons (≈70 kg person)
CADENCE = 1.0       # steps per second per foot
CYCLE_PERIOD = 1 / CADENCE  # seconds per step

# Normal foot: stance = 60% of cycle
NORMAL_STANCE_FRAC = 0.60

# Limp (right foot) parameters
LIMP_LOAD_FACTOR   = 0.72   # right foot bears 72% of normal load
LIMP_STANCE_FRAC   = 0.50   # shortened stance
LIMP_DELAY         = 0.04   # seconds — hesitation before right heel strike

NOISE_STD = 3.0   # Newtons RMS sensor noise
CUTOFF_HZ = 20.0  # low-pass filter cutoff


# ── Helpers ────────────────────────────────────────────────────────────────────

def gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def lowpass(signal, fs=FS, cutoff=CUTOFF_HZ):
    b, a = butter(4, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)


def build_stance_waveforms(n_samples, load_scale=1.0):
    """
    Returns dict of sensor waveforms for a single stance phase, normalized
    to [0, 1] time, then scaled by load_scale and BODY_WEIGHT fractions.
    """
    t = np.linspace(0, 1, n_samples)

    # Anatomically correct plantar pressure distribution:
    #   Heel (initial contact, persists through midstance): HIGH
    #   Midsole/arch (normal foot arch rarely contacts ground):  VERY LOW
    #   Forefoot/metatarsal heads (push-off):                   HIGHEST
    heel_peak = BODY_WEIGHT * 0.40 * load_scale
    mid_peak  = BODY_WEIGHT * 0.09 * load_scale   # arch region — minimal loading
    toe_peak  = BODY_WEIGHT * 0.51 * load_scale   # metatarsal heads — primary push-off

    # Heel: peaks at initial contact, stays loaded through midstance
    heel_l = heel_peak * 0.50 * gaussian(t, mu=0.20, sigma=0.18)
    heel_r = heel_peak * 0.50 * gaussian(t, mu=0.20, sigma=0.18)

    # Midsole (arch): low, broad midstance contact
    mid_l  = mid_peak  * 0.50 * gaussian(t, mu=0.45, sigma=0.12)
    mid_r  = mid_peak  * 0.50 * gaussian(t, mu=0.45, sigma=0.12)

    # Forefoot sensors (metatarsal heads): push-off spike, center heavier
    toe_l  = toe_peak  * 0.28 * gaussian(t, mu=0.80, sigma=0.11)
    toe_c  = toe_peak  * 0.44 * gaussian(t, mu=0.80, sigma=0.10)
    toe_r  = toe_peak  * 0.28 * gaussian(t, mu=0.80, sigma=0.11)

    return dict(heel_l=heel_l, heel_r=heel_r,
                mid_l=mid_l,   mid_r=mid_r,
                toe_l=toe_l,   toe_c=toe_c,  toe_r=toe_r)


def stamp_foot(signal_dict, prefix, start_idx, n_stance, total_len):
    """
    Stamps one stance waveform into a full-length signal array.
    signal_dict: {sensor_name: full-length np.array}
    prefix: 'lf' or 'rf'
    """
    end_idx = min(start_idx + n_stance, total_len)
    clip_len = end_idx - start_idx

    waves = build_stance_waveforms(n_stance, load_scale=1.0)
    for sensor in ['heel_l', 'heel_r', 'mid_l', 'mid_r', 'toe_l', 'toe_c', 'toe_r']:
        col = f"{prefix}_{sensor}_N"
        signal_dict[col][start_idx:end_idx] += waves[sensor][:clip_len]


# ── Main generation ────────────────────────────────────────────────────────────

def run(output_path='data/gait_data.csv'):
    np.random.seed(42)
    n_samples = int(DURATION * FS)
    t = np.arange(n_samples) / FS

    lf_cols = ['lf_heel_l_N','lf_heel_r_N','lf_mid_l_N','lf_mid_r_N',
               'lf_toe_l_N','lf_toe_c_N','lf_toe_r_N']
    rf_cols = ['rf_heel_l_N','rf_heel_r_N','rf_mid_l_N','rf_mid_r_N',
               'rf_toe_l_N','rf_toe_c_N','rf_toe_r_N']

    signals = {col: np.zeros(n_samples) for col in lf_cols + rf_cols}

    # ── Left foot (normal) ──
    lf_stance_samples = int(NORMAL_STANCE_FRAC * CYCLE_PERIOD * FS)
    lf_strike = 0.0  # first strike at t=0
    while lf_strike < DURATION:
        idx = int(lf_strike * FS)
        stamp_foot(signals, 'lf', idx, lf_stance_samples, n_samples)
        lf_strike += CYCLE_PERIOD

    # ── Right foot (limp) ──
    # Offset by half a cycle (alternating), plus hesitation delay
    rf_stance_samples = int(LIMP_STANCE_FRAC * CYCLE_PERIOD * FS)
    rf_strike = CYCLE_PERIOD / 2 + LIMP_DELAY
    while rf_strike < DURATION:
        idx = int(rf_strike * FS)
        # stamp at full amplitude then scale down
        temp = {col: np.zeros(n_samples) for col in rf_cols}
        stamp_foot(temp, 'rf', idx, rf_stance_samples, n_samples)
        for col in rf_cols:
            signals[col] += temp[col] * LIMP_LOAD_FACTOR
        rf_strike += CYCLE_PERIOD

    # ── Add noise and filter ──
    for col in lf_cols + rf_cols:
        signals[col] += np.random.normal(0, NOISE_STD, n_samples)
        signals[col] = lowpass(signals[col])
        signals[col] = np.clip(signals[col], 0, None)

    # ── Build DataFrame ──
    df = pd.DataFrame({'timestamp_s': t})
    for col in lf_cols + rf_cols:
        df[col] = signals[col]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.3f')
    print(f"  Generated {len(df)} rows → {output_path}")
    return df


if __name__ == '__main__':
    run()
