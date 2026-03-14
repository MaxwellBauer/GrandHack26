"""
Smart Insole Gait Analysis Pipeline
====================================
Processes raw pressure sensor data from smart insoles:
- Detects individual gait cycles (heel strike to heel strike)
- Computes clinically relevant biomechanical metrics
- Sends metrics to Claude for clinical interpretation
- Returns structured analysis ready for FHIR resource creation
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
try:
    import anthropic
except ImportError:
    anthropic = None


# ── Sensor layout ──────────────────────────────────────────────────
# Left foot: lf_heel_l, lf_heel_r, lf_mid_l, lf_mid_r, lf_toe_l, lf_toe_c, lf_toe_r
# Right foot: rf_heel_l, rf_heel_r, rf_mid_l, rf_mid_r, rf_toe_l, rf_toe_c, rf_toe_r

LEFT_SENSORS = ["lf_heel_l_N", "lf_heel_r_N", "lf_mid_l_N", "lf_mid_r_N",
                "lf_toe_l_N", "lf_toe_c_N", "lf_toe_r_N"]
RIGHT_SENSORS = ["rf_heel_l_N", "rf_heel_r_N", "rf_mid_l_N", "rf_mid_r_N",
                 "rf_toe_l_N", "rf_toe_c_N", "rf_toe_r_N"]

HEEL_SENSORS_L = ["lf_heel_l_N", "lf_heel_r_N"]
HEEL_SENSORS_R = ["rf_heel_l_N", "rf_heel_r_N"]
TOE_SENSORS_L = ["lf_toe_l_N", "lf_toe_c_N", "lf_toe_r_N"]
TOE_SENSORS_R = ["rf_toe_l_N", "rf_toe_c_N", "rf_toe_r_N"]


@dataclass
class StrideMetrics:
    """Metrics for a single gait cycle."""
    stride_number: int
    foot: str  # "left" or "right"
    start_time: float
    end_time: float
    stride_duration_s: float
    stance_duration_s: float
    swing_duration_s: float
    stance_pct: float
    peak_heel_force_N: float
    peak_toe_force_N: float
    peak_total_force_N: float
    loading_rate_N_per_s: float  # how fast force ramps up at heel strike
    time_to_peak_s: float


@dataclass
class GaitSummary:
    """Aggregate metrics across all strides."""
    total_strides_left: int
    total_strides_right: int
    recording_duration_s: float
    cadence_steps_per_min: float

    # Left foot averages
    avg_stride_duration_left_s: float
    avg_stance_pct_left: float
    avg_peak_force_left_N: float
    avg_loading_rate_left: float

    # Right foot averages
    avg_stride_duration_right_s: float
    avg_stance_pct_right: float
    avg_peak_force_right_N: float
    avg_loading_rate_right: float

    # Symmetry metrics (1.0 = perfect symmetry)
    symmetry_index_force: float      # ratio of avg peak forces
    symmetry_index_timing: float     # ratio of avg stride durations
    symmetry_index_stance: float     # ratio of stance %

    # Variability (coefficient of variation)
    cv_stride_duration_left: float
    cv_stride_duration_right: float

    # Weight bearing
    max_force_left_N: float
    max_force_right_N: float
    avg_total_force_N: float

    # Flags
    flags: list


def load_gait_data(csv_path: str) -> list[dict]:
    """Load CSV into list of dicts with float values."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k.strip()] = float(v)
                except ValueError:
                    parsed[k.strip()] = 0.0
            rows.append(parsed)
    return rows


def sum_sensors(row: dict, sensors: list[str]) -> float:
    """Sum force values across a list of sensor columns."""
    return sum(row.get(s, 0.0) for s in sensors)


def detect_strides(data: list[dict], heel_sensors: list[str],
                   all_sensors: list[str], toe_sensors: list[str],
                   foot: str, min_peak: float = 50.0) -> list[StrideMetrics]:
    """
    Detect gait cycles using heel strike peaks.
    A stride runs from one heel-strike peak to the next.
    """
    # Compute heel force trace
    heel_trace = [sum_sensors(row, heel_sensors) for row in data]
    total_trace = [sum_sensors(row, all_sensors) for row in data]
    toe_trace = [sum_sensors(row, toe_sensors) for row in data]
    timestamps = [row["timestamp_s"] for row in data]

    # Find heel strike peaks (local maxima above threshold)
    peaks = []
    for i in range(2, len(heel_trace) - 2):
        if (heel_trace[i] > heel_trace[i-1] and
            heel_trace[i] > heel_trace[i+1] and
            heel_trace[i] > heel_trace[i-2] and
            heel_trace[i] > min_peak):
            # Avoid double-counting: enforce min distance of 40 samples (0.4s)
            if not peaks or (i - peaks[-1]) > 40:
                peaks.append(i)

    strides = []
    for s in range(len(peaks) - 1):
        start_idx = peaks[s]
        end_idx = peaks[s + 1]
        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx]
        duration = t_end - t_start

        if duration < 0.3 or duration > 2.5:
            continue  # skip unrealistic stride durations

        # Find stance phase: force > 5N threshold
        stance_samples = sum(1 for i in range(start_idx, end_idx)
                            if total_trace[i] > 5.0)
        total_samples = end_idx - start_idx
        stance_duration = stance_samples * 0.01  # 100Hz
        swing_duration = duration - stance_duration

        # Peak forces in this stride
        stride_total = total_trace[start_idx:end_idx]
        stride_heel = heel_trace[start_idx:end_idx]
        stride_toe = toe_trace[start_idx:end_idx]

        peak_total = max(stride_total) if stride_total else 0
        peak_heel = max(stride_heel) if stride_heel else 0
        peak_toe = max(stride_toe) if stride_toe else 0

        # Loading rate: force increase from 10% to 90% of peak heel force
        peak_idx_local = stride_heel.index(peak_heel) if peak_heel > 0 else 0
        if peak_idx_local > 2:
            loading_rate = peak_heel / (peak_idx_local * 0.01)
        else:
            loading_rate = 0.0

        time_to_peak = peak_idx_local * 0.01

        strides.append(StrideMetrics(
            stride_number=len(strides) + 1,
            foot=foot,
            start_time=round(t_start, 3),
            end_time=round(t_end, 3),
            stride_duration_s=round(duration, 3),
            stance_duration_s=round(stance_duration, 3),
            swing_duration_s=round(swing_duration, 3),
            stance_pct=round(stance_duration / duration * 100, 1),
            peak_heel_force_N=round(peak_heel, 1),
            peak_toe_force_N=round(peak_toe, 1),
            peak_total_force_N=round(peak_total, 1),
            loading_rate_N_per_s=round(loading_rate, 1),
            time_to_peak_s=round(time_to_peak, 3),
        ))

    return strides


def compute_summary(left_strides: list[StrideMetrics],
                    right_strides: list[StrideMetrics],
                    recording_duration: float) -> GaitSummary:
    """Compute aggregate gait metrics and symmetry indices."""

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    def cv(values):
        """Coefficient of variation (std/mean)."""
        if len(values) < 2:
            return 0.0
        m = avg(values)
        if m == 0:
            return 0.0
        variance = sum((v - m) ** 2 for v in values) / len(values)
        return (variance ** 0.5) / m

    def symmetry_index(a, b):
        """SI = 2 * |a - b| / (a + b) * 100. 0 = perfect symmetry."""
        if a + b == 0:
            return 0.0
        return round(2 * abs(a - b) / (a + b) * 100, 1)

    l_durations = [s.stride_duration_s for s in left_strides]
    r_durations = [s.stride_duration_s for s in right_strides]
    l_forces = [s.peak_total_force_N for s in left_strides]
    r_forces = [s.peak_total_force_N for s in right_strides]
    l_stance = [s.stance_pct for s in left_strides]
    r_stance = [s.stance_pct for s in right_strides]
    l_loading = [s.loading_rate_N_per_s for s in left_strides]
    r_loading = [s.loading_rate_N_per_s for s in right_strides]

    total_steps = len(left_strides) + len(right_strides)
    cadence = (total_steps / recording_duration) * 60 if recording_duration > 0 else 0

    flags = []

    # Check symmetry thresholds
    si_force = symmetry_index(avg(l_forces), avg(r_forces))
    si_timing = symmetry_index(avg(l_durations), avg(r_durations))
    si_stance = symmetry_index(avg(l_stance), avg(r_stance))

    if si_force > 15:
        flags.append(f"ASYMMETRIC_FORCE: SI={si_force}% (threshold 15%)")
    if si_timing > 10:
        flags.append(f"ASYMMETRIC_TIMING: SI={si_timing}% (threshold 10%)")
    if si_stance > 8:
        flags.append(f"ASYMMETRIC_STANCE: SI={si_stance}% (threshold 8%)")

    # Check variability
    cv_l = cv(l_durations)
    cv_r = cv(r_durations)
    if cv_l > 0.1:
        flags.append(f"HIGH_VARIABILITY_LEFT: CV={cv_l:.2f}")
    if cv_r > 0.1:
        flags.append(f"HIGH_VARIABILITY_RIGHT: CV={cv_r:.2f}")

    # Check cadence
    if cadence < 80:
        flags.append(f"LOW_CADENCE: {cadence:.0f} steps/min")
    elif cadence > 140:
        flags.append(f"HIGH_CADENCE: {cadence:.0f} steps/min")

    return GaitSummary(
        total_strides_left=len(left_strides),
        total_strides_right=len(right_strides),
        recording_duration_s=round(recording_duration, 1),
        cadence_steps_per_min=round(cadence, 1),
        avg_stride_duration_left_s=round(avg(l_durations), 3),
        avg_stance_pct_left=round(avg(l_stance), 1),
        avg_peak_force_left_N=round(avg(l_forces), 1),
        avg_loading_rate_left=round(avg(l_loading), 1),
        avg_stride_duration_right_s=round(avg(r_durations), 3),
        avg_stance_pct_right=round(avg(r_stance), 1),
        avg_peak_force_right_N=round(avg(r_forces), 1),
        avg_loading_rate_right=round(avg(r_loading), 1),
        symmetry_index_force=si_force,
        symmetry_index_timing=si_timing,
        symmetry_index_stance=si_stance,
        cv_stride_duration_left=round(cv_l, 3),
        cv_stride_duration_right=round(cv_r, 3),
        max_force_left_N=round(max(l_forces) if l_forces else 0, 1),
        max_force_right_N=round(max(r_forces) if r_forces else 0, 1),
        avg_total_force_N=round(avg(l_forces + r_forces), 1),
        flags=flags,
    )


def interpret_with_claude(summary: GaitSummary,
                          patient_context: Optional[dict] = None) -> dict:
    """
    Send gait metrics to Claude for clinical interpretation.
    Returns structured analysis as a dict.
    """
    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    context_str = ""
    if patient_context:
        context_str = f"""
Patient context:
- Age: {patient_context.get('age', 'unknown')}
- Sex: {patient_context.get('sex', 'unknown')}
- Weight: {patient_context.get('weight_kg', 'unknown')} kg
- Condition: {patient_context.get('condition', 'unknown')}
- Surgery: {patient_context.get('surgery', 'N/A')}
- WB restriction: {patient_context.get('wb_restriction', 'full weight bearing')}
- Days post-op: {patient_context.get('days_post_op', 'N/A')}
"""

    system_prompt = """You are a clinical biomechanics expert analyzing smart insole gait data 
for post-surgical rehabilitation monitoring. You provide structured gait analysis reports 
for orthopedic surgeons and physical therapists.

Your analysis should be:
- Evidence-based and clinically actionable
- Focused on weight-bearing compliance, symmetry, and safety
- Clear about what's normal vs. concerning
- Include specific recommendations for the care team

Respond ONLY with valid JSON in this exact structure:
{
    "clinical_summary": "2-3 sentence overview of the gait pattern",
    "weight_bearing_assessment": {
        "compliance_level": "compliant|partial|non_compliant|exceeding",
        "max_force_pct_body_weight": <number or null>,
        "notes": "string"
    },
    "symmetry_assessment": {
        "overall": "symmetric|mildly_asymmetric|moderately_asymmetric|severely_asymmetric",
        "force_symmetry_notes": "string",
        "timing_symmetry_notes": "string",
        "clinical_significance": "string"
    },
    "gait_quality": {
        "cadence_assessment": "normal|slow|fast",
        "stride_variability": "normal|elevated|concerning",
        "stance_swing_ratio": "normal|abnormal",
        "notes": "string"
    },
    "risk_flags": ["list of clinical concerns"],
    "recommendations": ["list of specific recommendations for care team"],
    "recovery_stage_estimate": "early|progressing|advanced|near_discharge",
    "follow_up_priority": "routine|soon|urgent"
}"""

    user_message = f"""Analyze the following gait data from a smart insole system.

{context_str}

Gait Metrics Summary:
{json.dumps(asdict(summary), indent=2)}

Provide your clinical interpretation as structured JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    # Parse Claude's response
    response_text = message.content[0].text
    # Strip markdown code fences if present
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

    return json.loads(response_text.strip())


def analyze_gait_data(csv_path: str,
                      patient_context: Optional[dict] = None,
                      use_claude: bool = True) -> dict:
    """
    Full pipeline: load data → detect strides → compute metrics → interpret.
    Returns dict with raw metrics and clinical interpretation.
    """
    print(f"Loading gait data from {csv_path}...")
    data = load_gait_data(csv_path)
    recording_duration = data[-1]["timestamp_s"] - data[0]["timestamp_s"]

    print("Detecting left foot strides...")
    left_strides = detect_strides(data, HEEL_SENSORS_L, LEFT_SENSORS, TOE_SENSORS_L, "left")
    print(f"  Found {len(left_strides)} left strides")

    print("Detecting right foot strides...")
    right_strides = detect_strides(data, HEEL_SENSORS_R, RIGHT_SENSORS, TOE_SENSORS_R, "right")
    print(f"  Found {len(right_strides)} right strides")

    print("Computing summary metrics...")
    summary = compute_summary(left_strides, right_strides, recording_duration)
    print(f"  Cadence: {summary.cadence_steps_per_min} steps/min")
    print(f"  Symmetry (force): {summary.symmetry_index_force}%")
    print(f"  Flags: {summary.flags}")

    result = {
        "raw_metrics": asdict(summary),
        "left_strides": [asdict(s) for s in left_strides],
        "right_strides": [asdict(s) for s in right_strides],
    }

    if use_claude:
        print("\nSending to Claude for clinical interpretation...")
        interpretation = interpret_with_claude(summary, patient_context)
        result["clinical_interpretation"] = interpretation
        print(f"  Summary: {interpretation.get('clinical_summary', 'N/A')}")
        print(f"  Priority: {interpretation.get('follow_up_priority', 'N/A')}")
    else:
        print("\nSkipping Claude interpretation (use_claude=False)")

    return result


# ── CLI entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/gait_data.csv"

    # Example patient context (customize for your demo)
    patient = {
        "age": 58,
        "sex": "male",
        "weight_kg": 82,
        "condition": "tibial plateau fracture, right leg",
        "surgery": "ORIF with locking plate",
        "wb_restriction": "toe-touch weight bearing (max 20% BW)",
        "days_post_op": 14,
    }

    use_api = os.environ.get("ANTHROPIC_API_KEY") is not None
    results = analyze_gait_data(csv_file, patient, use_claude=use_api)

    # Save results
    output_path = "data/gait_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
