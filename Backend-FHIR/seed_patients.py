"""
seed_patients.py
================
Generates 13 additional synthetic patients (varying gait profiles) and runs
each through the full pipeline: CSV → gait analysis → Claude → FHIR → vector search.

Run from Backend-FHIR/ directory:
    python seed_patients.py

Takes ~2-3 minutes (one Claude API call per patient).
SI-DEMO-001 (Jane Smith) is already in the system from the previous run.
"""

import json
import os
import sys
import csv
import tempfile

import numpy as np
from scipy.signal import butter, filtfilt

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from gait_analysis import analyze_gait_data
from fhir_builder import push_gait_analysis_to_fhir, FHIR_BASE_URL
from vector_search import GaitVectorStore

# ── IRIS connection ──────────────────────────────────────────────────────────

IRIS_CONFIG = {
    "host":      os.environ.get("IRIS_HOST", "localhost"),
    "port":      int(os.environ.get("IRIS_PORT", "32782")),
    "namespace": os.environ.get("IRIS_NAMESPACE", "DEMO"),
    "username":  os.environ.get("IRIS_USER", "_SYSTEM"),
    "password":  os.environ.get("IRIS_PASSWORD", "ISCDEMO"),
}

FHIR_URL = os.environ.get(
    "IRIS_FHIR_URL",
    "http://localhost:32783/csp/healthshare/demo/fhir/r4"
)


# ── Synthetic gait CSV generator ─────────────────────────────────────────────

FS = 100
DURATION = 30


def _gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _lowpass(signal):
    b, a = butter(4, 20.0 / 50.0, btype="low")
    return filtfilt(b, a, signal)


def generate_gait_csv(
    output_path: str,
    body_weight_N: float = 700.0,
    cadence_hz: float = 1.0,
    left_load_factor: float = 1.0,
    right_load_factor: float = 1.0,
    left_stance_frac: float = 0.60,
    right_stance_frac: float = 0.60,
    left_delay_s: float = 0.0,
    right_delay_s: float = 0.04,
    left_toe_bias: float = 1.0,   # >1 = forefoot-heavy (plantar fasciitis etc.)
    right_toe_bias: float = 1.0,
    timing_jitter_s: float = 0.02,
    noise_std: float = 3.0,
    seed: int = 42,
):
    """Generate a 30-second 14-channel gait CSV with configurable asymmetry."""
    np.random.seed(seed)
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    cycle = 1.0 / cadence_hz

    cols = [
        "lf_heel_l_N", "lf_heel_r_N", "lf_mid_l_N", "lf_mid_r_N",
        "lf_toe_l_N",  "lf_toe_c_N",  "lf_toe_r_N",
        "rf_heel_l_N", "rf_heel_r_N", "rf_mid_l_N", "rf_mid_r_N",
        "rf_toe_l_N",  "rf_toe_c_N",  "rf_toe_r_N",
    ]
    sig = {c: np.zeros(n) for c in cols}

    def stamp(prefix, start_idx, n_stance, load, toe_b):
        end = min(start_idx + n_stance, n)
        clip = end - start_idx
        ts = np.linspace(0, 1, n_stance)

        # Scale so peak instantaneous total ≈ BW * load at both heel and push-off phases
        heel_pk = body_weight_N * 0.80 * load
        mid_pk  = body_weight_N * 0.18 * load
        toe_pk  = body_weight_N * 1.02 * load * toe_b

        waves = {
            f"{prefix}_heel_l_N": heel_pk * 0.50 * _gaussian(ts, 0.20, 0.18),
            f"{prefix}_heel_r_N": heel_pk * 0.50 * _gaussian(ts, 0.20, 0.18),
            f"{prefix}_mid_l_N":  mid_pk  * 0.50 * _gaussian(ts, 0.45, 0.12),
            f"{prefix}_mid_r_N":  mid_pk  * 0.50 * _gaussian(ts, 0.45, 0.12),
            f"{prefix}_toe_l_N":  toe_pk  * 0.28 * _gaussian(ts, 0.80, 0.11),
            f"{prefix}_toe_c_N":  toe_pk  * 0.44 * _gaussian(ts, 0.80, 0.10),
            f"{prefix}_toe_r_N":  toe_pk  * 0.28 * _gaussian(ts, 0.80, 0.11),
        }
        for col, wave in waves.items():
            sig[col][start_idx:end] += wave[:clip]

    # Left foot
    lf_stance = int(left_stance_frac * cycle * FS)
    strike = left_delay_s
    while strike < DURATION:
        jitter = np.random.normal(0, timing_jitter_s)
        idx = int((strike + jitter) * FS)
        if 0 <= idx < n:
            stamp("lf", idx, lf_stance, left_load_factor, left_toe_bias)
        strike += cycle

    # Right foot (offset by half cycle)
    rf_stance = int(right_stance_frac * cycle * FS)
    strike = cycle / 2 + right_delay_s
    while strike < DURATION:
        jitter = np.random.normal(0, timing_jitter_s)
        idx = int((strike + jitter) * FS)
        if 0 <= idx < n:
            stamp("rf", idx, rf_stance, right_load_factor, right_toe_bias)
        strike += cycle

    # Noise + filter + clip
    for col in cols:
        sig[col] += np.random.normal(0, noise_std, n)
        sig[col] = _lowpass(sig[col])
        sig[col] = np.clip(sig[col], 0, None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s"] + cols)
        for i in range(n):
            writer.writerow([f"{t[i]:.3f}"] + [f"{sig[c][i]:.3f}" for c in cols])

    print(f"    CSV: {output_path} ({n} rows)")


# ── Patient definitions ───────────────────────────────────────────────────────
# Each entry: (patient_context dict, gait_csv_kwargs dict)
# Varied conditions, recovery stages, asymmetry patterns, and body types.

PATIENTS = [
    # ── Moderate right-sided asymmetry ──────────────────────────────────────
    (
        {
            "patient_id": "SI-DEMO-002",
            "first_name": "Michael", "last_name": "Chen",
            "age": 45, "sex": "male", "birth_date": "1980-07-22",
            "weight_kg": 78,
            "condition": "ACL reconstruction, right knee",
            "surgery": "patellar tendon autograft ACL reconstruction",
            "wb_restriction": "full weight bearing as tolerated",
            "days_post_op": 90,
        },
        dict(body_weight_N=765, cadence_hz=1.05,
             right_load_factor=0.88, right_stance_frac=0.57,
             right_delay_s=0.02, timing_jitter_s=0.015, seed=2),
    ),

    # ── Severe right-sided asymmetry, very early post-op ────────────────────
    (
        {
            "patient_id": "SI-DEMO-003",
            "first_name": "Robert", "last_name": "Williams",
            "age": 67, "sex": "male", "birth_date": "1958-11-04",
            "weight_kg": 91,
            "condition": "total knee replacement, right knee",
            "surgery": "right total knee arthroplasty",
            "wb_restriction": "weight bearing as tolerated with walker",
            "days_post_op": 21,
        },
        dict(body_weight_N=892, cadence_hz=0.85,
             right_load_factor=0.70, right_stance_frac=0.48,
             right_delay_s=0.06, timing_jitter_s=0.03, seed=3),
    ),

    # ── Severe left-sided asymmetry, ankle fusion ────────────────────────────
    (
        {
            "patient_id": "SI-DEMO-004",
            "first_name": "Sarah", "last_name": "Johnson",
            "age": 34, "sex": "female", "birth_date": "1991-03-18",
            "weight_kg": 63,
            "condition": "left ankle fusion (tibiotalar arthrodesis)",
            "surgery": "ankle fusion with cannulated screws",
            "wb_restriction": "toe-touch weight bearing left foot (max 20% BW)",
            "days_post_op": 14,
        },
        dict(body_weight_N=618, cadence_hz=0.90,
             left_load_factor=0.62, left_stance_frac=0.45,
             left_delay_s=0.05, timing_jitter_s=0.025, seed=4),
    ),

    # ── Moderate right asymmetry, hip replacement, elderly ──────────────────
    (
        {
            "patient_id": "SI-DEMO-005",
            "first_name": "David", "last_name": "Martinez",
            "age": 72, "sex": "male", "birth_date": "1953-08-30",
            "weight_kg": 86,
            "condition": "right total hip arthroplasty",
            "surgery": "posterior approach total hip replacement",
            "wb_restriction": "weight bearing as tolerated, avoid hip flexion >90°",
            "days_post_op": 21,
        },
        dict(body_weight_N=843, cadence_hz=0.80,
             right_load_factor=0.80, right_stance_frac=0.53,
             right_delay_s=0.04, timing_jitter_s=0.04, seed=5),
    ),

    # ── Moderate right asymmetry, Achilles repair, mid-recovery ─────────────
    (
        {
            "patient_id": "SI-DEMO-006",
            "first_name": "Emily", "last_name": "Brown",
            "age": 29, "sex": "female", "birth_date": "1996-05-12",
            "weight_kg": 58,
            "condition": "right Achilles tendon rupture",
            "surgery": "open Achilles tendon repair",
            "wb_restriction": "partial weight bearing (max 50% BW) in boot",
            "days_post_op": 45,
        },
        dict(body_weight_N=569, cadence_hz=0.95,
             right_load_factor=0.75, right_stance_frac=0.52,
             right_toe_bias=0.5,  # avoids toe-off on right
             right_delay_s=0.03, timing_jitter_s=0.02, seed=6),
    ),

    # ── Severely non-compliant, calcaneus fracture ───────────────────────────
    (
        {
            "patient_id": "SI-DEMO-007",
            "first_name": "James", "last_name": "Wilson",
            "age": 55, "sex": "male", "birth_date": "1970-02-14",
            "weight_kg": 97,
            "condition": "right calcaneus fracture (Sanders type III)",
            "surgery": "ORIF calcaneus with plate and screws",
            "wb_restriction": "strict non-weight bearing right foot",
            "days_post_op": 10,
        },
        dict(body_weight_N=951, cadence_hz=0.78,
             right_load_factor=0.55, right_stance_frac=0.42,
             right_delay_s=0.08, timing_jitter_s=0.05, seed=7),
    ),

    # ── Bilateral heel avoidance — plantar fasciitis ─────────────────────────
    (
        {
            "patient_id": "SI-DEMO-008",
            "first_name": "Lisa", "last_name": "Anderson",
            "age": 48, "sex": "female", "birth_date": "1977-09-25",
            "weight_kg": 74,
            "condition": "bilateral plantar fasciitis",
            "surgery": "none — conservative management",
            "wb_restriction": "full weight bearing with custom orthotics",
            "days_post_op": 0,
        },
        dict(body_weight_N=726, cadence_hz=0.88,
             left_load_factor=0.90, right_load_factor=0.85,
             left_toe_bias=1.4, right_toe_bias=1.5,   # forefoot-dominant loading
             timing_jitter_s=0.02, seed=8),
    ),

    # ── Severe left asymmetry, tibial plateau fracture ───────────────────────
    (
        {
            "patient_id": "SI-DEMO-009",
            "first_name": "Thomas", "last_name": "Garcia",
            "age": 63, "sex": "male", "birth_date": "1962-12-08",
            "weight_kg": 88,
            "condition": "left tibial plateau fracture (Schatzker V)",
            "surgery": "ORIF with dual plating",
            "wb_restriction": "toe-touch weight bearing left foot (max 20% BW)",
            "days_post_op": 28,
        },
        dict(body_weight_N=863, cadence_hz=0.88,
             left_load_factor=0.68, left_stance_frac=0.47,
             left_delay_s=0.05, timing_jitter_s=0.03, seed=9),
    ),

    # ── Severe right forefoot avoidance, metatarsal fracture, acute ──────────
    (
        {
            "patient_id": "SI-DEMO-010",
            "first_name": "Jennifer", "last_name": "Taylor",
            "age": 52, "sex": "female", "birth_date": "1973-06-17",
            "weight_kg": 69,
            "condition": "right 5th metatarsal fracture (Jones fracture)",
            "surgery": "intramedullary screw fixation",
            "wb_restriction": "non-weight bearing right foot in boot",
            "days_post_op": 7,
        },
        dict(body_weight_N=676, cadence_hz=0.82,
             right_load_factor=0.58, right_stance_frac=0.43,
             right_toe_bias=0.3,   # avoids forefoot entirely
             right_delay_s=0.07, timing_jitter_s=0.04, seed=10),
    ),

    # ── Near-normal late recovery, ACL left, good prognosis ─────────────────
    (
        {
            "patient_id": "SI-DEMO-011",
            "first_name": "Amanda", "last_name": "Jackson",
            "age": 27, "sex": "female", "birth_date": "1998-04-03",
            "weight_kg": 61,
            "condition": "left ACL tear",
            "surgery": "hamstring autograft ACL reconstruction",
            "wb_restriction": "full weight bearing, return-to-sport phase",
            "days_post_op": 180,
        },
        dict(body_weight_N=598, cadence_hz=1.15,
             left_load_factor=0.95, left_stance_frac=0.59,
             timing_jitter_s=0.01, seed=11),
    ),

    # ── Moderate bilateral asymmetry, knee arthroscopy, early ────────────────
    (
        {
            "patient_id": "SI-DEMO-012",
            "first_name": "William", "last_name": "Harris",
            "age": 44, "sex": "male", "birth_date": "1981-10-29",
            "weight_kg": 83,
            "condition": "right knee medial meniscus tear",
            "surgery": "arthroscopic partial medial meniscectomy",
            "wb_restriction": "weight bearing as tolerated",
            "days_post_op": 7,
        },
        dict(body_weight_N=814, cadence_hz=0.95,
             right_load_factor=0.83, right_stance_frac=0.55,
             right_delay_s=0.02, timing_jitter_s=0.02, seed=12),
    ),

    # ── Near-complete recovery, patellar tendon repair ───────────────────────
    (
        {
            "patient_id": "SI-DEMO-013",
            "first_name": "Jessica", "last_name": "Thompson",
            "age": 38, "sex": "female", "birth_date": "1987-01-11",
            "weight_kg": 65,
            "condition": "right patellar tendon rupture",
            "surgery": "patellar tendon primary repair",
            "wb_restriction": "full weight bearing, final rehab phase",
            "days_post_op": 120,
        },
        dict(body_weight_N=638, cadence_hz=1.10,
             right_load_factor=0.92, right_stance_frac=0.58,
             timing_jitter_s=0.012, seed=13),
    ),

    # ── Bilateral slow gait, osteoarthritis, no surgery ─────────────────────
    (
        {
            "patient_id": "SI-DEMO-014",
            "first_name": "Daniel", "last_name": "White",
            "age": 70, "sex": "male", "birth_date": "1955-03-19",
            "weight_kg": 93,
            "condition": "bilateral knee osteoarthritis (Kellgren-Lawrence grade 3)",
            "surgery": "none — awaiting bilateral knee replacement",
            "wb_restriction": "full weight bearing, activity modification",
            "days_post_op": 0,
        },
        dict(body_weight_N=912, cadence_hz=0.72,
             left_load_factor=0.82, right_load_factor=0.78,
             left_stance_frac=0.65, right_stance_frac=0.63,
             timing_jitter_s=0.05, noise_std=5.0, seed=14),
    ),
]


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_patient(patient_context: dict, csv_kwargs: dict, store: GaitVectorStore):
    pid = patient_context["patient_id"]
    name = f"{patient_context['first_name']} {patient_context['last_name']}"
    print(f"\n{'─'*54}")
    print(f"  Patient {pid}: {name}")
    print(f"  {patient_context['condition']} | day {patient_context['days_post_op']} post-op")
    print(f"{'─'*54}")

    # 1. Generate CSV
    csv_path = os.path.join(tempfile.gettempdir(), f"gait_{pid}.csv")
    generate_gait_csv(csv_path, **csv_kwargs)

    # 2. Gait analysis + Claude
    use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not use_claude:
        print("  ⚠ No ANTHROPIC_API_KEY — running without Claude interpretation")

    analysis = analyze_gait_data(csv_path, patient_context, use_claude=use_claude)

    # Add fallback interpretation if Claude skipped
    if not use_claude and "clinical_interpretation" not in analysis:
        m = analysis["raw_metrics"]
        analysis["clinical_interpretation"] = {
            "clinical_summary": (
                f"Automated gait analysis for {name}: "
                f"cadence {m['cadence_steps_per_min']} steps/min, "
                f"force SI {m['symmetry_index_force']}%."
            ),
            "weight_bearing_assessment": {"compliance_level": "unknown", "notes": ""},
            "symmetry_assessment": {
                "overall": "mildly_asymmetric" if m["symmetry_index_force"] > 10 else "symmetric",
                "force_symmetry_notes": f"SI={m['symmetry_index_force']}%",
                "timing_symmetry_notes": f"SI={m['symmetry_index_timing']}%",
                "clinical_significance": "; ".join(m.get("flags", [])),
            },
            "gait_quality": {
                "cadence_assessment": "normal" if 80 <= m["cadence_steps_per_min"] <= 130 else "abnormal",
                "stride_variability": "normal",
                "stance_swing_ratio": "normal",
                "notes": "",
            },
            "risk_flags": m.get("flags", []),
            "recommendations": ["Clinical review recommended"],
            "recovery_stage_estimate": "unknown",
            "follow_up_priority": "routine",
        }

    si = analysis["raw_metrics"].get("symmetry_index_force", 0)
    priority = analysis.get("clinical_interpretation", {}).get("follow_up_priority", "?")
    print(f"    Force SI: {si:.1f}% | Priority: {priority}")

    # 3. FHIR
    try:
        fhir_refs = push_gait_analysis_to_fhir(analysis, patient_context, FHIR_URL)
        print(f"    FHIR: {fhir_refs.get('report_ref', 'stored')}")
    except Exception as e:
        print(f"    FHIR failed: {e} — using placeholder refs")
        fhir_refs = {
            "patient_id": pid,
            "patient_ref": f"Patient/{pid}",
            "report_ref": f"DiagnosticReport/placeholder-{pid}",
        }

    # 4. Vector search
    store.store_analysis(analysis, fhir_refs)

    # Clean up temp CSV
    try:
        os.remove(csv_path)
    except OSError:
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Clear existing vector store records before seeding")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Smart Insole — Patient Database Seeder              ║")
    print(f"║  Adding {len(PATIENTS)} patients to IRIS vector store          ║")
    print("╚══════════════════════════════════════════════════════╝")

    print("\nConnecting to IRIS vector store...")
    try:
        store = GaitVectorStore(**IRIS_CONFIG)
        print("  ✓ Connected")
    except Exception as e:
        print(f"  ✗ Cannot connect to IRIS: {e}")
        print("  Make sure IRIS is running: docker-compose up -d")
        sys.exit(1)

    if args.reset:
        print("\nResetting vector store…")
        store.clear_all()

    success = 0
    for patient_context, csv_kwargs in PATIENTS:
        try:
            run_patient(patient_context, csv_kwargs, store)
            success += 1
        except Exception as e:
            print(f"  ✗ Failed for {patient_context['patient_id']}: {e}")
            import traceback; traceback.print_exc()

    store.close()

    print(f"\n{'═'*54}")
    print(f"  Done: {success}/{len(PATIENTS)} patients stored in IRIS vector search")
    print(f"  Total in database: ~{success + 1} patients (including SI-DEMO-001)")
    print(f"  Open http://localhost:5050 and try the semantic search!")
    print(f"{'═'*54}\n")


if __name__ == "__main__":
    main()
