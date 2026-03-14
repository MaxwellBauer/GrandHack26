"""
Smart Insole Pipeline Orchestrator
====================================
Runs the full end-to-end pipeline:
  CSV → Gait Analysis → Claude Interpretation → FHIR Resources → Vector Search

Usage:
  python main.py                          # Full pipeline
  python main.py --no-fhir                # Skip FHIR server (just analysis)
  python main.py --no-vector              # Skip vector search
  python main.py --csv path/to/data.csv   # Custom data file
"""

import json
import os
import sys
import argparse

# Load .env file if present (keeps API keys out of shell history and chat)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
except ImportError:
    pass

from gait_analysis import analyze_gait_data
from fhir_builder import push_gait_analysis_to_fhir, FHIR_BASE_URL
from vector_search import GaitVectorStore


def main():
    parser = argparse.ArgumentParser(description="Smart Insole Gait Analysis Pipeline")
    parser.add_argument("--csv", default="data/gait_data.csv", help="Path to gait CSV data")
    parser.add_argument("--no-fhir", action="store_true", help="Skip FHIR server push")
    parser.add_argument("--no-vector", action="store_true", help="Skip vector search storage")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude interpretation")
    parser.add_argument("--fhir-url", default=FHIR_BASE_URL, help="FHIR server base URL")
    parser.add_argument("--output", default="data/gait_analysis_results.json", help="Output JSON path")
    args = parser.parse_args()

    # ── Patient context (customize per demo) ──
    patient_context = {
        "patient_id": "SI-DEMO-001",
        "first_name": "Jane",
        "last_name": "Smith",
        "age": 58,
        "sex": "female",
        "birth_date": "1967-03-15",
        "weight_kg": 82,
        "condition": "tibial plateau fracture, right leg",
        "surgery": "ORIF with locking plate",
        "wb_restriction": "toe-touch weight bearing (max 20% BW)",
        "days_post_op": 14,
    }

    print("╔══════════════════════════════════════════════════╗")
    print("║  Smart Insole Gait Analysis Pipeline             ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ── Step 1: Analyze gait data ──
    print("━━━ Step 1: Gait Analysis ━━━")
    use_claude = not args.no_claude and os.environ.get("ANTHROPIC_API_KEY")
    if not use_claude and not args.no_claude:
        print("  ⚠ ANTHROPIC_API_KEY not set — skipping Claude interpretation")
        print("  Set it with: export ANTHROPIC_API_KEY=sk-ant-...")

    analysis = analyze_gait_data(args.csv, patient_context, use_claude=use_claude)

    # Save intermediate results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  → Saved to {args.output}")

    # If no Claude, add a placeholder interpretation for FHIR/vector
    if not use_claude and "clinical_interpretation" not in analysis:
        metrics = analysis["raw_metrics"]
        analysis["clinical_interpretation"] = {
            "clinical_summary": (
                f"Automated gait analysis: {metrics['total_strides_left']} left and "
                f"{metrics['total_strides_right']} right strides detected. "
                f"Cadence {metrics['cadence_steps_per_min']} steps/min. "
                f"Force symmetry index {metrics['symmetry_index_force']}%."
            ),
            "weight_bearing_assessment": {
                "compliance_level": "unknown",
                "notes": "Requires clinical review — no AI interpretation available"
            },
            "symmetry_assessment": {
                "overall": "mildly_asymmetric" if metrics["symmetry_index_force"] > 10 else "symmetric",
                "force_symmetry_notes": f"SI = {metrics['symmetry_index_force']}%",
                "timing_symmetry_notes": f"SI = {metrics['symmetry_index_timing']}%",
                "clinical_significance": "Automated flags: " + "; ".join(metrics.get("flags", []))
            },
            "gait_quality": {
                "cadence_assessment": "normal" if 80 <= metrics["cadence_steps_per_min"] <= 130 else "abnormal",
                "stride_variability": "normal" if metrics["cv_stride_duration_left"] < 0.1 else "elevated",
                "stance_swing_ratio": "normal",
                "notes": "Placeholder — run with ANTHROPIC_API_KEY for full analysis"
            },
            "risk_flags": metrics.get("flags", []),
            "recommendations": ["Clinical review of automated metrics recommended"],
            "recovery_stage_estimate": "unknown",
            "follow_up_priority": "routine",
        }

    # ── Step 2: Push to FHIR ──
    fhir_refs = None
    if not args.no_fhir:
        print("\n━━━ Step 2: FHIR Server ━━━")
        try:
            fhir_refs = push_gait_analysis_to_fhir(
                analysis, patient_context, args.fhir_url
            )
            # Save FHIR refs
            refs_path = args.output.replace(".json", "_fhir_refs.json")
            with open(refs_path, "w") as f:
                json.dump(fhir_refs, f, indent=2)
            print(f"  → FHIR refs saved to {refs_path}")
        except Exception as e:
            print(f"  ✗ FHIR push failed: {e}")
            fhir_refs = {
                "patient_id": patient_context["patient_id"],
                "patient_ref": f"Patient/{patient_context['patient_id']}",
                "report_ref": "DiagnosticReport/placeholder",
            }
    else:
        print("\n━━━ Step 2: FHIR Server (skipped) ━━━")
        fhir_refs = {
            "patient_id": patient_context["patient_id"],
            "patient_ref": f"Patient/{patient_context['patient_id']}",
            "report_ref": "DiagnosticReport/placeholder",
        }

    # ── Step 3: Vector Search ──
    if not args.no_vector:
        print("\n━━━ Step 3: Vector Search ━━━")
        try:
            store = GaitVectorStore(
                host=os.environ.get("IRIS_HOST", "localhost"),
                port=int(os.environ.get("IRIS_PORT", "32782")),
                namespace=os.environ.get("IRIS_NAMESPACE", "DEMO"),
                username=os.environ.get("IRIS_USER", "_SYSTEM"),
                password=os.environ.get("IRIS_PASSWORD", "ISCDEMO"),
            )
            store.store_analysis(analysis, fhir_refs)

            # Demo query
            print("\n  Demo semantic search: 'asymmetric gait pattern'")
            results = store.semantic_search("asymmetric gait pattern", top_k=3)
            for r in results:
                print(f"    [{r['similarity']:.3f}] {r['patient_id']}: "
                      f"{r['clinical_summary'][:80]}...")

            store.close()
        except Exception as e:
            print(f"  ✗ Vector search failed: {e}")
            print("  Make sure IRIS is running and the iris driver is installed.")
    else:
        print("\n━━━ Step 3: Vector Search (skipped) ━━━")

    print("\n╔══════════════════════════════════════════════════╗")
    print("║  Pipeline Complete!                              ║")
    print("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
