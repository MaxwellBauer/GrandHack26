"""
Smart Insole Gait Intelligence — Web Dashboard
===============================================
Run from Backend-FHIR/ directory:

    python app.py

Then open: http://localhost:5050
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request, send_from_directory

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import requests as http_requests

app = Flask(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RESULTS_PATH  = os.path.join(DATA_DIR, "gait_analysis_results.json")
REFS_PATH     = os.path.join(DATA_DIR, "gait_analysis_results_fhir_refs.json")
ANALYSIS_DIR  = os.path.join(BASE_DIR, "..", "Analysis")
VIZ_DIR       = os.path.join(ANALYSIS_DIR, "data")   # where PNGs live
DATA_CSV      = os.path.join(DATA_DIR, "gait_data.csv")

FHIR_BASE = os.environ.get(
    "IRIS_FHIR_URL",
    "http://localhost:32783/csp/healthshare/demo/fhir/r4"
)
FHIR_AUTH = (
    os.environ.get("IRIS_USER", "_SYSTEM"),
    os.environ.get("IRIS_PASSWORD", "ISCDEMO"),
)
IRIS_CONFIG = {
    "host":      os.environ.get("IRIS_HOST", "localhost"),
    "port":      int(os.environ.get("IRIS_PORT", "32782")),
    "namespace": os.environ.get("IRIS_NAMESPACE", "DEMO"),
    "username":  FHIR_AUTH[0],
    "password":  FHIR_AUTH[1],
}


# ── Visualization auto-generation ────────────────────────────────────────────

def ensure_visualizations():
    """Generate heatmap and gait-analysis PNGs if they don't already exist."""
    heatmap_path = os.path.join(VIZ_DIR, "heatmap.png")
    gait_path    = os.path.join(VIZ_DIR, "gait_analysis.png")

    if os.path.exists(heatmap_path) and os.path.exists(gait_path):
        return  # already generated

    if not os.path.exists(DATA_CSV):
        print("  ⚠ No gait_data.csv found — skipping visualization generation")
        return

    print("  Generating visualizations (first run)…")
    sys.path.insert(0, ANALYSIS_DIR)
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend, safe for server use
        os.makedirs(VIZ_DIR, exist_ok=True)

        from heatmap_viz import render as render_heatmap
        render_heatmap(DATA_CSV, heatmap_path)

        from gait_viz import render as render_gait
        render_gait(DATA_CSV, gait_path)

        print(f"  ✓ heatmap.png and gait_analysis.png saved to {VIZ_DIR}")
    except Exception as e:
        print(f"  ⚠ Visualization generation failed: {e}")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/results")
def api_results():
    """Return the latest analysis JSON + FHIR refs."""
    out = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            out["analysis"] = json.load(f)
    if os.path.exists(REFS_PATH):
        with open(REFS_PATH) as f:
            out["fhir_refs"] = json.load(f)
    return jsonify(out)


@app.route("/api/fhir/patients")
def api_fhir_patients():
    """Proxy: list all patients from IRIS FHIR server."""
    try:
        resp = http_requests.get(
            f"{FHIR_BASE}/Patient",
            auth=FHIR_AUTH,
            headers={"Accept": "application/fhir+json"},
            timeout=5,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/fhir/report/<report_id>")
def api_fhir_report(report_id):
    """Proxy: fetch a single DiagnosticReport from IRIS."""
    try:
        resp = http_requests.get(
            f"{FHIR_BASE}/DiagnosticReport/{report_id}",
            auth=FHIR_AUTH,
            headers={"Accept": "application/fhir+json"},
            timeout=5,
        )
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/patients")
def api_patients():
    """Return all patients stored in the IRIS vector table."""
    try:
        from vector_search import GaitVectorStore
        store = GaitVectorStore(**IRIS_CONFIG)
        patients = store.get_all_patients()
        store.close()
        return jsonify({"patients": patients})
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/patient/<patient_id>")
def api_patient(patient_id):
    """Return the most recent analysis for a specific patient ID."""
    try:
        from vector_search import GaitVectorStore
        store = GaitVectorStore(**IRIS_CONFIG)
        data = store.get_patient_analysis(patient_id)
        store.close()
        if not data:
            return jsonify({"error": "Patient not found"}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/search", methods=["POST"])
def api_search():
    """Run semantic vector search against IRIS."""
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        from vector_search import GaitVectorStore
        store = GaitVectorStore(**IRIS_CONFIG)
        results = store.semantic_search(query, top_k=5)
        store.close()
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/images/<path:filename>")
def serve_image(filename):
    """Serve visualization PNGs."""
    return send_from_directory(VIZ_DIR, filename)


@app.route("/api/patient/<patient_id>/viz/<img_type>.png")
def patient_viz(patient_id, img_type):
    """Generate (and cache) a per-patient visualization PNG from stored metrics."""
    if img_type not in ("heatmap", "gait_analysis"):
        return "Not found", 404

    import re, tempfile
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", patient_id)
    cache_path = os.path.join(VIZ_DIR, f"{safe_id}_{img_type}.png")

    if os.path.exists(cache_path):
        return send_from_directory(VIZ_DIR, f"{safe_id}_{img_type}.png")

    # Fetch stored metrics for this patient
    try:
        from vector_search import GaitVectorStore
        store = GaitVectorStore(**IRIS_CONFIG)
        data = store.get_patient_analysis(patient_id)
        store.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 503

    if not data:
        return "Patient not found", 404

    # Derive CSV generation params from stored summary metrics
    cadence     = float(data.get("cadence", 100))
    cadence_hz  = max(0.4, min(cadence / 120.0, 2.0))   # steps/min → stride Hz
    max_L       = float(data.get("max_force_left",  350))
    max_R       = float(data.get("max_force_right", 350))
    bw          = 700.0
    # The stamp function peaks at ~0.51*bw*load_factor (toe-dominated phase),
    # so calibrate: load_factor = stored_peak / (0.51 * bw)
    # This ensures the generated CSV reproduces forces matching the stored values.
    MODEL_PEAK_FRACTION = 1.02
    left_load   = max(0.1, max_L / (MODEL_PEAK_FRACTION * bw))
    right_load  = max(0.1, max_R / (MODEL_PEAK_FRACTION * bw))
    seed        = abs(hash(patient_id)) % (2 ** 31)

    try:
        sys.path.insert(0, BASE_DIR)
        from seed_patients import generate_gait_csv

        import matplotlib
        matplotlib.use("Agg")
        os.makedirs(VIZ_DIR, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            generate_gait_csv(
                csv_path,
                body_weight_N=bw,
                cadence_hz=cadence_hz,
                left_load_factor=left_load,
                right_load_factor=right_load,
                seed=seed,
            )
            sys.path.insert(0, ANALYSIS_DIR)
            if img_type == "heatmap":
                from heatmap_viz import render as render_heatmap
                render_heatmap(csv_path, cache_path, body_weight_N=bw,
                               peak_force_left_N=max_L, peak_force_right_N=max_R)
            else:
                from gait_viz import render as render_gait
                render_gait(csv_path, cache_path)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

        return send_from_directory(VIZ_DIR, f"{safe_id}_{img_type}.png")
    except Exception as e:
        print(f"  ⚠ viz generation failed for {patient_id}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Insole Gait Intelligence Dashboard")
    print("  http://localhost:5050")
    print("=" * 55)
    ensure_visualizations()
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
