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

# Make src/ importable so we can use vector_search, gait_analysis, etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import requests as http_requests

app = Flask(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RESULTS_PATH = os.path.join(DATA_DIR, "gait_analysis_results.json")
REFS_PATH    = os.path.join(DATA_DIR, "gait_analysis_results_fhir_refs.json")
VIZ_DIR      = os.path.join(BASE_DIR, "..", "data")   # Part 1 PNG outputs

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


# ── Routes ──────────────────────────────────────────────────────────────────

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
    """Serve Part 1 visualization PNGs from the parent project's data/ dir."""
    return send_from_directory(VIZ_DIR, filename)


if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Insole Gait Intelligence Dashboard")
    print("  http://localhost:5050")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5050, debug=False)
