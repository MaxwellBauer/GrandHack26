"""
FHIR Resource Builder for Smart Insole Gait Analysis
=====================================================
Converts gait analysis results into HL7 FHIR R4 resources:
- Patient
- Device (the smart insole)
- Observation (one per metric: symmetry, cadence, peak force, etc.)
- DiagnosticReport (bundles observations + Claude's clinical narrative)

These resources are then POSTed to the InterSystems IRIS FHIR server.
"""

import json
import uuid
import requests
from datetime import datetime, timezone
from typing import Optional


# ── FHIR Server Config ────────────────────────────────────────────
# Default InterSystems IRIS FHIR endpoint from the hackathon kit
# Update these to match your Docker container's exposed ports
FHIR_BASE_URL = "http://localhost:32783/csp/healthshare/demo/fhir/r4"
FHIR_AUTH = ("_SYSTEM", "ISCDEMO")  # Hackathon kit credentials


def generate_id() -> str:
    return str(uuid.uuid4())


# ── Resource Builders ─────────────────────────────────────────────

def build_patient(patient_context: dict) -> dict:
    """Create a FHIR Patient resource."""
    patient_id = generate_id()
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [{
            "system": "http://smartinsole.example.com/patients",
            "value": patient_context.get("patient_id", f"SI-{patient_id[:8]}")
        }],
        "name": [{
            "use": "official",
            "family": patient_context.get("last_name", "Demo"),
            "given": [patient_context.get("first_name", "Patient")]
        }],
        "gender": patient_context.get("sex", "unknown"),
        "birthDate": patient_context.get("birth_date", "1967-03-15"),
        "extension": [{
            "url": "http://hl7.org/fhir/StructureDefinition/patient-bodyWeight",
            "valueQuantity": {
                "value": patient_context.get("weight_kg", 80),
                "unit": "kg",
                "system": "http://unitsofmeasure.org",
                "code": "kg"
            }
        }]
    }


def build_device() -> dict:
    """Create a FHIR Device resource for the smart insole."""
    return {
        "resourceType": "Device",
        "id": generate_id(),
        "identifier": [{
            "system": "http://smartinsole.example.com/devices",
            "value": "INSOLE-PROTO-001"
        }],
        "status": "active",
        "manufacturer": "SmartInsole Hackathon Team",
        "deviceName": [{
            "name": "Smart Pressure Insole v1",
            "type": "user-friendly-name"
        }],
        "type": {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": "706767009",
                "display": "Body worn patient sensor"
            }],
            "text": "Smart pressure-sensing insole with 14 force sensors"
        },
        "property": [
            {
                "type": {
                    "coding": [{"system": "http://smartinsole.example.com/device-properties",
                               "code": "sensor-count"}],
                    "text": "Number of pressure sensors"
                },
                "valueQuantity": [{"value": 14}]
            },
            {
                "type": {
                    "coding": [{"system": "http://smartinsole.example.com/device-properties",
                               "code": "sampling-rate"}],
                    "text": "Sampling rate"
                },
                "valueQuantity": [{"value": 100, "unit": "Hz"}]
            }
        ]
    }


def build_observation(patient_ref: str, device_ref: str,
                      code_system: str, code_value: str, display: str,
                      value: float, unit: str, unit_code: str,
                      interpretation_code: Optional[str] = None,
                      note: Optional[str] = None,
                      effective_dt: Optional[str] = None) -> dict:
    """Create a FHIR Observation for a single gait metric."""
    obs = {
        "resourceType": "Observation",
        "id": generate_id(),
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "exam",
                "display": "Exam"
            }]
        }],
        "code": {
            "coding": [{
                "system": code_system,
                "code": code_value,
                "display": display
            }],
            "text": display
        },
        "subject": {"reference": patient_ref},
        "device": {"reference": device_ref},
        "effectiveDateTime": effective_dt or datetime.now(timezone.utc).isoformat(),
        "valueQuantity": {
            "value": round(value, 2),
            "unit": unit,
            "system": "http://unitsofmeasure.org",
            "code": unit_code
        }
    }

    if interpretation_code:
        obs["interpretation"] = [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code": interpretation_code
            }]
        }]

    if note:
        obs["note"] = [{"text": note}]

    return obs


def build_gait_observations(analysis_results: dict,
                            patient_ref: str,
                            device_ref: str) -> list[dict]:
    """
    Build all FHIR Observations from gait analysis results.
    Uses SNOMED CT codes where available, custom codes for insole-specific metrics.
    """
    metrics = analysis_results["raw_metrics"]
    interp = analysis_results.get("clinical_interpretation", {})
    dt = datetime.now(timezone.utc).isoformat()

    CUSTOM_SYSTEM = "http://smartinsole.example.com/gait-metrics"
    SNOMED = "http://snomed.info/sct"
    LOINC = "http://loinc.org"

    observations = []

    # 1. Cadence
    observations.append(build_observation(
        patient_ref, device_ref,
        SNOMED, "364674007", "Gait cadence",
        metrics["cadence_steps_per_min"], "steps/min", "/min",
        note=f"Measured over {metrics['recording_duration_s']}s recording",
        effective_dt=dt
    ))

    # 2. Force symmetry index
    si = metrics["symmetry_index_force"]
    interp_code = "N" if si < 10 else ("H" if si < 20 else "HH")
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "force-symmetry-index", "Force symmetry index",
        si, "%", "%",
        interpretation_code=interp_code,
        note="0% = perfect symmetry. >15% clinically significant.",
        effective_dt=dt
    ))

    # 3. Timing symmetry index
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "timing-symmetry-index", "Stride timing symmetry index",
        metrics["symmetry_index_timing"], "%", "%",
        effective_dt=dt
    ))

    # 4. Stance symmetry index
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stance-symmetry-index", "Stance phase symmetry index",
        metrics["symmetry_index_stance"], "%", "%",
        effective_dt=dt
    ))

    # 5. Peak force left
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "peak-force-left", "Peak ground reaction force (left)",
        metrics["avg_peak_force_left_N"], "N", "N",
        effective_dt=dt
    ))

    # 6. Peak force right
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "peak-force-right", "Peak ground reaction force (right)",
        metrics["avg_peak_force_right_N"], "N", "N",
        effective_dt=dt
    ))

    # 7. Stride duration left
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stride-duration-left", "Average stride duration (left)",
        metrics["avg_stride_duration_left_s"], "s", "s",
        effective_dt=dt
    ))

    # 8. Stride duration right
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stride-duration-right", "Average stride duration (right)",
        metrics["avg_stride_duration_right_s"], "s", "s",
        effective_dt=dt
    ))

    # 9. Stance % left
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stance-pct-left", "Stance phase percentage (left)",
        metrics["avg_stance_pct_left"], "%", "%",
        effective_dt=dt
    ))

    # 10. Stance % right
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stance-pct-right", "Stance phase percentage (right)",
        metrics["avg_stance_pct_right"], "%", "%",
        effective_dt=dt
    ))

    # 11. Loading rate left
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "loading-rate-left", "Heel loading rate (left)",
        metrics["avg_loading_rate_left"], "N/s", "N/s",
        effective_dt=dt
    ))

    # 12. Loading rate right
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "loading-rate-right", "Heel loading rate (right)",
        metrics["avg_loading_rate_right"], "N/s", "N/s",
        effective_dt=dt
    ))

    # 13. Stride variability (CV) left
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stride-cv-left", "Stride duration variability (left, CV)",
        metrics["cv_stride_duration_left"], "ratio", "{ratio}",
        interpretation_code="H" if metrics["cv_stride_duration_left"] > 0.1 else "N",
        effective_dt=dt
    ))

    # 14. Stride variability (CV) right
    observations.append(build_observation(
        patient_ref, device_ref,
        CUSTOM_SYSTEM, "stride-cv-right", "Stride duration variability (right, CV)",
        metrics["cv_stride_duration_right"], "ratio", "{ratio}",
        interpretation_code="H" if metrics["cv_stride_duration_right"] > 0.1 else "N",
        effective_dt=dt
    ))

    return observations


def build_diagnostic_report(patient_ref: str,
                            observation_refs: list[str],
                            clinical_interpretation: dict) -> dict:
    """
    Create a FHIR DiagnosticReport that bundles all observations
    and includes Claude's clinical narrative.
    """
    summary = clinical_interpretation.get("clinical_summary", "")
    recommendations = clinical_interpretation.get("recommendations", [])
    risk_flags = clinical_interpretation.get("risk_flags", [])
    wb = clinical_interpretation.get("weight_bearing_assessment", {})
    sym = clinical_interpretation.get("symmetry_assessment", {})
    gait = clinical_interpretation.get("gait_quality", {})

    # Build narrative text
    narrative_parts = [
        f"<h3>Smart Insole Gait Analysis Report</h3>",
        f"<p><strong>Summary:</strong> {summary}</p>",
        f"<p><strong>Weight Bearing:</strong> {wb.get('compliance_level', 'N/A')} — {wb.get('notes', '')}</p>",
        f"<p><strong>Symmetry:</strong> {sym.get('overall', 'N/A')} — {sym.get('clinical_significance', '')}</p>",
        f"<p><strong>Gait Quality:</strong> Cadence {gait.get('cadence_assessment', 'N/A')}, "
        f"Variability {gait.get('stride_variability', 'N/A')}</p>",
    ]

    if risk_flags:
        narrative_parts.append(f"<p><strong>Risk Flags:</strong></p><ul>")
        for flag in risk_flags:
            narrative_parts.append(f"<li>{flag}</li>")
        narrative_parts.append("</ul>")

    if recommendations:
        narrative_parts.append(f"<p><strong>Recommendations:</strong></p><ul>")
        for rec in recommendations:
            narrative_parts.append(f"<li>{rec}</li>")
        narrative_parts.append("</ul>")

    narrative_html = "\n".join(narrative_parts)

    return {
        "resourceType": "DiagnosticReport",
        "id": generate_id(),
        "status": "final",
        "category": [{
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "PHY",
                "display": "Physical"
            }]
        }],
        "code": {
            "coding": [{
                "system": "http://smartinsole.example.com/report-types",
                "code": "gait-analysis",
                "display": "Smart Insole Gait Analysis"
            }],
            "text": "Smart Insole Gait Analysis Report"
        },
        "subject": {"reference": patient_ref},
        "effectiveDateTime": datetime.now(timezone.utc).isoformat(),
        "issued": datetime.now(timezone.utc).isoformat(),
        "result": [{"reference": ref} for ref in observation_refs],
        "conclusion": summary,
        "text": {
            "status": "generated",
            "div": f'<div xmlns="http://www.w3.org/1999/xhtml">{narrative_html}</div>'
        },
        "extension": [
            {
                "url": "http://smartinsole.example.com/extensions/follow-up-priority",
                "valueCode": clinical_interpretation.get("follow_up_priority", "routine")
            },
            {
                "url": "http://smartinsole.example.com/extensions/recovery-stage",
                "valueCode": clinical_interpretation.get("recovery_stage_estimate", "unknown")
            }
        ]
    }


# ── FHIR Server Interaction ──────────────────────────────────────

def post_resource(resource: dict, fhir_base: str = FHIR_BASE_URL) -> dict:
    """POST a FHIR resource to the server. Returns the server response."""
    resource_type = resource["resourceType"]
    url = f"{fhir_base}/{resource_type}"

    response = requests.post(
        url,
        json=resource,
        headers={
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        },
        auth=FHIR_AUTH,
    )

    if response.status_code in (200, 201):
        # IRIS returns 201 with empty body — extract ID from Location header
        if response.text.strip():
            result = response.json()
            server_id = result.get("id", resource.get("id"))
        else:
            location = response.headers.get("Location", "")
            # Location: .../ResourceType/123/_history/1 → grab the ID segment
            parts = [p for p in location.split("/") if p and p != "_history"]
            server_id = parts[-1] if parts else resource.get("id", "unknown")
            result = {"id": server_id, "resourceType": resource_type}
        print(f"  ✓ {resource_type}/{server_id} created")
        return result
    else:
        print(f"  ✗ {resource_type} failed: {response.status_code}")
        print(f"    {response.text[:500]}")
        return {"error": response.status_code, "body": response.text}


def post_bundle(resources: list[dict], fhir_base: str = FHIR_BASE_URL) -> dict:
    """POST a FHIR Bundle (transaction) containing multiple resources."""
    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": []
    }

    for resource in resources:
        rt = resource["resourceType"]
        rid = resource.get("id", generate_id())
        bundle["entry"].append({
            "fullUrl": f"urn:uuid:{rid}",
            "resource": resource,
            "request": {
                "method": "POST",
                "url": rt
            }
        })

    response = requests.post(
        fhir_base,
        json=bundle,
        headers={
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        },
        auth=FHIR_AUTH,
    )

    if response.status_code in (200, 201):
        print(f"  ✓ Bundle with {len(resources)} resources posted successfully")
        return response.json()
    else:
        print(f"  ✗ Bundle failed: {response.status_code}")
        print(f"    {response.text[:500]}")
        return {"error": response.status_code}


def push_gait_analysis_to_fhir(analysis_results: dict,
                                patient_context: dict,
                                fhir_base: str = FHIR_BASE_URL) -> dict:
    """
    Full pipeline: build FHIR resources from gait analysis and POST them.
    Returns dict with all created resource references.
    """
    print("\n═══ Pushing to FHIR Server ═══")

    # 1. Create Patient
    print("Creating Patient...")
    patient = build_patient(patient_context)
    patient_result = post_resource(patient, fhir_base)
    patient_id = patient_result.get("id", patient["id"])
    patient_ref = f"Patient/{patient_id}"

    # 2. Create Device
    print("Creating Device...")
    device = build_device()
    device_result = post_resource(device, fhir_base)
    device_id = device_result.get("id", device["id"])
    device_ref = f"Device/{device_id}"

    # 3. Create Observations
    print("Creating Observations...")
    observations = build_gait_observations(analysis_results, patient_ref, device_ref)
    obs_refs = []
    for obs in observations:
        result = post_resource(obs, fhir_base)
        obs_id = result.get("id", obs["id"])
        obs_refs.append(f"Observation/{obs_id}")

    # 4. Create DiagnosticReport
    interp = analysis_results.get("clinical_interpretation", {})
    if interp:
        print("Creating DiagnosticReport...")
        report = build_diagnostic_report(patient_ref, obs_refs, interp)
        report_result = post_resource(report, fhir_base)
        report_id = report_result.get("id", report["id"])
    else:
        report_id = None

    print(f"\n═══ Done ═══")
    print(f"  Patient:    {patient_ref}")
    print(f"  Device:     {device_ref}")
    print(f"  Observations: {len(obs_refs)}")
    print(f"  Report:     DiagnosticReport/{report_id}")

    return {
        "patient_ref": patient_ref,
        "device_ref": device_ref,
        "observation_refs": obs_refs,
        "report_ref": f"DiagnosticReport/{report_id}" if report_id else None,
        "patient_id": patient_context.get("patient_id", patient_id),
    }


# ── CLI ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fhir_builder.py <gait_analysis_results.json> [fhir_base_url]")
        sys.exit(1)

    results_path = sys.argv[1]
    fhir_base = sys.argv[2] if len(sys.argv) > 2 else FHIR_BASE_URL

    with open(results_path) as f:
        results = json.load(f)

    patient_ctx = {
        "patient_id": "SI-DEMO-001",
        "first_name": "Jane",
        "last_name": "Smith",
        "sex": "female",
        "birth_date": "1967-03-15",
        "weight_kg": 82,
    }

    refs = push_gait_analysis_to_fhir(results, patient_ctx, fhir_base)
    print(json.dumps(refs, indent=2))
