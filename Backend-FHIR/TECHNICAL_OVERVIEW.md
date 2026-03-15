# Smart Insole Gait Intelligence — Technical Overview

> **Hackathon project:** Post-surgical rehabilitation monitoring using pressure-sensing insoles, biomechanical gait analysis, Claude AI interpretation, HL7 FHIR R4 interoperability, and InterSystems IRIS vector search.

---

## Table of Contents

1. [Clinical Motivation](#1-clinical-motivation)
2. [Hardware Architecture](#2-hardware-architecture)
3. [Step 1 — Gait Analysis](#3-step-1--gait-analysis-gait_analysispy)
4. [Step 2 — Claude AI Interpretation](#4-step-2--claude-ai-interpretation)
5. [Step 3 — FHIR R4 Resources](#5-step-3--fhir-r4-resources-fhir_builderpy)
6. [Step 4 — InterSystems IRIS Vector Search](#6-step-4--intersystems-iris-vector-search-vector_searchpy)
7. [Complete Data Flow](#7-the-complete-data-flow)
8. [Key Design Decisions](#8-key-design-decisions)
9. [How to Run Everything](#9-how-to-run-everything)
10. [File Reference](#10-file-reference)

---

## 1. Clinical Motivation

Post-surgical orthopedic patients are routinely given weight-bearing (WB) restrictions after procedures such as tibial plateau fracture repair, ankle fusion, or total knee arthroplasty. Restrictions like "toe-touch weight bearing" (TTWB, ≤20% body weight) or "partial weight bearing" (PWB, ≤50% BW) exist because premature or excessive loading is a leading cause of hardware failure, delayed union, and the need for revision surgery. The problem is that compliance monitoring has historically been almost entirely subjective: a surgeon asks the patient "are you following your weight-bearing restrictions?" at a clinic visit every two to four weeks, and the patient reports their best recollection. Research consistently shows that patients dramatically underestimate how much force they are placing on a recovering limb — not from negligence, but because the neuromuscular compensation is largely unconscious. A patient with a right tibial fracture learns to limp with perfect timing but still loads the right foot at 40% body weight when they believe they are barely touching it.

Continuous sensor-based monitoring changes this entirely. By embedding calibrated pressure sensors in an insole and logging at 100 Hz, we capture the true ground-reaction-force profile of every step a patient takes throughout their day. This is not a self-report; it is an objective biomechanical record. The data can immediately flag WB violations, detect dangerous asymmetry patterns before they manifest as clinical complications, and build a longitudinal picture of how gait evolves through recovery. Historically, the infrastructure to close this loop — sensor hardware, data pipelines, clinical interpretation, and EHR integration — would require a multi-year research program. This project demonstrates that modern APIs and standards (Claude, FHIR R4, InterSystems IRIS) allow a small team to build a clinically meaningful prototype in a single hackathon.

The specific demo scenario in this project is a 58-year-old female patient (Jane Smith) who sustained a right tibial plateau fracture and underwent Open Reduction Internal Fixation (ORIF) with a locking plate. At 14 days post-operative, her surgeon has prescribed toe-touch weight bearing — defined as a maximum of 20% of body weight. At 82 kg, 20% BW corresponds to approximately 164 N. The smart insole data analyzed in this demo shows that her right foot is loading to an average peak of 261 N and a maximum of 268 N per step — roughly 33% of body weight, or 60% above her prescribed limit. Her left (unaffected) foot simultaneously shows 361 N average peak force, reflecting the classic antalgic compensation pattern where the patient offloads the injured side by weighting the healthy limb. Without objective measurement, neither the patient nor her care team would be aware of this violation occurring at home.

---

## 2. Hardware Architecture

### Sensor Layout

The system uses 14 force sensors arranged across both insoles — 7 per foot. The sensors are positioned to capture the clinically meaningful regions of the plantar surface: heel (medial and lateral subdivisions), midfoot (medial and lateral), and forefoot/toe (medial, central, lateral).

```
LEFT FOOT                          RIGHT FOOT

   Toe regions                        Toe regions
  ┌──────────┐                       ┌──────────┐
  │ lf_toe_l │                       │ rf_toe_l │
  │ lf_toe_c │                       │ rf_toe_c │
  │ lf_toe_r │                       │ rf_toe_r │
  ├──────────┤                       ├──────────┤
  │ lf_mid_l │  Midfoot              │ rf_mid_l │  Midfoot
  │ lf_mid_r │                       │ rf_mid_r │
  ├──────────┤                       ├──────────┤
  │ lf_heel_l│  Heel                 │ rf_heel_l│  Heel
  │ lf_heel_r│                       │ rf_heel_r│
  └──────────┘                       └──────────┘

Column naming convention: lf = left foot, rf = right foot
Subdivision: _l = medial (left side of sensor array), _r = lateral
```

All 14 channels are sampled simultaneously at **100 Hz** (one sample every 10 ms). Forces are calibrated and stored in **Newtons**. At 100 Hz over a 30-second recording, the result is 3,000 rows × 15 columns (1 timestamp + 14 sensor values).

### Data Format

The raw data file is a plain CSV:

```
timestamp_s,lf_heel_l_N,lf_heel_r_N,lf_mid_l_N,lf_mid_r_N,lf_toe_l_N,lf_toe_c_N,lf_toe_r_N,rf_heel_l_N,rf_heel_r_N,rf_mid_l_N,rf_mid_r_N,rf_toe_l_N,rf_toe_c_N,rf_toe_r_N
0.000,12.3,10.1,4.5,3.2,0.0,0.1,0.0,8.4,7.9,2.1,1.8,0.0,0.0,0.0
0.010,18.7,15.2,...
```

The `timestamp_s` column is wall-clock seconds from the start of recording. The sensor columns carry instantaneous force in Newtons at that sample.

### Simulated Data

In this demo, `generate_data.py` synthesizes realistic gait data using Gaussian pulse physics. Each heel-strike event is modeled as a Gaussian impulse (capturing the force ramp-up and ramp-down of a real step). The right foot pulses are intentionally attenuated to ~70% of the left foot amplitude to simulate the antalgic compensation pattern of the demo patient. Random timing jitter and sensor noise are added to replicate realistic measurement conditions. The result is statistically indistinguishable from real insole data for the purposes of demonstrating the analytical pipeline.

---

## 3. Step 1 — Gait Analysis (`gait_analysis.py`)

The gait analysis module converts raw sensor CSV data into clinically interpretable biomechanical metrics. The pipeline has five sub-steps.

### a. Data Loading

```
CSV file  →  list[dict]  →  float values
```

The CSV is read row by row using Python's `csv.DictReader`. Each row is converted to a dictionary mapping column name to float value. Non-numeric values are defaulted to `0.0`. The result is a list of dicts that can be indexed by sensor name and timestamp.

### b. Stride Detection

Gait analysis begins with detecting individual strides — a stride being one complete gait cycle from one heel-strike to the next heel-strike on the same foot.

**Algorithm:**

1. **Heel force trace** — for each foot, sum the two heel sensors at every sample to produce a single time-series representing heel loading over time. For the left foot: `heel_trace[i] = lf_heel_l_N[i] + lf_heel_r_N[i]`.

2. **Local maxima detection** — iterate through the heel trace and identify local maxima: samples where the force is greater than both immediate neighbors AND both second-order neighbors. This five-point check (`heel[i] > heel[i-1]`, `heel[i] > heel[i+1]`, `heel[i] > heel[i-2]`) suppresses noise-induced false peaks.

3. **Threshold filtering** — only maxima above 50 N are accepted as heel strikes. This eliminates sensor drift and very light toe-tapping from counting as steps.

4. **Minimum distance** — a heel strike must be at least 40 samples (0.4 seconds) after the previous one. This prevents a single broad peak from being counted as multiple strikes and enforces a physiologically realistic minimum cadence of ~75 steps/min.

5. **Stride extraction** — each consecutive pair of peaks (peak N to peak N+1) defines a stride interval. Strides with unrealistic durations (< 0.3 s or > 2.5 s) are filtered out as artifacts.

The result is a list of `(start_index, end_index)` pairs, each representing one complete stride.

### c. Per-Stride Metrics

For each detected stride, the following metrics are computed from the sensor data within that stride window:

| Metric | Computation |
|---|---|
| **Duration** | `timestamps[end_idx] - timestamps[start_idx]` in seconds |
| **Stance phase** | Count of samples in the stride where total foot force > 5 N, multiplied by 0.01 s (100 Hz). Stance = foot is on the ground. |
| **Swing phase** | `stride_duration - stance_duration` |
| **Stance %** | `(stance_duration / stride_duration) × 100` |
| **Peak heel force** | Maximum of the heel-sensor sum over the stride window |
| **Peak toe force** | Maximum of the toe-sensor sum (3 toe sensors) over the stride window |
| **Peak total force** | Maximum of all 7 sensors summed over the stride window |
| **Loading rate** | `peak_heel_force / time_to_peak` in N/s — measures how aggressively the patient loads their heel at initial contact |
| **Time to peak** | Samples from stride start to the peak heel force sample × 0.01 s |

These are stored as `StrideMetrics` dataclass instances, one per stride per foot.

### d. Summary Computation

After all strides are detected and measured, aggregate metrics are computed across the full recording session.

**Averages:** Standard means of duration, stance percentage, peak force, and loading rate across all left strides and all right strides separately.

**Cadence:**
```
cadence = (total_steps / recording_duration_s) × 60
```
Total steps = left strides + right strides + 1 (to account for the final incomplete stride).

**Symmetry Index (SI):** Applied to three paired metrics — force, stride timing, and stance percentage.
```
SI = 2 × |A − B| / (A + B) × 100%
```
Where A and B are the average values for the left and right foot respectively. SI = 0% means perfect bilateral symmetry. SI > 15% for force, > 10% for timing, or > 8% for stance are flagged as clinically significant. The SI formula is borrowed from rehabilitation medicine literature (Robinson et al., 1987) and is widely used in gait lab assessments.

**Coefficient of Variation (CV):**
```
CV = standard_deviation / mean
```
Applied to stride duration for each foot separately. CV > 0.10 (10%) indicates excessive stride-to-stride variability — a sign of gait instability or fatigue. Normal healthy adults have CV < 0.03.

**Clinical flags:** The following conditions automatically add a flag string to `summary.flags`:
- Force SI > 15%: `"ASYMMETRIC_FORCE: SI=32.1% (threshold 15%)"`
- Timing SI > 10%: `"ASYMMETRIC_TIMING: SI=...%"`
- Stance SI > 8%: `"ASYMMETRIC_STANCE: SI=...%"`
- CV > 0.10 (either foot): `"HIGH_VARIABILITY_LEFT/RIGHT: CV=..."`
- Cadence < 80: `"LOW_CADENCE: ... steps/min"`
- Cadence > 140: `"HIGH_CADENCE: ... steps/min"`

### e. Results for the Demo Patient

| Metric | Left Foot | Right Foot | Clinical Note |
|--------|-----------|------------|---------------|
| Strides detected | 29 | 29 | Equal count — symmetric cadence |
| Avg peak force | **361.0 N** | **261.1 N** | Left bears ~38% more load |
| Max peak force | 367.5 N | 268.6 N | Right exceeds 164 N WB limit by ~64% |
| Avg stance phase | **79.7%** | **74.6%** | Left compensates — longer contact |
| Avg stride duration | 1.00 s | 1.00 s | Timing perfectly symmetric |
| Avg loading rate | 87.5 N/s | 70.3 N/s | Left impacts harder |
| Stride variability (CV) | 0.008 | 0.009 | Very low — consistent pattern |
| Cadence | 116.0 steps/min | — | Normal range (80–130) |
| **Force SI** | **32.1%** | — | **Well above 15% threshold — FLAGGED** |
| Timing SI | 0.0% | — | Perfect timing symmetry |
| Stance SI | 6.7% | — | Below 8% threshold — acceptable |

The clinical picture is a textbook antalgic gait: the patient has preserved timing symmetry (the two feet alternate with perfect rhythm) but dramatically unequal loading. The right foot is offloaded not by taking fewer steps, but by pushing off with less force. This is the most common compensation pattern after lower-limb injury and is exactly what the force SI is designed to detect.

---

## 4. Step 2 — Claude AI Interpretation

### What Claude Receives

After gait metrics are computed, the full `GaitSummary` dataclass is serialized to JSON and sent to the Claude API along with the patient's clinical context. The structured patient context includes:

- Age, sex, body weight
- Diagnosis and surgical procedure
- Current weight-bearing restriction (e.g., "toe-touch weight bearing (max 20% BW)")
- Days post-operative

This context is critical: a 32.1% force symmetry index means something very different for a patient on day 14 post-ORIF (alarming, requires immediate review) vs. a patient three months after ACL reconstruction (expected, part of normal recovery progression).

### System Prompt Design

Claude is given the role of a **clinical biomechanics expert** with this system prompt:

> "You are a clinical biomechanics expert analyzing smart insole gait data for post-surgical rehabilitation monitoring. You provide structured gait analysis reports for orthopedic surgeons and physical therapists. Your analysis should be evidence-based and clinically actionable, focused on weight-bearing compliance, symmetry, and safety. Return ONLY valid JSON."

The structured-output-only constraint is enforced at the system level rather than through a tool call, which gives more flexibility in the model's reasoning before formatting the response. The `response_text` is cleaned of any Markdown code fences before JSON parsing.

### Output Schema

```json
{
  "clinical_summary": "2-3 sentence narrative of the gait pattern",
  "weight_bearing_assessment": {
    "compliance_level": "compliant | partial | non_compliant | exceeding",
    "max_force_pct_body_weight": 32.7,
    "notes": "Patient is loading to ~33% BW, exceeding 20% TTWB restriction"
  },
  "symmetry_assessment": {
    "overall": "severely_asymmetric",
    "force_symmetry_notes": "32.1% SI — well above clinical threshold",
    "timing_symmetry_notes": "0.0% SI — timing is preserved",
    "clinical_significance": "Antalgic pattern: offloading right, overloading left"
  },
  "gait_quality": {
    "cadence_assessment": "normal",
    "stride_variability": "normal",
    "stance_swing_ratio": "abnormal",
    "notes": "..."
  },
  "risk_flags": ["Weight bearing violation: 261N on restricted limb", "..."],
  "recommendations": ["Immediate physical therapy review", "..."],
  "recovery_stage_estimate": "early",
  "follow_up_priority": "urgent | soon | routine"
}
```

### Why This Beats a Rule-Based System

A naive threshold system could flag "force > 164 N on right foot" and call it a WB violation. But clinical interpretation requires reasoning across the combination of factors simultaneously:

- Is the cadence normal? (Affects how the force distributes across the gait cycle)
- Is the timing symmetric even though force is not? (Suggests conscious compensation vs. inability to move the limb)
- What stage of recovery is the patient in? (Day 14 ORIF vs. day 90 total knee — same SI value, completely different meaning)
- What is the specific surgical hardware? (A locking plate tolerates different loads than a unicondylar knee replacement)
- Are the risk flags additive? (High force SI + high loading rate + stance asymmetry together = urgent; any one alone = routine)

Claude synthesizes all of these dimensions into human-readable clinical reasoning that an orthopedic surgeon can act on immediately, without needing to map code numbers back to clinical meaning. The recommendations are directly actionable by the care team ("Schedule PT visit within 48 hours for gait retraining", "Consider crutch technique review with PT") rather than generic alerts.

---

## 5. Step 3 — FHIR R4 Resources (`fhir_builder.py`)

### What is FHIR?

**FHIR** (Fast Healthcare Interoperability Resources, pronounced "fire") is the global standard for exchanging healthcare information electronically, maintained by HL7 International. Version R4 (Release 4, 2019) is the current production standard and is mandated by the US ONC 21st Century Cures Act for all certified EHRs.

FHIR's significance for this project: every major hospital EHR system — Epic, Oracle Cerner, Meditech, MEDITECH Expanse — can send and receive FHIR R4 resources over standard HTTP/JSON. By formatting our insole data as FHIR R4, a surgeon using any modern EHR can query a patient's gait analysis the same way they query a lab result, without any custom integration work. The insole data becomes a first-class clinical record.

### Resources Created

Four FHIR resource types are created for each gait analysis session.

#### Patient

The `Patient` resource is the demographic anchor for all other records. It stores the patient's legal name, date of birth, administrative sex, and body weight (via the HL7-defined `patient-bodyWeight` extension). Every other FHIR resource in the system contains a `subject.reference` pointing back to this `Patient` resource. The body weight extension is clinically important because weight-bearing percentages must be computed relative to the patient's actual body weight.

#### Device

The `Device` resource represents the smart insole hardware. It is assigned SNOMED CT code `706767009` ("Body worn patient sensor"), which is the most specific applicable standard code. The device record stores the number of sensors (14) and sampling rate (100 Hz) as structured properties. This allows future FHIR queries to filter measurements by device type or version — important when hardware is updated and clinicians need to know which sensor generation produced a given result.

#### Observation × 14

One `Observation` resource is created for each of the 14 gait metrics extracted from the analysis. The use of individual Observation resources rather than a single compound Observation is intentional and follows FHIR best practice: each Observation is independently queryable. A surgeon can retrieve only the `force-symmetry-index` Observations across all patients with a single FHIR API call, without fetching entire report bundles.

SNOMED CT codes are used where a standard code exists:
- `364674007` — Gait cadence

Custom codes under the `http://smartinsole.example.com/gait-metrics` system are used for insole-specific metrics that have no standard code yet:
- `force-symmetry-index` — bilateral force symmetry index (%)
- `timing-symmetry-index` — bilateral timing symmetry index (%)
- `stance-symmetry-index` — bilateral stance-phase symmetry index (%)
- `peak-force-left` / `peak-force-right` — average peak ground reaction force per foot (N)
- `stride-duration-left` / `stride-duration-right` — average stride duration per foot (s)
- `stance-pct-left` / `stance-pct-right` — average stance phase percentage per foot (%)
- `loading-rate-left` / `loading-rate-right` — heel loading rate per foot (N/s)
- `stride-cv-left` / `stride-cv-right` — stride duration variability, coefficient of variation

Each Observation also carries an `interpretation` coding using the standard HL7 `v3-ObservationInterpretation` code system: `N` (normal), `H` (high), or `HH` (critically high). The force symmetry index uses `HH` when SI > 20%, making it immediately visible in any EHR's abnormal-result display.

#### DiagnosticReport

The `DiagnosticReport` resource bundles all 14 Observation references and contains Claude's clinical narrative as its `conclusion` field. The `text.div` field contains the full narrative as rendered HTML — this is what an EHR displays when a clinician opens the report in a patient chart. Two extensions are added:

- `follow-up-priority` — the urgency code from Claude's interpretation (`urgent`, `soon`, `routine`)
- `recovery-stage` — Claude's estimate of recovery progression (`early`, `progressing`, `advanced`, `near_discharge`)

The DiagnosticReport is the resource a surgeon would request to see the complete picture; the Observations are for analytics and querying.

### IRIS FHIR Server Specifics

The InterSystems IRIS FHIR server is accessed at:
```
http://localhost:32783/csp/healthshare/demo/fhir/r4/{ResourceType}
```

Each resource is POSTed individually with:
- `Content-Type: application/fhir+json`
- `Accept: application/fhir+json`
- Basic authentication: `_SYSTEM` / `ISCDEMO`

IRIS returns HTTP 201 with an **empty body** on successful creation. The server-assigned resource ID is extracted from the `Location` response header, which has the format:
```
/csp/healthshare/demo/fhir/r4/Patient/123/_history/1
```

The ID segment is parsed by splitting on `/`, filtering empty strings and `_history`, and taking the last segment before the history version. This ID is then used to construct reference strings for subsequent resources (e.g., the Patient reference passed into each Observation).

---

## 6. Step 4 — InterSystems IRIS Vector Search (`vector_search.py`)

### What is Vector Search?

Vector search converts text into a point in a high-dimensional geometric space. An embedding model (a small neural network) processes a string and outputs a fixed-length array of floating-point numbers — the "embedding" or "vector" — that encodes the semantic meaning of the text. The key property is that semantically similar texts produce vectors that are geometrically close to each other, regardless of whether they share any words.

For example:
- "Patient is excessively loading the post-surgical limb" → vector [0.12, -0.34, 0.08, ...]
- "Right foot force exceeding weight-bearing restrictions" → vector [0.13, -0.31, 0.09, ...]

These two sentences use completely different words but describe the same clinical situation. Their vectors will be near each other in 384-dimensional space. A SQL `LIKE` query would find neither sentence when searching for "loading restrictions" — vector search finds both.

### The Embedding Model

The system uses `all-MiniLM-L6-v2` from the `sentence-transformers` library. Key properties:

- **Dimensions:** 384 (a compact but semantically rich representation)
- **Runs locally:** No API call, no cost, no latency beyond inference time (~50ms on CPU)
- **License:** Apache 2.0 — production-safe
- **Normalization:** Output vectors are L2-normalized to unit length, meaning cosine similarity equals the dot product — computationally simple and efficient

The text fed to the embedding model is a rich concatenation of the full clinical narrative: the `clinical_summary`, symmetry notes, weight-bearing notes, gait quality notes, all risk flags, and all recommendations. This ensures the embedding captures the full clinical meaning of the analysis, not just the summary sentence.

### IRIS Vector Column

The IRIS `GaitAnalysis` table uses a native vector column type:

```sql
narrative_vector VECTOR(DOUBLE, 384)
```

This is an InterSystems-native type that stores a fixed-length array of 64-bit doubles. It is distinct from a `VARBINARY` blob — IRIS understands its structure and can apply vector operations directly in SQL.

**Inserting a vector:**
```sql
INSERT INTO SQLUser.GaitAnalysis (..., narrative_vector)
VALUES (..., TO_VECTOR(?, double))
```
The `TO_VECTOR` function accepts a comma-separated string of float values and the type keyword `double`, and converts it to the native vector type.

**Querying by similarity:**
```sql
SELECT TOP 5 patient_id, clinical_summary,
    VECTOR_COSINE(narrative_vector, TO_VECTOR(?, double)) AS similarity
FROM SQLUser.GaitAnalysis
ORDER BY similarity DESC
```
`VECTOR_COSINE` returns a value between 0 and 1, where 1.0 is identical and 0.0 is orthogonal (completely unrelated). In practice, scores above 0.7 represent strong semantic matches.

An HNSW (Hierarchical Navigable Small World) index is created on the vector column for approximate nearest-neighbor search — this keeps query latency sub-millisecond even for large patient cohorts.

### The `GaitAnalysis` Table Schema

| Column | Type | Purpose |
|--------|------|---------|
| `id` | INT AUTO_INCREMENT | Primary key |
| `patient_id` | VARCHAR(100) | Application-level patient identifier (e.g., SI-DEMO-001) |
| `fhir_patient_ref` | VARCHAR(200) | FHIR Patient reference (e.g., Patient/abc123) — links search result to full FHIR record |
| `fhir_report_ref` | VARCHAR(200) | FHIR DiagnosticReport reference — direct link to full clinical report |
| `recording_date` | TIMESTAMP | When the gait session was analyzed |
| `clinical_summary` | CLOB | Claude's summary sentence — shown in search result cards |
| `full_narrative` | CLOB | Complete clinical narrative — the text that was embedded |
| `symmetry_status` | VARCHAR(50) | Claude's symmetry category (symmetric, mildly_asymmetric, etc.) — for display |
| `follow_up_priority` | VARCHAR(20) | urgent / soon / routine — for filtering and display |
| `recovery_stage` | VARCHAR(30) | early / progressing / advanced / near_discharge |
| `cadence` | FLOAT | Steps/min — enables structured filtering alongside semantic search |
| `force_symmetry_pct` | FLOAT | Force SI % — enables hybrid SQL+vector queries |
| `max_force_left` | FLOAT | Peak left foot force (N) |
| `max_force_right` | FLOAT | Peak right foot force (N) — key for WB compliance filtering |
| `narrative_vector` | VECTOR(DOUBLE, 384) | The semantic embedding of the full narrative |

The structured numeric columns alongside the vector column enable powerful hybrid queries: "find all patients who had urgent follow-up priority AND semantically similar gait patterns to this query." This combines the precision of SQL filters with the semantic flexibility of vector similarity.

### Clinical Power of This Approach

A physical therapist working with 200 post-surgical patients could search:

> "patients with antalgic gait who were offloading their right foot at 2 weeks post-op ankle surgery"

The vector search finds all stored analyses whose clinical narratives are semantically similar — even if the stored records say "avoidance loading pattern on the right side, tibial fracture, day 14" rather than matching the exact words of the query. The result includes the FHIR references for each match, so the therapist can click through to the full structured record in their EHR.

As more patients are added to the system, the search becomes a pattern library for rehabilitation medicine. A new patient with an unusual gait signature can be compared against all historical patients to find similar cases, see what interventions were recommended, and track what recovery timelines were achieved.

---

## 7. The Complete Data Flow

```
Smart Insole Hardware
(14 force sensors per shoe pair @ 100 Hz)
            │
            │ raw CSV (timestamp + 14 sensor columns)
            ▼
      gait_data.csv
      (3,000 rows × 15 cols, 30-second session)
            │
            ▼
    ┌───────────────────────────────────────────┐
    │           gait_analysis.py                │
    │                                           │
    │  1. load_gait_data()                      │
    │     CSV → list[dict] with float values    │
    │                                           │
    │  2. detect_strides() per foot             │
    │     • Sum heel sensors → heel trace       │
    │     • Find local maxima > 50 N            │
    │     • Min 40-sample spacing (0.4 s)       │
    │     • Filter durations 0.3–2.5 s          │
    │     → 29 left strides, 29 right strides   │
    │                                           │
    │  3. Per-stride StrideMetrics              │
    │     • Duration, stance %, swing %         │
    │     • Peak heel / toe / total force (N)   │
    │     • Loading rate (N/s), time to peak    │
    │                                           │
    │  4. compute_summary()                     │
    │     • Averages across all strides         │
    │     • Cadence: 116.0 steps/min            │
    │     • Force SI: 32.1% ← FLAGGED           │
    │     • Timing SI: 0.0%                     │
    │     • Stance SI: 6.7%                     │
    │     → GaitSummary dataclass               │
    └───────────────────────────────────────────┘
            │
            │ GaitSummary (JSON) + patient context
            ▼
    ┌───────────────────────────────────────────┐
    │     Claude claude-sonnet-4-6 API          │
    │                                           │
    │  System: clinical biomechanics expert     │
    │  Input:  metrics JSON + patient context   │
    │                                           │
    │  Output (structured JSON):                │
    │  • clinical_summary                       │
    │  • weight_bearing_assessment              │
    │    compliance_level: "exceeding"          │
    │    max_force_pct_bw: ~33%                 │
    │  • symmetry_assessment: severely_asymmetric│
    │  • gait_quality: cadence=normal           │
    │  • risk_flags[]                           │
    │  • recommendations[]                      │
    │  • follow_up_priority: "urgent"           │
    │  • recovery_stage_estimate: "early"       │
    └───────────────────────────────────────────┘
            │
            │ full analysis dict (metrics + interpretation)
            ├────────────────────┬──────────────────────┐
            ▼                    ▼                      ▼
    ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
    │ fhir_        │   │ vector_search.py │   │   Flask API     │
    │ builder.py   │   │                  │   │   (main.py)     │
    │              │   │ 1. Build full    │   │                 │
    │ 1. Patient   │   │    narrative     │   │ Serves results  │
    │ 2. Device    │   │    text          │   │ to dashboard    │
    │ 3. 14 Obs.   │   │ 2. Embed with   │   │ at /api/results │
    │ 4. Diag.     │   │    MiniLM-L6-v2 │   │                 │
    │    Report    │   │    384-dim vec   │   │ /api/search     │
    └──────────────┘   └──────────────────┘   └─────────────────┘
            │                    │
            │ HTTP POST          │ IRIS DB-API
            ▼                    ▼
    ┌────────────────────────────────────────────┐
    │         InterSystems IRIS                  │
    │                                            │
    │  FHIR R4 Server                            │
    │  ├─ Patient/abc123                         │
    │  ├─ Device/def456                          │
    │  ├─ Observation/... (×14)                  │
    │  └─ DiagnosticReport/ghi789               │
    │     Queryable by any FHIR-capable EHR     │
    │     (Epic, Cerner, Meditech, etc.)         │
    │                                            │
    │  Vector Search Table (SQLUser.GaitAnalysis)│
    │  ├─ id, patient_id, fhir_refs             │
    │  ├─ clinical_summary, full_narrative       │
    │  ├─ symmetry_status, follow_up_priority    │
    │  ├─ cadence, force_symmetry_pct           │
    │  └─ narrative_vector VECTOR(DOUBLE, 384)  │
    │       HNSW index                          │
    │       VECTOR_COSINE similarity search      │
    └────────────────────────────────────────────┘
            │
            ▼
    Web Dashboard (templates/index.html)
    ├─ Patient demographics
    ├─ Key metrics with colored indicators
    ├─ Claude AI analysis with risk flags
    ├─ Plantar pressure heatmap
    ├─ Gait force trace visualization
    ├─ FHIR resource status table
    └─ Natural language vector search UI
```

---

## 8. Key Design Decisions

### Why not use a FHIR transaction bundle for all resources?

FHIR supports posting a `Bundle` resource of type `transaction`, which atomically creates all resources in one HTTP request. We attempted this approach but encountered a specific IRIS behavior: the server returned HTTP 200 with an empty body for transaction bundles, providing no way to extract the server-assigned IDs for the created resources. Individual POSTing was more debuggable at hackathon pace — each resource returns a `Location` header with its new ID, which we can immediately use in subsequent requests. A production implementation would use a transaction bundle for atomicity guarantees (either all resources are created or none are), but for the demo, sequential individual POSTs with explicit ID tracking is simpler and more transparent.

### Why sentence-transformers over Claude embeddings?

Claude (Anthropic's API) does not offer an embeddings endpoint — it is a generative model only. Embedding generation requires a different model architecture (encoder-only transformers like BERT-family models). `all-MiniLM-L6-v2` from `sentence-transformers` was chosen because it runs entirely locally (no API key, no cost, no network dependency), loads in under 2 seconds on CPU, and produces 384-dimensional vectors that are well-calibrated for semantic similarity tasks in English clinical text. The 384-dimension size matches the `VECTOR(DOUBLE, 384)` column type exactly without wasting storage on dimensions beyond what the model produces.

### Why `VECTOR(DOUBLE, 384)` and not `VECTOR(FLOAT, 384)`?

The InterSystems IRIS hackathon kit documentation and examples use `DOUBLE` as the vector element type. The `TO_VECTOR(?, double)` SQL function requires the type keyword to match the column definition. Using `FLOAT` in the column definition while passing `double` to `TO_VECTOR` causes a type mismatch error at insertion time. `DOUBLE` (64-bit IEEE 754) is also more numerically stable for cosine similarity computation than `FLOAT` (32-bit), at the cost of exactly double the storage per vector. For 384-dimensional vectors the storage cost is 384 × 8 bytes = 3,072 bytes per patient — negligible.

### Why `python-dotenv`?

The `ANTHROPIC_API_KEY` is loaded from a `.env` file in the project root using `python-dotenv`. This keeps the API key out of three places it should never appear: the shell history (which persists across sessions), the git repository (which may be public), and the process environment of child processes that don't need it. The `.env` file is added to `.gitignore`. At a hackathon where demos are screen-shared and repositories are sometimes accidentally made public, this is a simple but important safeguard.

### Why 14 separate Observations instead of one compound Observation?

FHIR's `Observation` resource supports "component" sub-observations that group related measurements. We could have created a single compound Observation with 14 components. We chose separate Observations for three reasons:

1. **Individual queryability.** A FHIR query like `GET /Observation?code=force-symmetry-index` returns only that specific metric across all patients, without fetching entire observation bundles. This is essential for population analytics ("show me the distribution of force symmetry index values across all day-14 post-op patients").

2. **Interpretation coding per observation.** Each Observation carries its own `interpretation` code (N / H / HH). A FHIR-aware EHR's abnormal-results display can immediately highlight the critically-high force symmetry index without parsing a compound structure.

3. **Standards alignment.** HL7 guidance recommends separate Observations when each measurement has distinct clinical meaning and could be queried or acted on independently. Compound Observations are recommended for measurements that are meaningless in isolation (e.g., blood pressure systolic/diastolic — you always want both).

---

## 9. How to Run Everything

### Prerequisites

- Python 3.11+
- Docker Desktop (for IRIS)
- An Anthropic API key (for Claude interpretation; pipeline runs in degraded mode without it)

### Step 1: Start InterSystems IRIS

```bash
cd Backend-FHIR
docker-compose up -d
```

Wait approximately 60–90 seconds for IRIS to initialize fully. Verify it is ready:

```bash
curl -s http://localhost:32783/csp/healthshare/demo/fhir/r4/metadata | python3 -m json.tool | head -20
```

You should see a FHIR CapabilityStatement. If you see a connection refused error, wait another 30 seconds and retry.

You can also check the IRIS Management Portal at:
```
http://localhost:32783/csp/sys/UtilHome.csp
Login: _SYSTEM / ISCDEMO
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `anthropic` — Claude API client
- `requests` — HTTP for FHIR REST calls
- `sentence-transformers` — local embedding model
- `python-dotenv` — .env file loading
- `flask` — the web dashboard API server

The IRIS DB-API driver must be installed separately from the hackathon kit wheel file:

```bash
# From the intersystems_irispython wheel included in the hackathon kit:
pip install ./install/intersystems_irispython-5.0.1-cp311-cp311-manylinux_2_17_x86_64.whl
# (select the wheel matching your OS and Python version)
```

### Step 3: Configure API Keys

Create a `.env` file in the `Backend-FHIR/` directory:

```bash
# Backend-FHIR/.env
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE

# Optional: override IRIS connection details
# IRIS_HOST=localhost
# IRIS_PORT=32782
# IRIS_NAMESPACE=DEMO
# IRIS_USER=_SYSTEM
# IRIS_PASSWORD=ISCDEMO
```

The `.env` file is gitignored. Never paste your API key directly into any source file.

### Step 4: Run the Pipeline

**Mode A — Full pipeline (analysis + FHIR + vector search):**
```bash
cd Backend-FHIR/src
python main.py --csv ../data/gait_data.csv
```

**Mode B — Analysis only (no IRIS required, useful for testing):**
```bash
python main.py --csv ../data/gait_data.csv --no-fhir --no-vector
```

**Mode C — Analysis + FHIR, skip vector search:**
```bash
python main.py --csv ../data/gait_data.csv --no-vector
```

**Mode D — Skip Claude interpretation (no API key needed):**
```bash
python main.py --csv ../data/gait_data.csv --no-claude --no-fhir --no-vector
```

The pipeline prints progress to stdout and saves:
- `data/gait_analysis_results.json` — full analysis output (metrics + interpretation)
- `data/gait_analysis_results_fhir_refs.json` — FHIR resource IDs returned by the server

### Step 5: Start the Web Dashboard

```bash
cd Backend-FHIR
python app.py
```

Open `http://localhost:5050` in your browser.

### Step 6: Run Vector Searches

Directly from the CLI:

```bash
cd Backend-FHIR/src

# Semantic search
python vector_search.py search "patients with asymmetric gait"
python vector_search.py search "high loading rate exceeding weight bearing limits"
python vector_search.py search "good symmetry normal recovery"
python vector_search.py search "early recovery, toe-touch restriction violated"

# List all stored analyses
python vector_search.py list
```

### Step 7: Verify FHIR Resources in IRIS

Browse the FHIR server directly:
```
http://localhost:32783/csp/healthshare/demo/fhir/r4/Patient
http://localhost:32783/csp/healthshare/demo/fhir/r4/Observation
http://localhost:32783/csp/healthshare/demo/fhir/r4/DiagnosticReport
```

Or query a specific patient's observations:
```
http://localhost:32783/csp/healthshare/demo/fhir/r4/Observation?subject=Patient/{id}
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on FHIR calls | IRIS is still starting — wait 60s and retry |
| `iris` module not found | Install the IRIS DB-API wheel for your OS/Python version |
| Empty `clinical_interpretation` in results | `ANTHROPIC_API_KEY` is not set — pipeline runs in degraded mode |
| `TO_VECTOR` SQL error | Check that the embedding dimension matches the `VECTOR(DOUBLE, 384)` column — if you changed models, also update `EMBEDDING_DIM` |
| 401 Unauthorized from FHIR server | Credentials are `_SYSTEM` / `ISCDEMO` — verify with the Management Portal |

---

## 10. File Reference

### `Backend-FHIR/` (project root)

| File | Description |
|------|-------------|
| `docker-compose.yml` | Defines the InterSystems IRIS container. Exposes port 32783 for FHIR HTTP and port 32782 for the DB-API TCP connection. Mounts local directories for persistence. |
| `requirements.txt` | Python dependency list: `anthropic`, `requests`, `sentence-transformers`, `torch`, `flask`, `python-dotenv`. |
| `README.md` | Quick-start instructions, architecture summary, demo tips. |
| `TECHNICAL_OVERVIEW.md` | This document. Full technical explanation of every component. |
| `smart-insole-fhir.tar.gz` | Snapshot archive of the project for submission. |

### `Backend-FHIR/src/`

| File | Description |
|------|-------------|
| `main.py` | Pipeline orchestrator. Parses CLI arguments (`--no-fhir`, `--no-vector`, `--no-claude`, `--csv`). Calls `analyze_gait_data`, `push_gait_analysis_to_fhir`, and `GaitVectorStore.store_analysis` in sequence. |
| `../app.py` | Flask web server (`python app.py` from `Backend-FHIR/`). Serves the dashboard at `GET /`, proxies FHIR queries to IRIS at `/api/fhir/*`, runs vector semantic search at `POST /api/search`, and serves Part 1 visualization PNGs at `/images/<filename>`. |
| `gait_analysis.py` | Core biomechanics module. Defines `StrideMetrics` and `GaitSummary` dataclasses. Implements `load_gait_data`, `detect_strides`, `compute_summary`, `interpret_with_claude`, and `analyze_gait_data`. The Claude API call and structured JSON response parsing live here. |
| `fhir_builder.py` | FHIR R4 resource construction and HTTP posting. Builds `Patient`, `Device`, `Observation` (×14), and `DiagnosticReport` resources as Python dicts conforming to the FHIR R4 JSON schema. Posts each to IRIS via `requests.post`. Extracts server-assigned IDs from `Location` response headers. |
| `vector_search.py` | InterSystems IRIS vector search integration. The `GaitVectorStore` class connects to IRIS via the DB-API, creates the `GaitAnalysis` table with `VECTOR(DOUBLE, 384)` column, stores analyses with `TO_VECTOR`, and performs similarity search with `VECTOR_COSINE`. Embedding generation via `sentence-transformers all-MiniLM-L6-v2`. |

### `Backend-FHIR/src/data/`

| File | Description |
|------|-------------|
| `gait_data.csv` | Synthetic 30-second gait recording. 3,000 rows × 15 columns (timestamp + 14 sensor values in Newtons). Generated by `generate_data.py` using Gaussian pulse physics with realistic asymmetry for the demo patient. |
| `gait_analysis_results.json` | Output of `analyze_gait_data()`. Contains `raw_metrics` (GaitSummary as dict), `left_strides` and `right_strides` (lists of StrideMetrics dicts), and `clinical_interpretation` (Claude's structured JSON output). |
| `gait_analysis_results_fhir_refs.json` | Output of `push_gait_analysis_to_fhir()`. Contains `patient_ref`, `device_ref`, `observation_refs` (list of 14), `report_ref`, and `patient_id`. Used by the vector search store step and the web dashboard. |
| `sample_fhir_bundle.json` | Example FHIR Bundle showing the resource structure for reference and testing. |

### `Backend-FHIR/templates/`

| File | Description |
|------|-------------|
| `index.html` | Complete single-page web dashboard. Dark-theme responsive layout. Fetches `/api/results` on load to populate patient demographics, key metrics with visual indicators, Claude AI analysis with risk flags, FHIR resource status table, and JSON preview. Loads gait visualization images from `/images/`. Contains the natural language vector search UI with example chips, async POST to `/api/search`, and result cards with similarity bars. No external dependencies — pure HTML/CSS/JS. |

### `Backend-FHIR/Analysis/` (if present)

| File | Description |
|------|-------------|
| `generate_data.py` | Synthetic gait data generator. Uses Gaussian pulse functions to simulate heel-strike patterns at 100 Hz. Right foot amplitude is attenuated by ~70% to simulate the demo patient's antalgic compensation. Adds Gaussian noise and random timing jitter. Outputs `gait_data.csv`. |
| `visualize_gait.py` | Generates the two visualization images served by the dashboard: (1) plantar pressure heatmap using RBF interpolation across the 14 sensor locations over a foot outline, (2) gait analysis chart showing force traces, detected stride peaks, stance/swing intervals, and bilateral symmetry comparison. Outputs `images/heatmap.png` and `images/gait_analysis.png`. |
