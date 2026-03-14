# Smart Insole — FHIR + Vector Search Gait Analysis

**Hackathon project: Turning raw pressure sensor data into clinically actionable, FHIR-interoperable gait intelligence.**

## What This Does

A post-surgical patient wears pressure-sensing insoles (14 sensors per pair, 100Hz). This pipeline:

1. **Analyzes** raw sensor CSV → detects individual strides, computes biomechanical metrics
2. **Interprets** via Claude API → generates clinical narrative with risk flags and recommendations  
3. **Structures** into FHIR R4 resources → Patient, Device, Observations, DiagnosticReport
4. **Stores** in InterSystems IRIS → FHIR server + vector embeddings for semantic search
5. **Enables** clinicians to query: *"Show me patients exceeding weight-bearing limits"*

## Architecture

```
CSV (14 sensors @ 100Hz)
  → Python stride detection & metrics
    → Claude agent (clinical interpretation)
      → FHIR R4 resources (Observation, DiagnosticReport)
        ├─→ InterSystems IRIS FHIR Server (structured clinical data)
        └─→ IRIS Vector Search (semantic embeddings of gait reports)
              → Doctor/PT can search: "asymmetric loading pattern"
```

## Quick Start

### 1. Start InterSystems IRIS

```bash
docker-compose up -d
# Wait ~60s for IRIS to initialize
# FHIR endpoint: http://localhost:32783/fhir/r4/metadata
# Management Portal: http://localhost:32783/csp/sys/UtilHome.csp (login: _SYSTEM / SYS)
```

### 2. Install Dependencies

```bash
pip install anthropic requests sentence-transformers

# Install IRIS DB-API driver from the hackathon kit:
# (pick the wheel matching your OS from the /install folder)
pip install ./install/intersystems_irispython-5.0.1-*.whl
```

### 3. Set API Key

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 4. Run the Pipeline

```bash
cd src

# Full pipeline (analysis + FHIR + vector search)
python main.py --csv ../data/gait_data.csv

# Just analysis (no IRIS needed)
python main.py --csv ../data/gait_data.csv --no-fhir --no-vector

# Analysis + FHIR but skip vector search
python main.py --csv ../data/gait_data.csv --no-vector
```

### 5. Query Semantically

```bash
python vector_search.py search "patients with asymmetric gait"
python vector_search.py search "high loading rate exceeding weight bearing limits"
python vector_search.py search "good recovery near discharge"
python vector_search.py list
```

## What Gets Extracted from the Data

From your `gait_data.csv` (30 seconds, ~3000 samples):

| Metric | Left Foot | Right Foot | Clinical Meaning |
|--------|-----------|------------|------------------|
| Strides detected | 29 | 29 | Equal count = consistent walking |
| Avg peak force | **361 N** | **261 N** | Left bears 38% more load |
| Avg stance phase | 79.7% | 74.6% | Left spends longer on ground |
| Stride duration | 1.00s | 1.00s | Timing is symmetric |
| Stride variability (CV) | 0.008 | 0.009 | Very consistent pattern |
| Loading rate | 87.5 N/s | 70.3 N/s | Left impacts harder |
| **Force symmetry index** | **32.1%** | | **>15% = clinically significant** |
| Cadence | 116 steps/min | | Normal range (80-130) |

**Key finding:** The patient has an antalgic gait — they maintain normal stride timing but consistently offload the right foot. If this patient has a right-leg fracture with a 20% body-weight restriction (~160N), they're exceeding it at 261N.

## FHIR Resources Created

Each analysis session generates a full FHIR R4 resource set:

| Resource | Purpose | Count |
|----------|---------|-------|
| **Patient** | Demographics, body weight | 1 |
| **Device** | Smart insole specs (14 sensors, 100Hz) | 1 |
| **Observation** | Individual metrics (cadence, symmetry, forces, stance %, loading rates, variability) | 14 |
| **DiagnosticReport** | Bundles all observations + Claude's clinical narrative | 1 |

Sample Observation (force symmetry index):
```json
{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://smartinsole.example.com/gait-metrics",
      "code": "force-symmetry-index",
      "display": "Force symmetry index"
    }]
  },
  "subject": { "reference": "Patient/abc123" },
  "device": { "reference": "Device/insole-001" },
  "valueQuantity": {
    "value": 32.1,
    "unit": "%",
    "system": "http://unitsofmeasure.org",
    "code": "%"
  },
  "interpretation": [{
    "coding": [{
      "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
      "code": "HH"
    }]
  }],
  "note": [{ "text": "0% = perfect symmetry. >15% clinically significant." }]
}
```

## Vector Search: How It Works

The clinical narrative from Claude's interpretation gets embedded as a vector (384 dimensions using `all-MiniLM-L6-v2`) and stored in IRIS alongside the FHIR patient reference. This enables semantic queries:

```python
from vector_search import GaitVectorStore

store = GaitVectorStore()

# Natural language search across all patient gait analyses
results = store.semantic_search("lateral loading bias on left foot")
results = store.semantic_search("patient exceeding weight bearing restrictions")
results = store.semantic_search("early recovery stable gait")

for r in results:
    print(f"Patient {r['patient_id']} [{r['similarity']:.3f}]")
    print(f"  {r['clinical_summary']}")
    print(f"  FHIR: {r['fhir_patient_ref']}")
```

Each result includes the FHIR patient reference, so the clinician can click through to the full structured record.

## Project Structure

```
smart-insole-fhir/
├── docker-compose.yml          # InterSystems IRIS container
├── requirements.txt
├── data/
│   ├── gait_data.csv                  # Raw sensor data (your upload)
│   ├── gait_analysis_results.json     # Extracted metrics + interpretation
│   └── sample_fhir_bundle.json        # Example FHIR output
└── src/
    ├── main.py                # Pipeline orchestrator
    ├── gait_analysis.py       # Stride detection + Claude interpretation
    ├── fhir_builder.py        # FHIR R4 resource construction + POST
    └── vector_search.py       # IRIS vector search integration
```

## Files Explained

### `gait_analysis.py`
- Loads CSV, sums sensor groups (heel, midfoot, toe) per foot
- Detects strides via heel-strike peak finding (local maxima > 50N, min 0.4s apart)
- Computes per-stride metrics: duration, stance/swing, peak forces, loading rate
- Aggregates into symmetry indices and variability measures
- Sends structured metrics to Claude with a clinical biomechanics system prompt
- Claude returns JSON: summary, symmetry assessment, WB compliance, risk flags, recommendations

### `fhir_builder.py`
- Maps each metric to a FHIR Observation with proper coding (SNOMED CT, LOINC, custom)
- Creates a DiagnosticReport with Claude's narrative as the conclusion
- Posts everything to IRIS's FHIR R4 endpoint as individual resources or a transaction Bundle

### `vector_search.py`
- Creates a SQL table in IRIS with a `VECTOR(FLOAT, 384)` column
- Embeds the full clinical narrative (summary + flags + recommendations)
- Stores alongside structured metadata (patient ID, FHIR refs, symmetry status)
- `semantic_search()` uses `VECTOR_COSINE()` similarity for natural-language queries
- Returns matches ranked by relevance with FHIR references for drill-down

## InterSystems Integration Points

| Technology | How We Use It | Why It Matters |
|------------|---------------|----------------|
| **FHIR R4 Server** | Store Patient, Device, Observation, DiagnosticReport | Doctors see standardized records in any EHR |
| **Vector Search** | Embed gait narratives for semantic querying | "Find similar patients" without writing SQL |
| **IRIS SQL** | Structured metadata alongside vectors | Filter by priority, recovery stage, date range |
| **Transaction Bundles** | Atomic multi-resource POST | All-or-nothing consistency for patient data |

## Hackathon Demo Tips

1. **Show the asymmetry detection**: The 32.1% force symmetry index is a compelling number — visualize left vs right force traces
2. **Show the FHIR resources**: Open `http://localhost:32783/fhir/r4/Patient` in browser to show the data is real and queryable
3. **Demo semantic search**: Run 3-4 different natural language queries to show vector search finding relevant patients
4. **Tell the clinical story**: "This patient was told 'toe-touch only' but our insole caught them loading 260N — that's 60% over their limit"
5. **Five use cases**: Connect each to specific metrics (post-trauma = force limits, remote gait lab = symmetry, pediatric = gamified alerts)

## Extending for the Demo

### Real-time feedback (if time permits)
The active feedback loop could be a simple threshold alert:
```python
# In a real-time loop reading from the insole:
if right_foot_force > wb_limit_N:
    trigger_vibration()  # Haptic feedback to patient
    log_violation(force=right_foot_force, limit=wb_limit_N)
```

### Multiple patients
Generate varied data (different asymmetry levels, cadences) and push multiple patients to show vector search ranking them by similarity.

### Dashboard
Query the FHIR server's REST API from a simple web frontend:
```
GET /fhir/r4/DiagnosticReport?patient=Patient/abc123&_sort=-date
GET /fhir/r4/Observation?patient=Patient/abc123&code=force-symmetry-index
```
