# InStride — Smart Insole Gait Intelligence

**MIT GrandHack 2026** — Post-surgical rehabilitation platform using pressure-sensing insoles to quantify foot loading, detect gait anomalies, and track recovery progress.

---

## Repository Structure

```
GrandHack26/
├── PressureAnimation/      # Real-sensor insole pressure animation
├── Analysis/               # Synthetic gait data generation & visualization
└── Backend-FHIR/           # Clinical dashboard, FHIR R4, IRIS vector search
```

---

## Modules

### `PressureAnimation/`
Generates a 15-second MP4 animation of plantar pressure from a 5-sensor insole CSV. A pre-rendered demo (`output.mp4`) is included.

- **Sensors:** U1 (medial forefoot), U2 (lateral forefoot), M1 (midfoot), L1 (medial heel), L2 (lateral heel)
- **Features:** RBF-interpolated heatmap, % body weight bar with MIN/MAX tolerance bands (blue/green/red), real sensor data scaled to physiological walking loads (~70 kg)
- **Input:** `scaled_sensor_data.csv` (16× scaled, realistic gram values)

```bash
cd PressureAnimation
pip install numpy pandas matplotlib scipy
python pressure_animation.py scaled_sensor_data.csv output.mp4
```

---

### `Analysis/`
Synthetic 14-sensor gait data generation and three clinical visualizations.

- `generate_data.py` — Simulates a 30s walk with a right-foot limp at 100 Hz
- `heatmap_viz.py` — Plantar pressure heatmap (left vs right foot)
- `gait_viz.py` — Force traces, step intervals, symmetry index
- `run_demo.py` — Runs all three in sequence

```bash
cd Analysis
pip install -r requirements.txt
python run_demo.py
```

---

### `Backend-FHIR/`
Flask web dashboard integrating Claude AI, FHIR R4 (InterSystems IRIS), and semantic vector search.

- Gait analysis → Claude narrative → FHIR DiagnosticReport → IRIS vector store
- Semantic search: *"patients with asymmetric gait"*, *"urgent follow-up needed"*
- 14 synthetic demo patients pre-seeded

```bash
cd Backend-FHIR
pip install -r requirements.txt
python app.py          # Dashboard at http://localhost:5050
python seed_patients.py  # Seed IRIS with demo patients (requires Docker)
```

See `Backend-FHIR/README.md` for full setup including Docker + IRIS.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Pressure sensing | 5-sensor FSR insole (U1, U2, M1, L1, L2) |
| Gait analysis AI | Claude (Anthropic) |
| Clinical data standard | FHIR R4 |
| Database / vector search | InterSystems IRIS |
| Dashboard | Flask + vanilla JS |
| Visualization | Matplotlib, RBF interpolation |
