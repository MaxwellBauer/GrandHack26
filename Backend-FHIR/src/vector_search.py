"""
InterSystems IRIS Vector Search Integration
=============================================
Embeds gait analysis reports into IRIS vector search for semantic querying.

This uses the InterSystems IRIS DB-API driver to:
1. Create a table with a VECTOR column for embeddings
2. Store gait analysis narratives with their embeddings
3. Query semantically ("find patients with lateral loading bias")

Prerequisites:
- InterSystems IRIS container running (from the hackathon Docker setup)
- intersystems_irispython driver installed
- An embedding model (uses sentence-transformers locally or OpenAI API)
"""

import json
import os
from typing import Optional

# ── Embedding Generation ──────────────────────────────────────────
# Option A: Local sentence-transformers (no API key needed, good for hackathon)
# Option B: OpenAI embeddings (if you have a key)

EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2 (local) or adjust for your model


def get_embeddings_local(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def get_embeddings_openai(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using OpenAI API."""
    import openai
    client = openai.OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings using available method."""
    try:
        return get_embeddings_local(texts)
    except ImportError:
        if os.environ.get("OPENAI_API_KEY"):
            return get_embeddings_openai(texts)
        raise RuntimeError(
            "No embedding method available. Install sentence-transformers "
            "or set OPENAI_API_KEY environment variable."
        )


# ── IRIS Vector Search ────────────────────────────────────────────

class GaitVectorStore:
    """Manages gait analysis embeddings in InterSystems IRIS."""

    def __init__(self, host="localhost", port=32782,
                 namespace="DEMO", username="_SYSTEM", password="ISCDEMO"):
        """Connect to IRIS using the DB-API driver."""
        import iris
        self.conn = iris.connect(host, port, namespace, username, password)
        self.cursor = self.conn.cursor()
        self._ensure_tables()

    def _ensure_tables(self):
        """Create the vector search table if it doesn't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS SQLUser.GaitAnalysis (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(100),
                fhir_patient_ref VARCHAR(200),
                fhir_report_ref VARCHAR(200),
                recording_date TIMESTAMP,
                clinical_summary CLOB,
                full_narrative CLOB,
                symmetry_status VARCHAR(50),
                follow_up_priority VARCHAR(20),
                recovery_stage VARCHAR(30),
                cadence FLOAT,
                force_symmetry_pct FLOAT,
                max_force_left FLOAT,
                max_force_right FLOAT,
                narrative_vector VECTOR(DOUBLE, %d)
            )
        """ % EMBEDDING_DIM)

        # Create HNSW vector index for fast similarity search
        try:
            self.cursor.execute("""
                CREATE INDEX gait_vector_idx
                ON SQLUser.GaitAnalysis (narrative_vector)
                AS 'VECTOR'
            """)
        except Exception:
            pass  # Index may already exist

        self.conn.commit()

    def store_analysis(self, analysis_results: dict, fhir_refs: dict):
        """
        Store a gait analysis with its vector embedding.
        
        Args:
            analysis_results: Output from gait_analysis.analyze_gait_data()
            fhir_refs: Output from fhir_builder.push_gait_analysis_to_fhir()
        """
        metrics = analysis_results["raw_metrics"]
        interp = analysis_results.get("clinical_interpretation", {})

        # Build rich narrative for embedding
        narrative_parts = [
            interp.get("clinical_summary", ""),
            f"Symmetry: {interp.get('symmetry_assessment', {}).get('overall', 'unknown')}.",
            interp.get("symmetry_assessment", {}).get("clinical_significance", ""),
            f"Weight bearing: {interp.get('weight_bearing_assessment', {}).get('compliance_level', 'unknown')}.",
            interp.get("weight_bearing_assessment", {}).get("notes", ""),
            f"Gait quality: cadence {interp.get('gait_quality', {}).get('cadence_assessment', 'unknown')}, "
            f"variability {interp.get('gait_quality', {}).get('stride_variability', 'unknown')}.",
            interp.get("gait_quality", {}).get("notes", ""),
        ]

        # Add risk flags
        for flag in interp.get("risk_flags", []):
            narrative_parts.append(f"Risk: {flag}.")

        # Add recommendations
        for rec in interp.get("recommendations", []):
            narrative_parts.append(f"Recommendation: {rec}.")

        full_narrative = " ".join(p for p in narrative_parts if p)
        clinical_summary = interp.get("clinical_summary", "No interpretation available")

        # Generate embedding
        embedding = get_embeddings([full_narrative])[0]

        # Format vector for IRIS SQL
        vector_str = ",".join(str(v) for v in embedding)

        self.cursor.execute("""
            INSERT INTO SQLUser.GaitAnalysis (
                patient_id, fhir_patient_ref, fhir_report_ref,
                recording_date, clinical_summary, full_narrative,
                symmetry_status, follow_up_priority, recovery_stage,
                cadence, force_symmetry_pct, max_force_left, max_force_right,
                narrative_vector
            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      TO_VECTOR(?,double))
        """, [
            fhir_refs.get("patient_id", "unknown"),
            fhir_refs.get("patient_ref", ""),
            fhir_refs.get("report_ref", ""),
            clinical_summary,
            full_narrative,
            interp.get("symmetry_assessment", {}).get("overall", "unknown"),
            interp.get("follow_up_priority", "routine"),
            interp.get("recovery_stage_estimate", "unknown"),
            metrics.get("cadence_steps_per_min", 0),
            metrics.get("symmetry_index_force", 0),
            metrics.get("max_force_left_N", 0),
            metrics.get("max_force_right_N", 0),
            vector_str,
        ])

        self.conn.commit()
        print(f"  ✓ Stored analysis for patient {fhir_refs.get('patient_id')}")

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search gait analyses by natural language query.
        
        Examples:
            "patients with asymmetric gait"
            "high loading rate on right foot"
            "early recovery stage needing urgent follow-up"
            "good symmetry, normal cadence"
        """
        query_embedding = get_embeddings([query])[0]
        vector_str = ",".join(str(v) for v in query_embedding)

        self.cursor.execute("""
            SELECT TOP %d
                patient_id,
                fhir_patient_ref,
                fhir_report_ref,
                clinical_summary,
                symmetry_status,
                follow_up_priority,
                recovery_stage,
                cadence,
                force_symmetry_pct,
                VECTOR_COSINE(narrative_vector, TO_VECTOR(?,double)) AS similarity
            FROM SQLUser.GaitAnalysis
            ORDER BY similarity DESC
        """ % top_k, [vector_str])

        results = []
        for row in self.cursor.fetchall():
            results.append({
                "patient_id": row[0],
                "fhir_patient_ref": row[1],
                "fhir_report_ref": row[2],
                "clinical_summary": row[3],
                "symmetry_status": row[4],
                "follow_up_priority": row[5],
                "recovery_stage": row[6],
                "cadence": row[7],
                "force_symmetry_pct": row[8],
                "similarity": round(row[9], 4) if row[9] else 0,
            })

        return results

    def get_all_analyses(self) -> list[dict]:
        """Retrieve all stored analyses (for dashboard display)."""
        self.cursor.execute("""
            SELECT patient_id, fhir_patient_ref, clinical_summary,
                   symmetry_status, follow_up_priority, recovery_stage,
                   cadence, force_symmetry_pct, recording_date
            FROM SQLUser.GaitAnalysis
            ORDER BY recording_date DESC
        """)

        return [{
            "patient_id": row[0],
            "fhir_patient_ref": row[1],
            "clinical_summary": row[2],
            "symmetry_status": row[3],
            "follow_up_priority": row[4],
            "recovery_stage": row[5],
            "cadence": row[6],
            "force_symmetry_pct": row[7],
            "recording_date": str(row[8]) if row[8] else None,
        } for row in self.cursor.fetchall()]

    def close(self):
        self.cursor.close()
        self.conn.close()


# ── CLI Demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python vector_search.py store <analysis_results.json> <fhir_refs.json>")
        print("  python vector_search.py search \"patients with asymmetric gait\"")
        print("  python vector_search.py list")
        sys.exit(1)

    cmd = sys.argv[1]

    store = GaitVectorStore(
        host=os.environ.get("IRIS_HOST", "localhost"),
        port=int(os.environ.get("IRIS_PORT", "1972")),
        namespace=os.environ.get("IRIS_NAMESPACE", "USER"),
        username=os.environ.get("IRIS_USER", "demo"),
        password=os.environ.get("IRIS_PASSWORD", "demo"),
    )

    try:
        if cmd == "store":
            with open(sys.argv[2]) as f:
                results = json.load(f)
            with open(sys.argv[3]) as f:
                refs = json.load(f)
            store.store_analysis(results, refs)

        elif cmd == "search":
            query = sys.argv[2]
            print(f"\nSearching: \"{query}\"\n")
            matches = store.semantic_search(query)
            for i, m in enumerate(matches, 1):
                print(f"{i}. [{m['similarity']:.3f}] Patient {m['patient_id']}")
                print(f"   {m['clinical_summary'][:120]}...")
                print(f"   Priority: {m['follow_up_priority']}, "
                      f"Symmetry: {m['symmetry_status']}, "
                      f"Cadence: {m['cadence']}")
                print()

        elif cmd == "list":
            all_analyses = store.get_all_analyses()
            print(f"\n{len(all_analyses)} analyses stored:\n")
            for a in all_analyses:
                print(f"  Patient {a['patient_id']}: {a['clinical_summary'][:80]}...")
    finally:
        store.close()
