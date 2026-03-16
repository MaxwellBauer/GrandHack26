"""
run_demo.py
Entry point for the GrandHack26 shoe pressure sensor demo.
"""

from pathlib import Path

DATA_PATH   = 'data/gait_data.csv'
HEATMAP_PNG = 'data/heatmap.png'
GAIT_PNG    = 'data/gait_analysis.png'


def main():
    print("=" * 55)
    print("  GrandHack26 — Shoe Pressure Sensor Demo")
    print("=" * 55)

    # Step 1: Generate sensor data
    if not Path(DATA_PATH).exists():
        print("\n[1/3] Generating synthetic sensor data...")
        import generate_data
        generate_data.run(output_path=DATA_PATH)
    else:
        print(f"\n[1/3] Data already exists → {DATA_PATH}  (skipping)")

    # Step 2: Footprint heat map
    print("\n[2/3] Rendering plantar pressure heat map...")
    import heatmap_viz
    heatmap_viz.render(data_path=DATA_PATH, output_png=HEATMAP_PNG)

    # Step 3: Gait analysis dashboard
    print("\n[3/3] Rendering gait analysis dashboard...")
    import gait_viz
    gait_viz.render(data_path=DATA_PATH, output_png=GAIT_PNG)

    print("\n" + "=" * 55)
    print("  Done! Outputs saved to:")
    print(f"    {DATA_PATH}")
    print(f"    {HEATMAP_PNG}")
    print(f"    {GAIT_PNG}")
    print("=" * 55)


if __name__ == '__main__':
    main()
