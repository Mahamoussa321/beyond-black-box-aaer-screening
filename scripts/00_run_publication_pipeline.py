from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "01_make_final_features.py",
    "02_train_baseline_models.py",
    "03_train_ai_teacher_guided_vc_validation_selected.py",
    "04_make_publication_tables_and_figures.py",
    "05_make_highres_publication_figures.py",
]

for script in SCRIPTS:
    path = ROOT / "scripts" / script
    print("\n" + "="*90)
    print(f"Running {path}")
    print("="*90)
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"Pipeline stopped because {script} failed with exit code {result.returncode}")

print("\nPublication pipeline complete.")
print("Key table folder:", ROOT / "outputs" / "paper_tables_figures")
print("High-resolution figure folder:", ROOT / "outputs" / "publication_figures_highres")
