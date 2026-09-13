from pathlib import Path
import json

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODEL_DIR.parent

source = MODEL_DIR / "results" / "parameters.json"
destination = PROJECT_DIR / "config" / "pace_model.txt"

report = json.loads(source.read_text(encoding="utf-8"))
model = report["full_data_linear"]

keys = [
    "soft_base_s",
    "soft_degradation_s_per_lap",
    "hard_base_s",
    "hard_degradation_s_per_lap",
    "race_lap_gain_s_per_lap",
]

destination.parent.mkdir(parents=True, exist_ok=True)

lines = [f"{key} {model[key]:.12f}" for key in keys]
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"Exported: {destination}")