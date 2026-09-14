from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

SESSION_KEY = 9472
DRIVER_NUMBER = 4
MIN_GAP_S = 2.0


laps = pd.read_csv(PROJECT_DIR / "data" / "processed" / "bahrain_2024" / "pace_laps.csv")

numeric_columns = [
    "session_key",
    "driver_number",
    "lap_number",
    "stint_number",
    "tyre_age",
    "lap_duration",
    "interval_min",
]

for column in numeric_columns:
    laps[column] = pd.to_numeric(laps[column], errors="coerce")

valid_pace = (
    laps["is_valid_for_pace"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

selected = laps.loc[
    valid_pace
    & laps["session_key"].eq(SESSION_KEY)
    & laps["driver_number"].eq(DRIVER_NUMBER)
    & laps["interval_min"].ge(MIN_GAP_S)
    ].copy()

selected = selected.dropna(subset=numeric_columns)
selected = selected.sort_values("lap_number")

print(f"Selected laps: {len(selected)}")
print(
    "fuel_gain,compound,n,base_time,degradation,"
    "fit_mae,stint_2_bias,stint_3_bias"
)

for fuel_gain in [0.00, 0.03, 0.05, 0.07, 0.09]:
    for compound in ["SOFT", "HARD"]:
        sample = selected.loc[
            selected["compound"].eq(compound)
        ].copy()

        if len(sample) < 3 or sample["tyre_age"].nunique() < 2:
            raise ValueError(f"Insufficient data for {compound}")

        age = sample["tyre_age"].to_numpy(dtype=float)
        race_lap = sample["lap_number"].to_numpy(dtype=float)
        observed = sample["lap_duration"].to_numpy(dtype=float)

        adjusted = observed + fuel_gain * (race_lap - 1)

        design = np.column_stack([np.ones(len(sample)), age])
        coefficients, _, _, _ = np.linalg.lstsq(
            design, adjusted, rcond=None
        )
        base_time, degradation = coefficients

        prediction = (
                base_time
                + degradation * age
                - fuel_gain * (race_lap - 1)
        )

        sample["residual_s"] = observed - prediction
        mae = np.mean(np.abs(sample["residual_s"]))

        biases = sample.groupby("stint_number")["residual_s"].mean()
        stint_2_bias = biases.get(2, float("nan"))
        stint_3_bias = biases.get(3, float("nan"))

        print(
            f"{fuel_gain:.3f},{compound},{len(sample)},"
            f"{base_time:.4f},{degradation:.4f},{mae:.4f},"
            f"{stint_2_bias:+.4f},{stint_3_bias:+.4f}"
        )

print("\nShared-model chronological holdout:")
print("Assumed race-lap gain: 0.070 s/lap")

race_lap_gain = 0.07

train_parts = []
test_parts = []

for stint_number, stint in selected.groupby("stint_number"):
    stint = stint.sort_values("lap_number")
    split = int(len(stint) * 0.7)

    train_parts.append(stint.iloc[:split].copy())
    test_parts.append(stint.iloc[split:].copy())

train = pd.concat(train_parts)
test = pd.concat(test_parts)

for compound in ["SOFT", "HARD"]:
    compound_train = train.loc[
        train["compound"].eq(compound)
    ]

    age = compound_train["tyre_age"].to_numpy(dtype=float)
    race_lap = compound_train["lap_number"].to_numpy(dtype=float)
    observed = compound_train["lap_duration"].to_numpy(dtype=float)

    adjusted = observed + race_lap_gain * (race_lap - 1)
    design = np.column_stack([np.ones(len(age)), age])

    coefficients, _, _, _ = np.linalg.lstsq(
        design, adjusted, rcond=None
    )
    base_time, degradation = coefficients

    compound_test = test.loc[
        test["compound"].eq(compound)
    ]

    for stint_number, stint_test in compound_test.groupby("stint_number"):
        prediction = (
                base_time
                + degradation * stint_test["tyre_age"]
                - race_lap_gain * (stint_test["lap_number"] - 1)
        )

        # Baseline: mean observed time from this stint's training laps.
        stint_train = compound_train.loc[
            compound_train["stint_number"].eq(stint_number)
        ]
        baseline = stint_train["lap_duration"].mean()

        model_mae = (
                stint_test["lap_duration"] - prediction
        ).abs().mean()

        baseline_mae = (
                stint_test["lap_duration"] - baseline
        ).abs().mean()

        print(
            f"Stint {int(stint_number)} ({compound}): "
            f"train={len(stint_train)}, test={len(stint_test)}, "
            f"shared model MAE={model_mae:.3f} s, "
            f"baseline MAE={baseline_mae:.3f} s"
        )