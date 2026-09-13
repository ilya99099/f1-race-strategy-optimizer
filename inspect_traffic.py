from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent
laps = pd.read_csv(project_dir / "pace_laps.csv")

valid = laps[
    laps["is_valid_for_pace"].astype(str).str.lower().eq("true")
].copy()

interval = pd.to_numeric(valid["interval_min"], errors="coerce")

print("Valid pace laps:", len(valid))
print("Laps without numeric interval_min:", interval.isna().sum())

for threshold in [1, 2, 3, 4, 5]:
    count = (interval >= threshold).sum()
    print(f"Laps with minimum observed gap >= {threshold}s: {count}")

driver_number = pd.to_numeric(valid["driver_number"], errors="coerce")

for threshold in [2, 3, 4]:
    selected = valid[
        driver_number.eq(4) & interval.ge(threshold)
        ]

    print(f"\nNorris: minimum observed gap >= {threshold}s")
    print(f"Total laps: {len(selected)}")

    print(
        selected.groupby(["session_key", "stint_number", "compound"])
        .size()
        .rename("lap_count")
        .to_string()
    )

selected = valid[
    driver_number.eq(4) & interval.ge(2)
    ].copy()

columns = [
    "stint_number",
    "compound",
    "lap_number",
    "tyre_age",
    "lap_duration",
    "interval_min",
    "interval_sample_count",
]

print("\nNorris: selected laps at the 2s threshold")
print(
    selected.sort_values(["session_key", "stint_number", "lap_number"])
    [columns]
    .to_string(index=False)
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))

for (stint, compound), group in selected.groupby(
        ["stint_number", "compound"]
):
    group = group.sort_values("tyre_age")

    ax.scatter(
        group["tyre_age"],
        group["lap_duration"],
        label=f"Stint {stint} — {compound}",
    )

ax.set_xlabel("Tyre age (laps)")
ax.set_ylabel("Lap time (s)")
ax.set_title("Norris: lap times with minimum observed gap >= 2 s")
ax.legend()
ax.grid(alpha=0.3)

output_dir = project_dir / "results"
output_dir.mkdir(exist_ok=True)

fig.tight_layout()
fig.savefig(output_dir / "norris_filtered_pace.png", dpi=160)
plt.close(fig)

print("\nSaved: results/norris_filtered_pace.png")

import numpy as np

print("\nLinear pace trend by stint:")

for (session, stint, compound), group in selected.groupby(
        ["session_key", "stint_number", "compound"]
):
    x = group["tyre_age"].to_numpy(dtype=float)
    y = group["lap_duration"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = intercept + slope * x
    rmse = np.sqrt(np.mean((y - predicted) ** 2))

    print(
        f"Stint {stint} ({compound}): "
        f"n={len(group)}, "
        f"slope={slope:+.4f} s/lap, "
        f"RMSE={rmse:.3f} s"
    )

print("\nChronological holdout validation:")

for (session, stint, compound), group in selected.groupby(
        ["session_key", "stint_number", "compound"]
):
    group = group.sort_values("lap_number")
    split = int(len(group) * 2 / 3)

    train = group.iloc[:split]
    test = group.iloc[split:]

    x_train = train["tyre_age"].to_numpy(dtype=float)
    y_train = train["lap_duration"].to_numpy(dtype=float)
    x_test = test["tyre_age"].to_numpy(dtype=float)
    y_test = test["lap_duration"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x_train, y_train, deg=1)

    linear_prediction = intercept + slope * x_test
    baseline_prediction = y_train.mean()

    linear_mae = np.mean(np.abs(y_test - linear_prediction))
    baseline_mae = np.mean(np.abs(y_test - baseline_prediction))

    print(
        f"Stint {stint} ({compound}): "
        f"train={len(train)}, test={len(test)}, "
        f"linear MAE={linear_mae:.3f} s, "
        f"baseline MAE={baseline_mae:.3f} s"
    )

print("\nExpanding-window validation:")

for (session, stint, compound), group in selected.groupby(
        ["session_key", "stint_number", "compound"]
):
    group = group.sort_values("lap_number")
    x = group["tyre_age"].to_numpy(dtype=float)
    y = group["lap_duration"].to_numpy(dtype=float)

    linear_errors = []
    baseline_errors = []

    for i in range(4, len(group)):
        slope, intercept = np.polyfit(x[:i], y[:i], deg=1)

        linear_prediction = intercept + slope * x[i]
        baseline_prediction = y[:i].mean()

        linear_errors.append(abs(y[i] - linear_prediction))
        baseline_errors.append(abs(y[i] - baseline_prediction))

    print(
        f"Stint {stint} ({compound}): "
        f"predictions={len(linear_errors)}, "
        f"linear MAE={np.mean(linear_errors):.3f} s, "
        f"baseline MAE={np.mean(baseline_errors):.3f} s"
    )

output_path = project_dir / "data" / "processed" / "norris_pace_laps.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

selected.sort_values(
    ["session_key", "stint_number", "lap_number"]
).to_csv(output_path, index=False)

print(f"\nSaved {len(selected)} laps to {output_path}")