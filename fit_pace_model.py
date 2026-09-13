from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "data" / "processed" / "norris_pace_laps.csv"
OUTPUT_DIR = PROJECT_DIR / "results"
MIN_TRAIN_LAPS = 4


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(intercept), float(slope)


def mae(actual, predicted):
    return float(np.mean(np.abs(actual - predicted)))


def evaluate_holdout(x, y):
    split = max(MIN_TRAIN_LAPS, int(len(x) * 2 / 3))

    intercept, slope = fit_line(x[:split], y[:split])
    prediction = intercept + slope * x[split:]

    return {
        "holdout_train_laps": split,
        "holdout_test_laps": len(x) - split,
        "holdout_linear_mae_s": mae(y[split:], prediction),
        "holdout_mean_mae_s": mae(y[split:], y[:split].mean()),
    }


def evaluate_expanding_window(x, y):
    linear_errors = []
    mean_errors = []

    for i in range(MIN_TRAIN_LAPS, len(x)):
        intercept, slope = fit_line(x[:i], y[:i])

        linear_errors.append(abs(y[i] - (intercept + slope * x[i])))
        mean_errors.append(abs(y[i] - y[:i].mean()))

    return {
        "expanding_predictions": len(linear_errors),
        "expanding_linear_mae_s": float(np.mean(linear_errors)),
        "expanding_mean_mae_s": float(np.mean(mean_errors)),
    }


def main():
    laps = pd.read_csv(INPUT_PATH)

    required_columns = [
        "session_key",
        "driver_number",
        "stint_number",
        "compound",
        "lap_number",
        "tyre_age",
        "lap_duration",
    ]

    missing = set(required_columns) - set(laps.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for column in required_columns:
        if column != "compound":
            laps[column] = pd.to_numeric(laps[column], errors="raise")

    if laps[required_columns].isna().any().any():
        raise ValueError("Required columns contain missing values.")

    numeric = laps[
        [column for column in required_columns if column != "compound"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        raise ValueError("Required numeric columns contain non-finite values.")

    if laps.duplicated(
            ["session_key", "driver_number", "lap_number"]
    ).any():
        raise ValueError("Duplicate driver laps found.")

    group_columns = [
        "session_key",
        "driver_number",
        "stint_number",
        "compound",
    ]

    groups = list(laps.groupby(group_columns, sort=True))
    if not groups:
        raise ValueError("No laps available for modelling.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(9, 3.5 * len(groups)),
        squeeze=False,
    )

    results = []

    for ax, (key, group) in zip(axes[:, 0], groups):
        session, driver, stint, compound = key
        group = group.sort_values("lap_number")

        if len(group) <= MIN_TRAIN_LAPS:
            raise ValueError(
                f"Stint {stint}: at least {MIN_TRAIN_LAPS + 1} laps required."
            )

        x = group["tyre_age"].to_numpy(dtype=float)
        y = group["lap_duration"].to_numpy(dtype=float)

        if not np.all(np.diff(x) > 0):
            raise ValueError(
                f"Stint {stint}: tyre age must increase with lap number."
            )

        intercept, slope = fit_line(x, y)
        fitted = intercept + slope * x

        result = {
            "session_key": session,
            "driver_number": driver,
            "stint_number": stint,
            "compound": compound,
            "lap_count": len(group),
            "tyre_age_min": float(x.min()),
            "tyre_age_max": float(x.max()),
            "intercept_s": intercept,
            "pace_trend_s_per_lap": slope,
            "training_rmse_s": float(np.sqrt(np.mean((y - fitted) ** 2))),
            **evaluate_holdout(x, y),
            **evaluate_expanding_window(x, y),
        }
        results.append(result)

        ax.scatter(x, y, label="Observed laps")
        ax.plot(
            x,
            fitted,
            color="tab:red",
            label=f"Full-data fit: {slope:+.4f} s/lap",
        )

        ax.set_title(
            f"Session {session} | Driver {driver} | "
            f"Stint {stint} — {compound}"
        )
        ax.set_xlabel("Tyre age (laps)")
        ax.set_ylabel("Lap time (s)")
        ax.grid(alpha=0.3)
        ax.legend()

    summary = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / "norris_pace_model_summary.csv"
    plot_path = OUTPUT_DIR / "norris_pace_model_fits.png"

    summary.to_csv(summary_path, index=False)

    fig.suptitle(
        "Observed pace trends — not fuel-corrected tyre degradation",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    display_columns = [
        "stint_number",
        "compound",
        "lap_count",
        "pace_trend_s_per_lap",
        "holdout_linear_mae_s",
        "holdout_mean_mae_s",
        "expanding_linear_mae_s",
        "expanding_mean_mae_s",
    ]

    print(summary[display_columns].round(4).to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()