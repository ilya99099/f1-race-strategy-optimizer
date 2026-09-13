from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INPUT_PATH = Path("laps_enriched.csv")
OUTPUT_PATH = Path("results/norris_stints.png")
TARGET_DRIVER = 4
OUTLIER_RESIDUAL_THRESHOLD = 0.25


@dataclass(frozen=True)
class LapPoint:
    lap_number: int
    stint_number: int
    compound: str
    tyre_age: int
    lap_duration: float
    sector_1: float
    sector_2: float
    sector_3: float


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    r_squared: float


def load_valid_laps(
        path: Path,
        target_driver: int,
) -> list[LapPoint]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    laps: list[LapPoint] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "driver_number",
            "lap_number",
            "stint_number",
            "compound",
            "tyre_age",
            "lap_duration",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "is_valid_for_pace",
        }

        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV file is missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            driver_number = int(row["driver_number"])

            if driver_number != target_driver:
                continue

            if row["is_valid_for_pace"].strip().lower() != "true":
                continue

            laps.append(
                LapPoint(
                    lap_number=int(row["lap_number"]),
                    stint_number=int(row["stint_number"]),
                    compound=row["compound"],
                    tyre_age=int(row["tyre_age"]),
                    lap_duration=float(row["lap_duration"]),
                    sector_1=float(row["duration_sector_1"]),
                    sector_2=float(row["duration_sector_2"]),
                    sector_3=float(row["duration_sector_3"]),
                )
            )

    if not laps:
        raise ValueError(
            f"No valid laps found for driver {target_driver}"
        )

    return laps


def group_laps_by_stint(
        laps: list[LapPoint],
) -> dict[int, list[LapPoint]]:
    groups: dict[int, list[LapPoint]] = defaultdict(list)

    for lap in laps:
        groups[lap.stint_number].append(lap)

    for stint_laps in groups.values():
        stint_laps.sort(key=lambda lap: lap.lap_number)

    return dict(sorted(groups.items()))


def fit_linear_regression(
        laps: list[LapPoint],
        value_getter: Callable[[LapPoint], float],
) -> RegressionResult:
    if len(laps) < 2:
        raise ValueError("At least two laps are required for regression")

    x_values = [lap.tyre_age for lap in laps]
    y_values = [value_getter(lap) for lap in laps]

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)

    denominator = sum(
        (x - mean_x) ** 2
        for x in x_values
    )

    if denominator == 0:
        raise ValueError("All laps have the same tyre age")

    slope = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values)
    ) / denominator

    intercept = mean_y - slope * mean_x

    predicted_values = [
        intercept + slope * x
        for x in x_values
    ]

    residual_sum_of_squares = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(
            y_values,
            predicted_values,
        )
    )

    total_sum_of_squares = sum(
        (actual - mean_y) ** 2
        for actual in y_values
    )

    if total_sum_of_squares == 0:
        r_squared = 1.0
    else:
        r_squared = (
                1.0
                - residual_sum_of_squares
                / total_sum_of_squares
        )

    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
    )

def print_stint_summary(
        stint_number: int,
        laps: list[LapPoint],
        regression: RegressionResult,
) -> None:
    compound = laps[0].compound

    print()
    print(
        f"Stint {stint_number} | "
        f"{compound} | "
        f"{len(laps)} valid laps"
    )
    print(
        f"Slope: {regression.slope:+.4f} seconds per tyre lap"
    )
    print(f"R²: {regression.r_squared:.4f}")
    print()
    print(
        f"{'Lap':>4} "
        f"{'Age':>4} "
        f"{'Time':>8} "
        f"{'Predicted':>10} "
        f"{'Residual':>10}"
    )

    for lap in laps:
        predicted = (
                regression.intercept
                + regression.slope * lap.tyre_age
        )
        residual = lap.lap_duration - predicted

        print(
            f"{lap.lap_number:>4} "
            f"{lap.tyre_age:>4} "
            f"{lap.lap_duration:>8.3f} "
            f"{predicted:>10.3f} "
            f"{residual:>+10.3f}"
        )

def print_sector_summary(
        stint_number: int,
        lap_regression: RegressionResult,
        sector_regressions: dict[str, RegressionResult],
) -> None:
    print()
    print(f"Sector trends for stint {stint_number}")

    for sector_name, regression in sector_regressions.items():
        print(
            f"{sector_name}: "
            f"slope={regression.slope:+.4f} s/lap, "
            f"R²={regression.r_squared:.4f}"
        )

    combined_sector_slope = sum(
        regression.slope
        for regression in sector_regressions.values()
    )

    difference = (
            lap_regression.slope
            - combined_sector_slope
    )

    print(
        f"Sum of sector slopes: "
        f"{combined_sector_slope:+.4f} s/lap"
    )
    print(
        f"Lap-time slope:       "
        f"{lap_regression.slope:+.4f} s/lap"
    )
    print(
        f"Difference:           "
        f"{difference:+.6f} s/lap"
    )

def calculate_residual(
        tyre_age: int,
        actual_value: float,
        regression: RegressionResult,
) -> float:
    predicted_value = (
            regression.intercept
            + regression.slope * tyre_age
    )

    return actual_value - predicted_value

def print_outlier_details(
        laps: list[LapPoint],
        lap_regression: RegressionResult,
        sector_regressions: dict[str, RegressionResult],
        threshold: float,
) -> None:
    print()
    print(
        f"Flagged laps with "
        f"|lap residual| >= {threshold:.3f} s"
    )

    found_outlier = False

    for lap in laps:
        lap_residual = calculate_residual(
            lap.tyre_age,
            lap.lap_duration,
            lap_regression,
        )

        if abs(lap_residual) < threshold:
            continue

        found_outlier = True

        sector_actual_values = {
            "Sector 1": lap.sector_1,
            "Sector 2": lap.sector_2,
            "Sector 3": lap.sector_3,
        }

        sector_residuals: dict[str, float] = {}

        for sector_name, actual_value in sector_actual_values.items():
            sector_residuals[sector_name] = calculate_residual(
                lap.tyre_age,
                actual_value,
                sector_regressions[sector_name],
            )

        combined_sector_residual = sum(
            sector_residuals.values()
        )

        print()
        print(
            f"Lap {lap.lap_number} | "
            f"tyre age {lap.tyre_age} | "
            f"lap residual {lap_residual:+.3f} s"
        )

        for sector_name, residual in sector_residuals.items():
            print(
                f"  {sector_name}: "
                f"{residual:+.3f} s"
            )

        print(
            f"  Sum of sector residuals: "
            f"{combined_sector_residual:+.3f} s"
        )

    if not found_outlier:
        print("  None")


def save_stint_plot(
        groups: dict[int, list[LapPoint]],
        regressions: dict[int, RegressionResult],
        output_path: Path,
) -> None:
    number_of_stints = len(groups)

    figure, axes = plt.subplots(
        1,
        number_of_stints,
        figsize=(5 * number_of_stints, 4),
        squeeze=False,
    )

    for column, (stint_number, laps) in enumerate(groups.items()):
        axis = axes[0][column]
        regression = regressions[stint_number]

        x_values = [lap.tyre_age for lap in laps]
        y_values = [lap.lap_duration for lap in laps]

        axis.scatter(
            x_values,
            y_values,
            color="tab:blue",
            label="Valid laps",
        )

        line_x = [
            min(x_values),
            max(x_values),
        ]
        line_y = [
            regression.intercept
            + regression.slope * tyre_age
            for tyre_age in line_x
        ]

        axis.plot(
            line_x,
            line_y,
            color="tab:red",
            label="Linear trend",
        )

        for lap in laps:
            axis.annotate(
                str(lap.lap_number),
                (lap.tyre_age, lap.lap_duration),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

        axis.set_title(
            f"Stint {stint_number}: {laps[0].compound}\n"
            f"slope={regression.slope:+.4f}, "
            f"R²={regression.r_squared:.3f}"
        )
        axis.set_xlabel("Tyre age at start of lap")
        axis.set_ylabel("Lap duration, seconds")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(
        f"Driver {TARGET_DRIVER}: valid race laps",
        fontsize=14,
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    laps = load_valid_laps(
        INPUT_PATH,
        TARGET_DRIVER,
    )

    groups = group_laps_by_stint(laps)

    regressions: dict[int, RegressionResult] = {}

    for stint_number, stint_laps in groups.items():
        regression = fit_linear_regression(
            stint_laps,
            lambda lap: lap.lap_duration,
        )
        regressions[stint_number] = regression


        print_stint_summary(
            stint_number,
            stint_laps,
            regression,
        )

        sector_regressions = {
            "Sector 1": fit_linear_regression(
                stint_laps,
                lambda lap: lap.sector_1,
            ),
            "Sector 2": fit_linear_regression(
                stint_laps,
                lambda lap: lap.sector_2,
            ),
            "Sector 3": fit_linear_regression(
                stint_laps,
                lambda lap: lap.sector_3,
            ),
        }

        print_sector_summary(
            stint_number,
            regression,
            sector_regressions,
        )

        print_outlier_details(
            stint_laps,
            regression,
            sector_regressions,
            OUTLIER_RESIDUAL_THRESHOLD,
        )



    save_stint_plot(
        groups,
        regressions,
        OUTPUT_PATH,
    )

    print()
    print(f"Plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()