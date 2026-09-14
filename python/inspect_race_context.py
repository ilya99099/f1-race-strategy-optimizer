"""Inspect race context around laps flagged by the pace model."""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from prepare_laps import load_json_array, parse_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bahrain_2024"
LAPS_PATH = PROJECT_ROOT / "data" / "processed" / "bahrain_2024" / "laps_enriched.csv"

TARGET_DRIVER = 4
FLAGGED_LAPS = {15, 26, 35, 36, 57}


@dataclass(frozen=True)
class LapWindow:
    lap_number: int
    start: datetime
    end: datetime


DatedRow = tuple[datetime, dict[str, Any]]


def load_flagged_lap_windows(
        path: Path,
        target_driver: int,
        flagged_laps: set[int],
) -> dict[int, LapWindow]:
    if not path.exists():
        raise FileNotFoundError(f"Enriched lap file not found: {path}")

    windows: dict[int, LapWindow] = {}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "driver_number",
            "lap_number",
            "date_start",
            "lap_duration",
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
            lap_number = int(row["lap_number"])

            if driver_number != target_driver:
                continue

            if lap_number not in flagged_laps:
                continue

            start = parse_timestamp(
                row["date_start"],
                "lap.date_start",
            )
            duration = float(row["lap_duration"])

            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"Invalid duration for lap {lap_number}: {duration}"
                )

            windows[lap_number] = LapWindow(
                lap_number=lap_number,
                start=start,
                end=start + timedelta(seconds=duration),
            )

    missing_laps = flagged_laps - set(windows)

    if missing_laps:
        raise ValueError(
            f"Flagged laps not found in CSV: {sorted(missing_laps)}"
        )

    return dict(sorted(windows.items()))


def prepare_dated_rows(
        rows: Iterable[dict[str, Any]],
) -> list[DatedRow]:
    dated_rows: list[DatedRow] = []

    for row in rows:
        event_time = parse_timestamp(
            row.get("date"),
            "event.date",
        )
        dated_rows.append((event_time, row))

    dated_rows.sort(key=lambda item: item[0])
    return dated_rows

def lower_bound_by_time(
        rows: list[DatedRow],
        moment: datetime,
) -> int:
    left = 0
    right = len(rows)

    while left < right:
        middle = (left + right) // 2

        if rows[middle][0] < moment:
            left = middle + 1
        else:
            right = middle

    return left


def upper_bound_by_time(
        rows: list[DatedRow],
        moment: datetime,
) -> int:
    left = 0
    right = len(rows)

    while left < right:
        middle = (left + right) // 2

        if rows[middle][0] <= moment:
            left = middle + 1
        else:
            right = middle

    return left


def rows_during_window(
        rows: list[DatedRow],
        window: LapWindow,
) -> list[dict[str, Any]]:
    left = lower_bound_by_time(
        rows,
        window.start,
    )

    right = lower_bound_by_time(
        rows,
        window.end,
    )

    return [
        rows[index][1]
        for index in range(left, right)
    ]


def latest_row_at_or_before(
        rows: list[DatedRow],
        moment: datetime,
) -> dict[str, Any] | None:
    index = upper_bound_by_time(
        rows,
        moment,
    ) - 1

    if index < 0:
        return None

    return rows[index][1]

def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            return None
    else:
        return None

    if not math.isfinite(result):
        return None

    return result


def numeric_values(
        rows: Iterable[dict[str, Any]],
        field_name: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        value = numeric_value(row.get(field_name))

        if value is not None:
            values.append(value)

    return values


def format_summary(values: list[float]) -> str:
    if not values:
        return "no numeric samples"

    return (
        f"min={min(values):.3f}, "
        f"median={statistics.median(values):.3f}, "
        f"max={max(values):.3f}, "
        f"samples={len(values)}"
    )


def format_position(row: dict[str, Any] | None) -> str:
    if row is None:
        return "unknown"

    position = row.get("position")

    if not isinstance(position, int) or isinstance(position, bool):
        return "unknown"

    return str(position)


def build_driver_names(
        rows: Iterable[dict[str, Any]],
) -> dict[int, str]:
    names: dict[int, str] = {}

    for row in rows:
        driver_number = row.get("driver_number")

        if not isinstance(driver_number, int) or isinstance(
                driver_number,
                bool,
        ):
            continue

        name = row.get("name_acronym")

        if not isinstance(name, str) or not name:
            name = str(driver_number)

        names[driver_number] = name

    return names


def driver_label(
        driver_number: Any,
        driver_names: dict[int, str],
) -> str:
    if not isinstance(driver_number, int) or isinstance(
            driver_number,
            bool,
    ):
        return "unknown driver"

    name = driver_names.get(driver_number, str(driver_number))
    return f"{name} (#{driver_number})"


def print_lap_context(
        window: LapWindow,
        position_rows: list[DatedRow],
        interval_rows: list[DatedRow],
        overtake_rows: list[DatedRow],
        race_control_rows: list[DatedRow],
        driver_names: dict[int, str],
) -> None:
    position_at_start = latest_row_at_or_before(
        position_rows,
        window.start,
    )
    position_at_end = latest_row_at_or_before(
        position_rows,
        window.end,
    )

    position_changes = rows_during_window(
        position_rows,
        window,
    )
    interval_samples = rows_during_window(
        interval_rows,
        window,
    )

    related_overtakes = [
        row
        for row in rows_during_window(overtake_rows, window)
        if TARGET_DRIVER
           in {
               row.get("overtaking_driver_number"),
               row.get("overtaken_driver_number"),
           }
    ]

    race_control_events = rows_during_window(
        race_control_rows,
        window,
    )

    intervals = numeric_values(
        interval_samples,
        "interval",
    )
    gaps_to_leader = numeric_values(
        interval_samples,
        "gap_to_leader",
    )

    print()
    print(
        f"Lap {window.lap_number} | "
        f"{window.start.isoformat()} -> {window.end.isoformat()}"
    )
    print(
        f"Position: "
        f"{format_position(position_at_start)} -> "
        f"{format_position(position_at_end)}"
    )
    print(f"Interval to car ahead: {format_summary(intervals)}")
    print(f"Gap to leader:         {format_summary(gaps_to_leader)}")

    if position_changes:
        print("Position updates during lap:")

        for row in position_changes:
            print(f"  position={row.get('position')}")
    else:
        print("Position updates during lap: none")

    if related_overtakes:
        print("Related position changes:")

        for row in related_overtakes:
            overtaking_driver = driver_label(
                row.get("overtaking_driver_number"),
                driver_names,
            )
            overtaken_driver = driver_label(
                row.get("overtaken_driver_number"),
                driver_names,
            )

            print(
                f"  {overtaking_driver} passed "
                f"{overtaken_driver}; "
                f"position={row.get('position')}"
            )
    else:
        print("Related position changes: none")

    if race_control_events:
        print("Race-control events:")

        for row in race_control_events:
            print(
                f"  {row.get('category')}: "
                f"{row.get('message')}"
            )
    else:
        print("Race-control events: none")


def main() -> None:
    windows = load_flagged_lap_windows(
        LAPS_PATH,
        TARGET_DRIVER,
        FLAGGED_LAPS,
    )

    drivers = load_json_array(RAW_DIR / "drivers.json")
    positions = load_json_array(RAW_DIR / "position.json")
    intervals = load_json_array(RAW_DIR / "intervals.json")
    overtakes = load_json_array(RAW_DIR / "overtakes.json")
    race_control = load_json_array(
        RAW_DIR / "race_control.json"
    )

    driver_names = build_driver_names(drivers)

    target_positions = prepare_dated_rows(
        row
        for row in positions
        if row.get("driver_number") == TARGET_DRIVER
    )
    target_intervals = prepare_dated_rows(
        row
        for row in intervals
        if row.get("driver_number") == TARGET_DRIVER
    )

    dated_overtakes = prepare_dated_rows(overtakes)
    dated_race_control = prepare_dated_rows(race_control)

    print(
        f"Race context for "
        f"{driver_label(TARGET_DRIVER, driver_names)}"
    )

    for window in windows.values():
        print_lap_context(
            window,
            target_positions,
            target_intervals,
            dated_overtakes,
            dated_race_control,
            driver_names,
        )


if __name__ == "__main__":
    main()