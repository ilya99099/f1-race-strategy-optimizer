"""Add race-context features to every enriched lap."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from inspect_race_context import (
    DatedRow,
    LapWindow,
    latest_row_at_or_before,
    numeric_values,
    prepare_dated_rows,
    rows_during_window,
)
from prepare_laps import (
    bool_text,
    load_json_array,
    parse_timestamp,
    require_int,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bahrain_2024"

INPUT_PATH = PROJECT_ROOT / "laps_enriched.csv"
OUTPUT_PATH = PROJECT_ROOT / "pace_laps.csv"


CONTEXT_FIELDS = (
    "has_position_data",
    "position_at_start",
    "position_at_end",
    "position_update_count",
    "position_changed",
    "interval_min",
    "interval_median",
    "interval_max",
    "interval_sample_count",
    "gap_to_leader_min",
    "gap_to_leader_median",
    "gap_to_leader_max",
    "gap_to_leader_sample_count",
    "has_related_position_change",
    "related_position_change_count",
)

def load_csv_rows(
        path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header")

        fieldnames = list(reader.fieldnames)

        required_columns = {
            "driver_number",
            "lap_number",
            "date_start",
            "lap_duration",
        }

        missing_columns = required_columns - set(fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV file is missing columns: {sorted(missing_columns)}"
            )

        existing_context_fields = (
                set(fieldnames) & set(CONTEXT_FIELDS)
        )

        if existing_context_fields:
            raise ValueError(
                "CSV already contains context columns: "
                f"{sorted(existing_context_fields)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Input CSV does not contain any lap rows")

    return fieldnames, rows

def build_dated_index(
        rows: Iterable[dict[str, Any]],
) -> dict[int, list[DatedRow]]:
    grouped_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        driver_number = require_int(
            row.get("driver_number"),
            "driver_number",
        )
        grouped_rows[driver_number].append(row)

    return {
        driver_number: prepare_dated_rows(driver_rows)
        for driver_number, driver_rows in grouped_rows.items()
    }

def build_lap_window(
        row: dict[str, str],
) -> LapWindow | None:
    lap_number = int(row["lap_number"])
    date_start = row.get("date_start")
    duration_text = row.get("lap_duration")

    if not date_start or not duration_text:
        return None

    try:
        duration = float(duration_text)
    except ValueError as error:
        raise ValueError(
            f"Invalid lap duration for lap {lap_number}: "
            f"{duration_text!r}"
        ) from error

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"Invalid lap duration for lap {lap_number}: "
            f"{duration_text!r}"
        )

    start = parse_timestamp(
        date_start,
        "lap.date_start",
    )

    return LapWindow(
        lap_number=lap_number,
        start=start,
        end=start + timedelta(seconds=duration),
    )

def extract_position(
        row: dict[str, Any] | None,
) -> int | None:
    if row is None:
        return None

    position = row.get("position")

    if not isinstance(position, int) or isinstance(position, bool):
        return None

    return position

def build_numeric_summary(
        prefix: str,
        values: list[float],
) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_min": "",
            f"{prefix}_median": "",
            f"{prefix}_max": "",
            f"{prefix}_sample_count": 0,
        }

    return {
        f"{prefix}_min": round(min(values), 6),
        f"{prefix}_median": round(
            statistics.median(values),
            6,
        ),
        f"{prefix}_max": round(max(values), 6),
        f"{prefix}_sample_count": len(values),
    }

def empty_context_fields() -> dict[str, Any]:
    return {
        "has_position_data": bool_text(False),
        "position_at_start": "",
        "position_at_end": "",
        "position_update_count": 0,
        "position_changed": bool_text(False),
        "interval_min": "",
        "interval_median": "",
        "interval_max": "",
        "interval_sample_count": 0,
        "gap_to_leader_min": "",
        "gap_to_leader_median": "",
        "gap_to_leader_max": "",
        "gap_to_leader_sample_count": 0,
        "has_related_position_change": bool_text(False),
        "related_position_change_count": 0,
    }
