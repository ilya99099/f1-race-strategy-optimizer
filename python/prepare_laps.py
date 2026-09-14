"""Join raw OpenF1 data and label laps that are unsuitable for pace fitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable




PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bahrain_2024" / "laps_enriched.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bahrain_2024" / "pace_laps.csv"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bahrain_2024"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "bahrain_2024" / "laps_enriched.csv"
)

DELETED_LAP_RE = re.compile(
    r"\bCAR\s+(?P<driver>\d+)\b.*\bDELETED\b.*\bLAP\s+(?P<lap>\d+)\b",
    re.IGNORECASE,
)
DELETED_NEXT_LAP_RE = re.compile(
    r"\bCAR\s+(?P<driver>\d+)\b.*\bDELETED\b.*\(NEXT LAP\)",
    re.IGNORECASE,
)

OUTPUT_FIELDS = (
    "session_key",
    "meeting_key",
    "driver_number",
    "lap_number",
    "date_start",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "compound",
    "stint_number",
    "lap_in_stint",
    "tyre_age_at_start",
    "tyre_age",
    "has_complete_timing",
    "is_start_lap",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "is_deleted",
    "is_neutralised",
    "is_valid_for_pace",
    "exclusion_reason",
)


@dataclass(frozen=True)
class StintInfo:
    stint_number: int
    lap_start: int
    lap_end: int
    compound: str
    tyre_age_at_start: int


@dataclass(frozen=True)
class TimeInterval:
    start: datetime
    end: datetime | None


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load one raw OpenF1 response and validate its outer structure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Required raw dataset not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Raw dataset is not valid JSON: {path}") from error

    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise RuntimeError(f"Expected a JSON array of objects: {path}")

    return data


def require_int(value: Any, field_name: str) -> int:
    """Return a real integer, rejecting bools and missing values."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, received {value!r}")
    return value


def parse_timestamp(value: Any, field_name: str = "date") -> datetime:
    """Parse an ISO-8601 timestamp emitted by OpenF1."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty ISO timestamp")

    normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalised)
    except ValueError as error:
        raise ValueError(f"Invalid {field_name} timestamp: {value!r}") from error


def build_stint_index(
    stints: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], StintInfo]:
    """Map every driver/lap pair to exactly one stint."""
    index: dict[tuple[int, int], StintInfo] = {}

    for row in stints:
        driver_number = require_int(row.get("driver_number"), "driver_number")
        stint_number = require_int(row.get("stint_number"), "stint_number")
        lap_start = require_int(row.get("lap_start"), "lap_start")
        lap_end = require_int(row.get("lap_end"), "lap_end")
        tyre_age_at_start = require_int(
            row.get("tyre_age_at_start"), "tyre_age_at_start"
        )
        compound = row.get("compound")

        if lap_start < 1 or lap_end < lap_start:
            raise ValueError(
                f"Invalid stint range for driver {driver_number}: "
                f"{lap_start}-{lap_end}"
            )
        if tyre_age_at_start < 0:
            raise ValueError("tyre_age_at_start cannot be negative")
        if not isinstance(compound, str) or not compound:
            raise ValueError("compound must be a non-empty string")

        stint = StintInfo(
            stint_number=stint_number,
            lap_start=lap_start,
            lap_end=lap_end,
            compound=compound,
            tyre_age_at_start=tyre_age_at_start,
        )

        for lap_number in range(lap_start, lap_end + 1):
            key = (driver_number, lap_number)
            if key in index:
                raise ValueError(
                    f"Overlapping stints for driver {driver_number}, lap {lap_number}"
                )
            index[key] = stint

    return index


def build_pit_in_laps(pit_rows: Iterable[dict[str, Any]]) -> set[tuple[int, int]]:
    """Return the driver/lap pairs on which the driver entered the pits."""
    pit_in_laps: set[tuple[int, int]] = set()

    for row in pit_rows:
        driver_number = require_int(row.get("driver_number"), "driver_number")
        lap_number = require_int(row.get("lap_number"), "lap_number")
        key = (driver_number, lap_number)
        if key in pit_in_laps:
            raise ValueError(
                f"Duplicate pit record for driver {driver_number}, lap {lap_number}"
            )
        pit_in_laps.add(key)

    return pit_in_laps


def extract_deleted_laps(
    race_control_rows: Iterable[dict[str, Any]],
) -> set[tuple[int, int]]:
    """Extract affected laps from lap-deletion messages.

    OpenF1's race-control ``lap_number`` is the lap on which the message was
    emitted, not necessarily the lap whose time was deleted.  The actual lap
    is therefore parsed from ``message``.  ``(NEXT LAP)`` refers to the lap
    immediately after the preceding explicit deletion for the same driver.
    """
    ordered_rows = sorted(
        race_control_rows,
        key=lambda row: parse_timestamp(row.get("date"), "race_control.date"),
    )
    deleted_laps: set[tuple[int, int]] = set()
    last_deleted_lap: dict[int, int] = {}

    for row in ordered_rows:
        message = row.get("message")
        if not isinstance(message, str) or "DELETED" not in message.upper():
            continue

        explicit_match = DELETED_LAP_RE.search(message)
        if explicit_match:
            driver_number = int(explicit_match.group("driver"))
            lap_number = int(explicit_match.group("lap"))
        else:
            next_match = DELETED_NEXT_LAP_RE.search(message)
            if not next_match:
                continue
            driver_number = int(next_match.group("driver"))
            previous_lap = last_deleted_lap.get(driver_number)
            if previous_lap is None:
                raise ValueError(
                    "Encountered '(NEXT LAP)' deletion without a preceding "
                    f"explicit deletion for driver {driver_number}"
                )
            lap_number = previous_lap + 1

        deleted_laps.add((driver_number, lap_number))
        last_deleted_lap[driver_number] = lap_number

    return deleted_laps


def build_neutralisation_intervals(
    race_control_rows: Iterable[dict[str, Any]],
) -> list[TimeInterval]:
    """Build yellow/red-flag intervals from race-control events.

    For the deterministic MVP, any lap overlapping a local or track-wide
    yellow, double yellow, or red flag is excluded as a whole lap.  Bahrain
    2024 contains only short local-yellow intervals.
    """
    ordered_rows = sorted(
        race_control_rows,
        key=lambda row: parse_timestamp(row.get("date"), "race_control.date"),
    )
    active: dict[tuple[str, int | None], datetime] = {}
    intervals: list[TimeInterval] = []

    for row in ordered_rows:
        if row.get("category") != "Flag":
            continue

        raw_flag = row.get("flag")
        if not isinstance(raw_flag, str):
            continue

        flag = raw_flag.upper()
        scope = str(row.get("scope") or "Track")
        sector_value = row.get("sector")
        sector = sector_value if isinstance(sector_value, int) else None
        key = (scope, sector)
        event_time = parse_timestamp(row.get("date"), "race_control.date")

        if flag in {"YELLOW", "DOUBLE YELLOW", "RED"}:
            active.setdefault(key, event_time)
        elif flag == "CLEAR":
            start = active.pop(key, None)
            if start is not None and event_time >= start:
                intervals.append(TimeInterval(start=start, end=event_time))
        elif flag == "GREEN" and scope.lower() == "track":
            track_keys = [active_key for active_key in active if active_key[0] == scope]
            for active_key in track_keys:
                start = active.pop(active_key)
                if event_time >= start:
                    intervals.append(TimeInterval(start=start, end=event_time))

    intervals.extend(TimeInterval(start=start, end=None) for start in active.values())
    return sorted(intervals, key=lambda interval: interval.start)


def overlaps_any_interval(
    start: datetime,
    end: datetime,
    intervals: Iterable[TimeInterval],
) -> bool:
    """Return whether the half-open lap interval overlaps any flagged interval."""
    for interval in intervals:
        interval_end = interval.end
        if interval_end is None:
            if end > interval.start:
                return True
        elif start < interval_end and interval.start < end:
            return True
    return False


def is_positive_number(value: Any) -> bool:
    """Return whether a timing value is numeric, finite enough, and positive."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def bool_text(value: bool) -> str:
    """Write booleans explicitly and consistently in CSV output."""
    return "true" if value else "false"


def enrich_laps(
    laps: Iterable[dict[str, Any]],
    stint_index: dict[tuple[int, int], StintInfo],
    pit_in_laps: set[tuple[int, int]],
    deleted_laps: set[tuple[int, int]],
    neutralisation_intervals: Iterable[TimeInterval],
) -> list[dict[str, Any]]:
    """Attach stint features and deterministic exclusion labels to raw laps."""
    output: list[dict[str, Any]] = []
    seen_laps: set[tuple[int, int]] = set()
    intervals = tuple(neutralisation_intervals)

    for row in laps:
        driver_number = require_int(row.get("driver_number"), "driver_number")
        lap_number = require_int(row.get("lap_number"), "lap_number")
        key = (driver_number, lap_number)

        if key in seen_laps:
            raise ValueError(
                f"Duplicate lap record for driver {driver_number}, lap {lap_number}"
            )
        seen_laps.add(key)

        stint = stint_index.get(key)
        timing_values = (
            row.get("lap_duration"),
            row.get("duration_sector_1"),
            row.get("duration_sector_2"),
            row.get("duration_sector_3"),
        )
        has_complete_timing = all(is_positive_number(value) for value in timing_values)

        date_start_value = row.get("date_start")
        is_neutralised = False
        if is_positive_number(row.get("lap_duration")) and isinstance(
            date_start_value, str
        ):
            lap_start_time = parse_timestamp(date_start_value, "lap.date_start")
            lap_end_time = lap_start_time + timedelta(
                seconds=float(row["lap_duration"])
            )
            is_neutralised = overlaps_any_interval(
                lap_start_time,
                lap_end_time,
                intervals,
            )

        raw_pit_out = row.get("is_pit_out_lap")
        if raw_pit_out is not None and not isinstance(raw_pit_out, bool):
            raise ValueError("is_pit_out_lap must be a boolean or null")

        is_start_lap = lap_number == 1
        is_pit_in_lap = key in pit_in_laps
        is_pit_out_lap = bool(raw_pit_out) or (
            lap_number > 1 and (driver_number, lap_number - 1) in pit_in_laps
        )
        is_deleted = key in deleted_laps

        reasons: list[str] = []
        if not has_complete_timing:
            reasons.append("missing_timing")
        if stint is None:
            reasons.append("missing_stint")
        if is_start_lap:
            reasons.append("standing_start")
        if is_pit_in_lap:
            reasons.append("pit_in")
        if is_pit_out_lap:
            reasons.append("pit_out")
        if is_deleted:
            reasons.append("deleted_lap")
        if is_neutralised:
            reasons.append("neutralised")

        lap_in_stint = None
        tyre_age_at_start = None
        tyre_age = None
        compound = None
        stint_number = None
        if stint is not None:
            lap_in_stint = lap_number - stint.lap_start + 1
            tyre_age_at_start = stint.tyre_age_at_start
            tyre_age = stint.tyre_age_at_start + lap_number - stint.lap_start
            compound = stint.compound
            stint_number = stint.stint_number

        output.append(
            {
                "session_key": row.get("session_key"),
                "meeting_key": row.get("meeting_key"),
                "driver_number": driver_number,
                "lap_number": lap_number,
                "date_start": date_start_value,
                "lap_duration": row.get("lap_duration"),
                "duration_sector_1": row.get("duration_sector_1"),
                "duration_sector_2": row.get("duration_sector_2"),
                "duration_sector_3": row.get("duration_sector_3"),
                "compound": compound,
                "stint_number": stint_number,
                "lap_in_stint": lap_in_stint,
                "tyre_age_at_start": tyre_age_at_start,
                "tyre_age": tyre_age,
                "has_complete_timing": bool_text(has_complete_timing),
                "is_start_lap": bool_text(is_start_lap),
                "is_pit_in_lap": bool_text(is_pit_in_lap),
                "is_pit_out_lap": bool_text(is_pit_out_lap),
                "is_deleted": bool_text(is_deleted),
                "is_neutralised": bool_text(is_neutralised),
                "is_valid_for_pace": bool_text(not reasons),
                "exclusion_reason": ";".join(reasons),
            }
        )

    return sorted(output, key=lambda item: (item["driver_number"], item["lap_number"]))


def save_csv(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write the enriched lap table with a stable column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=OUTPUT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join raw OpenF1 laps and label deterministic pace exclusions."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-driver", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    laps = load_json_array(input_dir / "laps.json")
    stints = load_json_array(input_dir / "stints.json")
    pit_rows = load_json_array(input_dir / "pit.json")
    race_control_rows = load_json_array(input_dir / "race_control.json")

    stint_index = build_stint_index(stints)
    pit_in_laps = build_pit_in_laps(pit_rows)
    deleted_laps = extract_deleted_laps(race_control_rows)
    neutralisation_intervals = build_neutralisation_intervals(race_control_rows)
    enriched_laps = enrich_laps(
        laps,
        stint_index,
        pit_in_laps,
        deleted_laps,
        neutralisation_intervals,
    )
    save_csv(enriched_laps, args.output)

    target_rows = [
        row for row in enriched_laps if row["driver_number"] == args.target_driver
    ]
    valid_target_rows = [
        row for row in target_rows if row["is_valid_for_pace"] == "true"
    ]
    reason_counts = Counter(
        reason
        for row in target_rows
        for reason in str(row["exclusion_reason"]).split(";")
        if reason
    )

    print(f"Saved {len(enriched_laps)} enriched laps to {args.output}")
    print(
        f"Driver {args.target_driver}: {len(valid_target_rows)}/{len(target_rows)} "
        "laps valid for deterministic pace fitting"
    )
    print(f"Driver {args.target_driver} exclusions: {dict(sorted(reason_counts.items()))}")

if __name__ == "__main__":
    main()