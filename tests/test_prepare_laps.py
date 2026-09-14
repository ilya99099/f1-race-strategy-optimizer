"""Tests for deterministic lap enrichment and exclusion rules."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from prepare_laps import (  # noqa: E402
    TimeInterval,
    build_neutralisation_intervals,
    build_pit_in_laps,
    build_stint_index,
    enrich_laps,
    extract_deleted_laps,
    overlaps_any_interval,
)


class PrepareLapsTests(unittest.TestCase):
    def test_extract_deleted_laps_uses_message_lap_and_next_lap(self) -> None:
        rows = [
            {
                "date": "2024-03-02T15:11:18+00:00",
                "lap_number": 5,
                "message": (
                    "CAR 4 (NOR) TIME 1:37.787 DELETED - "
                    "TRACK LIMITS AT TURN 15 LAP 3"
                ),
            },
            {
                "date": "2024-03-02T15:11:19+00:00",
                "lap_number": 5,
                "message": (
                    "CAR 4 (NOR) TIME 1:37.450 DELETED - "
                    "TRACK LIMITS AT TURN 15 (NEXT LAP)"
                ),
            },
        ]

        self.assertEqual(extract_deleted_laps(rows), {(4, 3), (4, 4)})

    def test_yellow_interval_is_closed_by_matching_clear(self) -> None:
        rows = [
            {
                "date": "2024-03-02T15:19:07+00:00",
                "category": "Flag",
                "flag": "YELLOW",
                "scope": "Sector",
                "sector": 4,
            },
            {
                "date": "2024-03-02T15:19:58+00:00",
                "category": "Flag",
                "flag": "CLEAR",
                "scope": "Sector",
                "sector": 4,
            },
        ]

        intervals = build_neutralisation_intervals(rows)

        self.assertEqual(len(intervals), 1)
        self.assertEqual(
            intervals[0],
            TimeInterval(
                start=datetime(2024, 3, 2, 15, 19, 7, tzinfo=timezone.utc),
                end=datetime(2024, 3, 2, 15, 19, 58, tzinfo=timezone.utc),
            ),
        )
        self.assertTrue(
            overlaps_any_interval(
                datetime(2024, 3, 2, 15, 18, 27, tzinfo=timezone.utc),
                datetime(2024, 3, 2, 15, 20, 5, tzinfo=timezone.utc),
                intervals,
            )
        )
        self.assertFalse(
            overlaps_any_interval(
                datetime(2024, 3, 2, 15, 20, 6, tzinfo=timezone.utc),
                datetime(2024, 3, 2, 15, 21, 44, tzinfo=timezone.utc),
                intervals,
            )
        )

    def test_enrichment_labels_pit_laps_and_computes_tyre_age(self) -> None:
        stints = [
            {
                "driver_number": 4,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 2,
                "compound": "SOFT",
                "tyre_age_at_start": 3,
            },
            {
                "driver_number": 4,
                "stint_number": 2,
                "lap_start": 3,
                "lap_end": 4,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
        ]
        pit_rows = [{"driver_number": 4, "lap_number": 2}]
        laps = [
            self.make_lap(lap_number=1, date_start="2024-03-02T15:03:42+00:00"),
            self.make_lap(lap_number=2, date_start="2024-03-02T15:05:20+00:00"),
            self.make_lap(
                lap_number=3,
                date_start="2024-03-02T15:07:00+00:00",
                is_pit_out_lap=True,
            ),
            self.make_lap(lap_number=4, date_start="2024-03-02T15:09:00+00:00"),
        ]

        enriched = enrich_laps(
            laps=laps,
            stint_index=build_stint_index(stints),
            pit_in_laps=build_pit_in_laps(pit_rows),
            deleted_laps={(4, 4)},
            neutralisation_intervals=[],
        )

        self.assertEqual(enriched[0]["tyre_age"], 3)
        self.assertEqual(enriched[1]["tyre_age"], 4)
        self.assertEqual(enriched[2]["tyre_age"], 0)
        self.assertEqual(enriched[3]["tyre_age"], 1)

        self.assertEqual(enriched[0]["exclusion_reason"], "standing_start")
        self.assertEqual(enriched[1]["exclusion_reason"], "pit_in")
        self.assertEqual(enriched[2]["exclusion_reason"], "pit_out")
        self.assertEqual(enriched[3]["exclusion_reason"], "deleted_lap")
        self.assertTrue(
            all(row["is_valid_for_pace"] == "false" for row in enriched)
        )

    @staticmethod
    def make_lap(
        lap_number: int,
        date_start: str,
        is_pit_out_lap: bool = False,
    ) -> dict[str, object]:
        return {
            "session_key": 9472,
            "meeting_key": 1229,
            "driver_number": 4,
            "lap_number": lap_number,
            "date_start": date_start,
            "lap_duration": 98.0,
            "duration_sector_1": 31.0,
            "duration_sector_2": 43.0,
            "duration_sector_3": 24.0,
            "is_pit_out_lap": is_pit_out_lap,
        }


if __name__ == "__main__":
    unittest.main()
