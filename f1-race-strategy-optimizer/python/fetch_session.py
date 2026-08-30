"""Resolve and save metadata for one OpenF1 session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://api.openf1.org/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "bahrain_2024" / "session.json"


def build_sessions_url(year: int, country: str, session_name: str) -> str:
    """Build an encoded OpenF1 sessions query."""
    query = urlencode(
        {
            "year": year,
            "country_name": country,
            "session_name": session_name,
        }
    )
    return f"{BASE_URL}/sessions?{query}"


def fetch_json_array(url: str) -> list[dict[str, Any]]:
    """Fetch a JSON array from OpenF1 and validate its outer structure."""
    request = Request(
        url,
        headers={"User-Agent": "f1-race-strategy-optimizer/0.1"},
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"OpenF1 request failed: {error}") from error

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError("OpenF1 returned invalid JSON") from error

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise RuntimeError("Expected OpenF1 to return a JSON array of objects")

    return data


def select_single_session(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Require the descriptive query to identify exactly one session."""
    if not sessions:
        raise ValueError("No session matched the supplied filters")

    if len(sessions) != 1:
        candidates = [
            {
                "session_key": item.get("session_key"),
                "session_name": item.get("session_name"),
                "date_start": item.get("date_start"),
            }
            for item in sessions
        ]
        raise ValueError(
            "Expected exactly one session, "
            f"but received {len(sessions)} candidates: {candidates}"
        )

    session = sessions[0]
    required_fields = {
        "session_key",
        "meeting_key",
        "session_name",
        "session_type",
        "year",
        "country_name",
        "date_start",
    }
    missing_fields = sorted(required_fields.difference(session))
    if missing_fields:
        raise ValueError(f"Session is missing required fields: {missing_fields}")

    return session


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """Create the output directory and save stable, readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve one historical OpenF1 session and save its metadata."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = build_sessions_url(args.year, args.country, args.session_name)
    sessions = fetch_json_array(url)
    session = select_single_session(sessions)
    save_json(session, args.output)

    print(
        f"Saved {session['country_name']} {session['year']} "
        f"{session['session_name']} (session_key={session['session_key']}) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
