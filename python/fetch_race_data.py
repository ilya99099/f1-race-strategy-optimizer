"""Download the raw OpenF1 datasets needed for one race."""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from fetch_session import BASE_URL, fetch_json_array


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_FILE = (
        PROJECT_ROOT / "data" / "raw" / "bahrain_2024" / "session.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "bahrain_2024"

ENDPOINTS = (
    "drivers",
    "laps",
    "stints",
    "pit",
    "race_control",
    "position",
    "intervals",
    "overtakes",
)


def load_session_key(session_file: Path) -> int:
    """Read and validate the numeric session key saved by fetch_session.py."""
    try:
        metadata = json.loads(session_file.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Session metadata file not found: {session_file}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Session metadata is not valid JSON: {session_file}") from error

    if not isinstance(metadata, dict):
        raise RuntimeError("Expected session metadata to be a JSON object")

    session_key = metadata.get("session_key")
    if not isinstance(session_key, int) or isinstance(session_key, bool):
        raise RuntimeError("Session metadata must contain an integer session_key")

    return session_key


def build_endpoint_url(endpoint: str, session_key: int) -> str:
    """Build an OpenF1 endpoint URL filtered to one session."""
    query = urlencode({"session_key": session_key})
    return f"{BASE_URL}/{endpoint}?{query}"


def save_json_array(data: list[dict[str, Any]], output_path: Path) -> None:
    """Save one raw endpoint response as readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download raw OpenF1 datasets for one resolved session."
    )
    parser.add_argument("--session-file", type=Path, default=DEFAULT_SESSION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_key = load_session_key(args.session_file)

    for endpoint in ENDPOINTS:
        url = build_endpoint_url(endpoint, session_key)
        rows = fetch_json_array(url)
        output_path = args.output_dir / f"{endpoint}.json"
        save_json_array(rows, output_path)
        print(f"Saved {len(rows)} {endpoint} rows to {output_path}")


if __name__ == "__main__":
    main()
