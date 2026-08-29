#!/usr/bin/env python3
"""Prefetch structured, natural-language Cala travel research for demo cities."""

import argparse
import json
import os
import re
import socket
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.cala.ai/v1/knowledge/query"
DEMO_CITIES = ["Valencia", "Barcelona", "Mexico City", "Tokyo"]
STARTER_REQUESTS_PER_MINUTE = 10


def cala_api_key() -> str:
    """Read CALA_API_KEY from the environment or the local .env.local file."""
    if api_key := os.environ.get("CALA_API_KEY"):
        return api_key

    env_file = ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?CALA_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                return match.group(1).strip().strip("\"'")

    raise SystemExit(
        "CALA_API_KEY is required. Add it to .env.local or export it in the environment."
    )


def city_slug(city: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def output_file(city: str, category: str) -> Path:
    directory = ROOT / "results" / "cala" / city_slug(city)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{category}.json"


def queries(city: str) -> dict[str, str]:
    return {
        "food": (
            f"For {city}, recommend its most distinctive local foods and essential "
            "places to try them. Return one structured row per recommendation with: "
            "name, recommendation_type, locality_or_neighbourhood, why_it_is_special, "
            "and source_url. Prioritize locally meaningful choices over generic tourist lists."
        ),
        "culture": (
            f"For {city}, recommend distinctive cultural places worth visiting, including "
            "museums, architecture, markets, and local institutions. Return one structured "
            "row per recommendation with: name, recommendation_type, locality_or_neighbourhood, "
            "why_it_is_special, and source_url."
        ),
        "outdoors": (
            f"For {city}, recommend worthwhile outdoor and nature experiences in the city "
            "or suitable for an easy day outing. Return one structured row per recommendation "
            "with: name, recommendation_type, locality, principal_activities, why_it_is_special, "
            "and source_url."
        ),
        "neighbourhoods": (
            f"For {city}, recommend the most characterful neighbourhoods to explore for a "
            "visitor who wants a local feel. Return one structured row per neighbourhood with: "
            "name, character, what_to_do_there, why_it_is_special, and source_url. Avoid generic "
            "tourist advice."
        ),
    }


def is_successful_result(result: object) -> bool:
    if not isinstance(result, dict) or result.get("error"):
        return False
    rows = result.get("results")
    return bool(rows) and not (
        isinstance(rows[0], dict) and "error" in rows[0]
    )


def fetch_query(city: str, category: str, query: str, api_key: str, timeout: int) -> Path:
    request = Request(
        API_URL,
        data=json.dumps({"input": query, "return_entities": True}).encode(),
        headers={"Content-Type": "application/json", "X-API-KEY": api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except (HTTPError, URLError, socket.timeout, TimeoutError) as error:
        result = {
            "results": [],
            "entities": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    file = output_file(city, category)
    file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch Cala travel recommendations for demo cities."
    )
    parser.add_argument(
        "cities",
        nargs="*",
        help="Cities to prefetch. Defaults to Valencia, Barcelona, Mexico City, and Tokyo.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Maximum seconds to wait for each Cala request (default: 180).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip categories that already have a non-error response file.",
    )
    args = parser.parse_args()
    cities = [city.strip() for city in args.cities] or DEMO_CITIES
    if invalid_cities := [city for city in cities if not city_slug(city)]:
        parser.error(f"Cities must contain letters or numbers: {', '.join(invalid_cities)}")

    api_key = cala_api_key()
    jobs = []
    for city in cities:
        for category, query in queries(city).items():
            file = output_file(city, category)
            if args.resume and file.exists():
                try:
                    if is_successful_result(json.loads(file.read_text())):
                        print(f"Skipping {file}", flush=True)
                        continue
                except json.JSONDecodeError:
                    pass
            jobs.append((city, category, query))

    for batch_number, offset in enumerate(
        range(0, len(jobs), STARTER_REQUESTS_PER_MINUTE), start=1
    ):
        batch = jobs[offset : offset + STARTER_REQUESTS_PER_MINUTE]
        batch_started = monotonic()
        print(f"Starting batch {batch_number}: {len(batch)} requests", flush=True)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(fetch_query, city, category, query, api_key, args.timeout)
                for city, category, query in batch
            ]
            for future in as_completed(futures):
                print(f"Wrote {future.result()}", flush=True)

        if offset + STARTER_REQUESTS_PER_MINUTE < len(jobs):
            delay = max(0, 60 - (monotonic() - batch_started))
            if delay:
                print(f"Waiting {delay:.0f}s for the rate-limit window", flush=True)
                sleep(delay)


if __name__ == "__main__":
    main()
