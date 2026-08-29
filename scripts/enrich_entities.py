#!/usr/bin/env python3
"""Resolve saved Cala recommendation names to entities and cache full profiles."""

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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.cala.ai/v1"
REQUESTS_PER_MINUTE = 10
MAX_CONCURRENT_REQUESTS = 2
DEFAULT_CATEGORIES = ("food", "culture", "outdoors", "neighbourhoods")


def cala_api_key() -> str:
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


def error_result(error: Exception) -> dict[str, object]:
    return {"error": {"type": type(error).__name__, "message": str(error)}}


def request_json(request: Request, timeout: int) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (HTTPError, URLError, socket.timeout, TimeoutError) as error:
        return error_result(error)


def recommendation_candidates(city: str) -> list[dict[str, object]]:
    directory = ROOT / "results" / "cala" / city_slug(city)
    candidates: dict[str, dict[str, object]] = {}
    for category in DEFAULT_CATEGORIES:
        file = directory / f"{category}.json"
        if not file.exists():
            continue
        result = json.loads(file.read_text())
        for row in result.get("results", []):
            name = row.get("name") if isinstance(row, dict) else None
            if not isinstance(name, str) or not name.strip():
                continue
            key = name.casefold().strip()
            candidate = candidates.setdefault(
                key,
                {"name": name.strip(), "recommendations": []},
            )
            candidate["recommendations"].append({"category": category, "row": row})
    return list(candidates.values())


def search_entity(candidate: dict[str, object], api_key: str, timeout: int) -> dict[str, object]:
    name = str(candidate["name"])
    query = urlencode({"name": name, "limit": 5})
    request = Request(
        f"{API_URL}/entities?{query}",
        headers={"X-API-KEY": api_key},
        method="GET",
    )
    search = request_json(request, timeout)
    entities = search.get("entities", [])
    selected = entities[0] if entities else None
    return {
        **candidate,
        "entity_search": search,
        "selected_entity": selected,
        "selection_strategy": "top_relevance" if selected else "no_match",
    }


def retrieve_profile(record: dict[str, object], api_key: str, timeout: int) -> dict[str, object]:
    selected = record.get("selected_entity")
    if not isinstance(selected, dict) or not selected.get("id"):
        return record
    request = Request(
        f"{API_URL}/entities/{selected['id']}",
        data=b"",
        headers={"Content-Type": "application/json", "X-API-KEY": api_key},
        method="POST",
    )
    return {**record, "entity_profile": request_json(request, timeout)}


def run_in_rate_limited_batches(items: list, worker, *worker_args) -> list:
    results = []
    for batch_number, offset in enumerate(
        range(0, len(items), REQUESTS_PER_MINUTE), start=1
    ):
        batch = items[offset : offset + REQUESTS_PER_MINUTE]
        batch_started = monotonic()
        print(f"Starting batch {batch_number}: {len(batch)} requests", flush=True)
        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(batch))) as executor:
            futures = [executor.submit(worker, item, *worker_args) for item in batch]
            for future in as_completed(futures):
                results.append(future.result())
        if offset + REQUESTS_PER_MINUTE < len(items):
            delay = max(0, 60 - (monotonic() - batch_started))
            if delay:
                print(f"Waiting {delay:.0f}s for the rate-limit window", flush=True)
                sleep(delay)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a city's saved Cala recommendation rows to full Cala entities."
    )
    parser.add_argument("city", help='City whose cached results to enrich, for example "Barcelona"')
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Maximum seconds to wait for each Cala request (default: 180).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of unique recommendation names to enrich.",
    )
    args = parser.parse_args()
    city = args.city.strip()
    candidates = recommendation_candidates(city)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit(f"No saved recommendation rows found for {city}.")

    api_key = cala_api_key()
    print(f"Resolving {len(candidates)} unique recommendations", flush=True)
    resolved = run_in_rate_limited_batches(candidates, search_entity, api_key, args.timeout)
    directory = ROOT / "results" / "cala" / city_slug(city)
    profiles = run_in_rate_limited_batches(resolved, retrieve_profile, api_key, args.timeout)

    output_file = directory / "entity_enrichment.json"
    output_file.write_text(
        json.dumps({"city": city, "entities": profiles}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
