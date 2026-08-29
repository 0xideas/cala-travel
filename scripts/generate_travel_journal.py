#!/usr/bin/env python3
"""Generate imagined travel-journal entries from cached Cala entities."""

import argparse
import json
import os
import re
import socket
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_WORKERS = 4


def city_slug(city: str) -> str:
    """Convert a user-facing city name to the directory naming convention."""
    ascii_name = unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def env_value(name: str) -> str | None:
    """Read a setting from the environment or the repository's .env.local."""
    if value := os.environ.get(name):
        return value

    env_file = ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(rf"\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.*?)\s*$", line)
            if match:
                return match.group(1).strip().strip("\"'")
    return None


def openai_api_key() -> str:
    if api_key := env_value("OPENAI_API_KEY"):
        return api_key
    raise SystemExit(
        "OPENAI_API_KEY is required. Add it to .env.local or export it in the environment."
    )


def entity_context(entity: dict[str, object]) -> dict[str, object]:
    """Keep useful grounding data while omitting noisy alternative search matches."""
    recommendations = entity.get("recommendations", [])
    categories = []
    if isinstance(recommendations, list):
        categories = [
            item["category"]
            for item in recommendations
            if isinstance(item, dict) and isinstance(item.get("category"), str)
        ]

    return {
        "name": entity.get("name"),
        "categories": list(dict.fromkeys(categories)),
        "recommendations": recommendations,
        "selected_entity": entity.get("selected_entity"),
        "entity_profile": entity.get("entity_profile"),
    }


def output_text(response: dict[str, object]) -> str:
    """Extract output text from a raw Responses API response."""
    chunks = []
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if not chunks:
        status = response.get("status", "unknown")
        incomplete = response.get("incomplete_details")
        detail = f", incomplete_details: {incomplete}" if incomplete else ""
        raise ValueError(
            f"OpenAI response contained no output text (status: {status}{detail})"
        )
    return "".join(chunks)


def generate_entry(
    city: str,
    entity: dict[str, object],
    api_key: str,
    model: str,
    timeout: int,
) -> dict[str, object]:
    context = entity_context(entity)
    payload = {
        "model": model,
        "store": False,
        "instructions": (
            "Write an understated imagined travel-journal note for a prospective visitor. "
            "Use 45-80 words in first-person past tense, choosing an activity that suits the "
            "named entity. The traveller is relaxed, observant, culturally sensitive, and "
            "comfortable letting a place speak for itself. Notice one or two precise sensory "
            "details without making the visitor the centre of the scene. Keep the voice "
            "sophisticated, unhurried, warm, and low-key. Avoid superlatives, sales language, "
            "breathless excitement, exoticising descriptions, bucket-list framing, grand "
            "conclusions, and claims of discovering an 'authentic' or hidden place. Ground "
            "recognizable facts in the supplied data. You may imagine modest personal feelings "
            "and atmosphere, but never invent prices, opening hours, historical facts, named "
            "people, or accessibility claims. Do not mention the source data or AI generation."
        ),
        "input": (
            f"City: {city}\n"
            "Write one standalone journal entry for this entity:\n"
            + json.dumps(context, ensure_ascii=False)
        ),
        "reasoning": {"effort": "none"},
        "max_output_tokens": 800,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "travel_journal_entry",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"journal_entry": {"type": "string"}},
                    "required": ["journal_entry"],
                    "additionalProperties": False,
                },
            }
        },
    }
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {detail}") from error
    except (URLError, socket.timeout, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OpenAI request failed: {error}") from error

    generated = json.loads(output_text(result))
    entry = generated.get("journal_entry")
    if not isinstance(entry, str) or not entry.strip():
        raise ValueError("OpenAI response did not contain a non-empty journal_entry")

    selected = entity.get("selected_entity")
    selected = selected if isinstance(selected, dict) else {}
    return {
        "name": entity.get("name"),
        "entity_id": selected.get("id"),
        "entity_type": selected.get("entity_type"),
        "categories": context["categories"],
        "journal_entry": entry.strip(),
    }


def write_journal(path: Path, city: str, model: str, entries: list[dict[str, object]]) -> None:
    """Atomically save entries in their original entity order."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"city": city, "model": model, "entries": entries},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a virtual travel journal entry for every enriched Cala entity."
    )
    parser.add_argument("city", help='City to write about, for example "Barcelona"')
    parser.add_argument(
        "--model",
        default=env_value("OPENAI_MODEL") or DEFAULT_MODEL,
        help=f"OpenAI model to use (default: OPENAI_MODEL or {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: the city's virtual_travel_journal.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate only the first N entities; useful for testing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of concurrent API requests (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Maximum seconds to wait for each API request (default: 180).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful entries already present in the output file.",
    )
    args = parser.parse_args()

    city = args.city.strip()
    slug = city_slug(city)
    if not slug:
        parser.error("city must contain letters or numbers")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")

    directory = ROOT / "results" / "cala" / slug
    input_path = directory / "entity_enrichment.json"
    if not input_path.exists():
        raise SystemExit(f"Enrichment file not found: {input_path}")
    try:
        enrichment = json.loads(input_path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {input_path}: {error}") from error

    entities = enrichment.get("entities")
    if not isinstance(entities, list) or not entities:
        raise SystemExit(f"No entities found in {input_path}")
    entities = [entity for entity in entities if isinstance(entity, dict)]
    if args.limit is not None:
        entities = entities[: args.limit]

    output_path = args.output or directory / "virtual_travel_journal.json"
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_by_name: dict[str, dict[str, object]] = {}
    if args.resume and output_path.exists():
        try:
            previous = json.loads(output_path.read_text())
            saved_by_name = {
                entry["name"]: entry
                for entry in previous.get("entries", [])
                if isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and isinstance(entry.get("journal_entry"), str)
            }
        except json.JSONDecodeError as error:
            raise SystemExit(f"Cannot resume from invalid JSON in {output_path}: {error}") from error

    api_key = openai_api_key()
    city_name = enrichment.get("city") or city
    pending = [entity for entity in entities if entity.get("name") not in saved_by_name]
    generated_by_name = dict(saved_by_name)
    failures = 0

    print(
        f"Generating {len(pending)} entries for {city_name} "
        f"({len(saved_by_name)} already saved)",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(args.workers, max(1, len(pending)))) as executor:
        futures = {
            executor.submit(
                generate_entry, city_name, entity, api_key, args.model, args.timeout
            ): entity
            for entity in pending
        }
        for future in as_completed(futures):
            entity = futures[future]
            name = str(entity.get("name") or "Unnamed entity")
            try:
                generated_by_name[name] = future.result()
                print(f"Generated: {name}", flush=True)
            except Exception as error:  # Preserve other successful generations.
                failures += 1
                print(f"Failed: {name}: {error}", flush=True)

            ordered_entries = [
                generated_by_name[str(item.get("name"))]
                for item in entities
                if str(item.get("name")) in generated_by_name
            ]
            write_journal(output_path, str(city_name), args.model, ordered_entries)

    if not pending:
        ordered_entries = [
            generated_by_name[str(item.get("name"))]
            for item in entities
            if str(item.get("name")) in generated_by_name
        ]
        write_journal(output_path, str(city_name), args.model, ordered_entries)

    print(f"Wrote {output_path}", flush=True)
    if failures:
        raise SystemExit(
            f"{failures} entries failed; rerun with --resume to retry only missing entries."
        )


if __name__ == "__main__":
    main()
