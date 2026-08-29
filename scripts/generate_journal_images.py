#!/usr/bin/env python3
"""Generate transparent, blue-pen journal illustrations with fal.

Each subject is a separate FLUX.2 [pro] request: that model has no ``num_images``
input. A BiRefNet pass then removes the generated background and returns a PNG
with an alpha channel.

Examples:
  FAL_KEY=... python scripts/generate_journal_images.py "a tram" "a coffee cup"
  FAL_KEY=... python scripts/generate_journal_images.py --subjects-file subjects.txt
"""

import argparse
import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FLUX_ENDPOINT = "https://fal.run/fal-ai/flux-2-pro"
BACKGROUND_REMOVAL_ENDPOINT = "https://fal.run/fal-ai/birefnet/v2"

PROMPT_TEMPLATE = """Generate a hand-drawn picture of {subject}, as it would be drawn in a personal journal by an ordinary person.

Make it a very simple, recognizable blue ballpoint-pen sketch. Preserve the overall shape and remove non-essential details. Use slightly wobbly, imperfect hand-drawn lines and a small, natural tilt; avoid a rigid rectangular composition. It should feel like an average person's quick journal drawing, not polished illustration or technical art.

Show the object isolated: no scene, no setting, no paper texture, no border, no shadow, and no background. Leave generous empty space around it."""


def fal_key() -> str:
    """Read FAL_KEY from the environment or the local, ignored .env.local file."""
    if key := os.environ.get("FAL_KEY"):
        return key

    env_file = ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            match = re.match(r"\s*(?:export\s+)?FAL_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                return match.group(1).strip().strip("\"'")

    raise SystemExit("FAL_KEY is required. Export it or add it to .env.local (which is gitignored).")


def request_json(endpoint: str, payload: dict, key: str, timeout: int) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"fal returned HTTP {error.code}: {detail}") from error
    except (URLError, socket.timeout, TimeoutError) as error:
        raise RuntimeError(f"fal request failed: {error}") from error


def download(url: str, destination: Path, timeout: int) -> None:
    request = Request(url, headers={"User-Agent": "cala-travel-journal-image-script"})
    try:
        with urlopen(request, timeout=timeout) as response:
            destination.write_bytes(response.read())
    except (HTTPError, URLError, socket.timeout, TimeoutError) as error:
        raise RuntimeError(f"Could not download generated image: {error}") from error


def safe_filename(subject: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")[:72] or "image"
    return f"{index:02d}-{slug}.png"


def generate_one(subject: str, index: int, key: str, output_dir: Path, timeout: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(subject=subject)
    generated = request_json(
        FLUX_ENDPOINT,
        {
            "prompt": prompt,
            "image_size": "square_hd",
            "output_format": "png",
            "enable_safety_checker": True,
        },
        key,
        timeout,
    )
    try:
        generated_url = generated["images"][0]["url"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected FLUX.2 response: {generated}") from error

    cutout = request_json(
        BACKGROUND_REMOVAL_ENDPOINT,
        {
            "image_url": generated_url,
            "model": "General Use (Light)",
            "operating_resolution": "2048x2048",
            "refine_foreground": True,
            "output_format": "png",
        },
        key,
        timeout,
    )
    try:
        transparent_url = cutout["image"]["url"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Unexpected BiRefNet response: {cutout}") from error

    filename = safe_filename(subject, index)
    download(transparent_url, output_dir / filename, timeout)
    return {"subject": subject, "file": filename, "source_url": generated_url, "transparent_url": transparent_url}


def read_subjects(args: argparse.Namespace) -> list[str]:
    subjects = [subject.strip() for subject in args.subjects if subject.strip()]
    if args.subjects_file:
        subjects.extend(
            line.strip()
            for line in args.subjects_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.enrichment_file:
        try:
            records = json.loads(args.enrichment_file.read_text())["entities"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(f"Could not read entities from {args.enrichment_file}: {error}") from error
        subjects.extend(
            str(record["name"]).strip()
            for record in records
            if isinstance(record, dict)
            and isinstance(record.get("selected_entity"), dict)
            and str(record.get("name", "")).strip()
        )
    if not subjects:
        raise SystemExit("Provide at least one subject, --subjects-file, or --enrichment-file.")
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-generate transparent journal sketch PNGs with fal.")
    parser.add_argument("subjects", nargs="*", help='Subjects to draw, e.g. "a tram" "a coffee cup".')
    parser.add_argument("--subjects-file", type=Path, help="Text file with one subject per line; # comments are ignored.")
    parser.add_argument(
        "--enrichment-file",
        type=Path,
        help="Generate the original names of records that have a selected entity, from an entity_enrichment.json file.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "journal-images")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel subjects (default: 2; raise only within your fal concurrency limit).")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds allowed for each API call or download (default: 180).")
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout < 1:
        parser.error("--concurrency and --timeout must be positive integers.")

    subjects = read_subjects(args)
    key = fal_key()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(subjects)} image(s), with up to {args.concurrency} subjects in parallel.", flush=True)

    manifest: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(subjects))) as executor:
        futures = {
            executor.submit(generate_one, subject, index, key, args.output_dir, args.timeout): subject
            for index, subject in enumerate(subjects, start=1)
        }
        for future in as_completed(futures):
            subject = futures[future]
            try:
                result = future.result()
                manifest.append(result)
                print(f"Wrote {args.output_dir / result['file']}", flush=True)
            except Exception as error:
                print(f"FAILED {subject}: {error}", flush=True)

    manifest.sort(key=lambda item: item["file"])
    manifest_file = args.output_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_file}", flush=True)
    if len(manifest) != len(subjects):
        raise SystemExit("One or more generations failed; see messages above.")


if __name__ == "__main__":
    main()
