#!/usr/bin/env python3
"""Query Cala for Toulouse travel recommendations."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.cala.ai/v1/knowledge/search"
QUERIES = {
    "local_foods": "What local foods and traditional dishes should a visitor try in Toulouse and the surrounding Occitanie region?",
    "local_drinks_and_party": "What local drinks, nightlife, bars, and party experiences are characteristic of Toulouse and the surrounding Occitanie region?",
    "natural_attractions": "What are the best natural attractions and outdoor places to visit around Toulouse and in the surrounding Occitanie region?",
    "cultural_attractions_and_events": "What are the main cultural attractions and notable recurring cultural events in Toulouse?",
}


def main() -> None:
    secret = (ROOT / "secrets").read_text().strip()
    api_key = secret.partition(":")[2].strip() if ":" in secret else secret
    output_dir = ROOT / "results" / "cala"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, query in QUERIES.items():
        request = Request(
            API_URL,
            data=json.dumps({"input": query}).encode(),
            headers={"Content-Type": "application/json", "X-API-KEY": api_key},
            method="POST",
        )
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        (output_dir / f"{name}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"Wrote {output_dir / f'{name}.json'}")


if __name__ == "__main__":
    main()
