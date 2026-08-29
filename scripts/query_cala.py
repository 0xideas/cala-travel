#!/usr/bin/env python3
"""Query Cala for structured travel recommendations for a city."""

import argparse
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.cala.ai/v1/knowledge/query"


def queries(city: str) -> dict[str, str]:
    today = date.today().isoformat()
    return {
        "local_foods": f"""Return at most 10 foods specifically associated with {city} or its immediate surrounding region, excluding items associated only with distant parts of the wider region. Return one row per food with: name, category, locality, short_description, strength_of_local_association, source_url.""",
        "local_drinks_and_party": f"""Return at most 12 drinks and nightlife experiences specifically associated with {city} or its immediate surrounding region. Return one row per item with: name, item_type, locality_or_neighbourhood, short_description, current_status, source_url, source_date. For venues, include only those shown to be operating as of {today}.""",
        "natural_attractions": f"""Return at most 12 natural attractions suitable for a day trip from {city}, limited to approximately two hours of travel each way. Return one row per attraction with: name, locality, attraction_type, approximate_distance_km, approximate_travel_time, principal_activities, source_url. Exclude urban monuments and overnight destinations.""",
        "cultural_attractions_and_events": f"""Return at most 15 cultural attractions and recurring cultural events physically located in {city}. Return one row per item with: name, item_type, locality_or_venue, short_description, current_status_or_confirmed_dates, official_url, source_url, source_date. For events, include only editions confirmed on or after {today}; omit unverified events.""",
    }


def city_slug(city: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("city_name", help='City to query, for example "Toulouse"')
    city = parser.parse_args().city_name.strip()
    slug = city_slug(city)
    if not slug:
        parser.error("city_name must contain letters or numbers")

    secret = (ROOT / "secrets").read_text().strip()
    api_key = secret.partition(":")[2].strip() if ":" in secret else secret
    output_dir = ROOT / "results" / "cala" / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, query in queries(city).items():
        request = Request(
            API_URL,
            data=json.dumps({"input": query, "return_entities": False}).encode(),
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
