#!/usr/bin/env python3
"""
Aggregate matched emission factors into high-level categories and visualise the total
emission intensity (GHG factor × certainty) across those categories.

Example:
    python summarize_emission_intensity.py \
        --input Case-Study-Data/processed/purchase_emission_matches_certainty.json \
        --plot Case-Study-Data/processed/plots/emission_intensity_pie.png
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

# Simple keyword maps for categorisation; matched on lowercase strings.
CATEGORY_KEYWORDS = {
    "travel": {"travel", "taxi", "flight", "air", "train", "ticket", "journey", "ride"},
    "accommodation": {"hotel", "accommodation", "lodging"},
    "homeworking": {"home working", "remote work", "home office"},
    "office_equipment": {"laptop", "computer", "monitor", "keyboard", "printer", "mouse", "id card"},
    "office_supplies": {"paper", "stationery", "toner", "pen", "notebook", "stapler"},
    "energy": {"electricity", "gas", "energy", "fuel", "diesel", "petrol"},
    "shipping_transport": {"delivery", "shipping", "courier", "logistics", "parcel"},
    "material_use": {"furniture", "desk", "chair", "fixture", "fit-out"},
    "services": {"consulting", "professional services", "maintenance", "support"},
}


def normalise_score(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return float("nan")
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return float("nan")
        return val
    except (TypeError, ValueError):
        return float("nan")


def identify_category(text: str) -> str:
    lowered = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def extract_intensity(record: Dict[str, Any]) -> Tuple[str, float]:
    purchase = record.get("purchase") or {}
    matches = record.get("selected_matches") or []
    if not matches:
        return "other", 0.0

    primary = matches[0]
    certainty = normalise_score(primary.get("relative_certainty") or primary.get("score"))
    emission = primary.get("matched_emission") or {}
    ghg = normalise_score(emission.get("GHG Conversion Factor 2025"))
    if not (certainty == certainty and ghg == ghg):
        return "other", 0.0

    emitted = certainty * ghg

    description_fields = [
        purchase.get("item_description"),
        purchase.get("description_level_2"),
        purchase.get("description_level_3"),
        purchase.get("category"),
        primary.get("matched_emission_text"),
    ]
    combined = " | ".join(str(field) for field in description_fields if isinstance(field, str))
    category = identify_category(combined) if combined else "other"
    return category, emitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise emission intensity across high-level categories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_emission_matches_certainty.json"),
        help="Matches JSON with relative certainty scores.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("Case-Study-Data/processed/plots/emission_intensity_pie.png"),
        help="Location to save the emission intensity pie chart.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = args.input if args.input.is_absolute() else base_dir / args.input
    plot_path = args.plot if args.plot.is_absolute() else base_dir / args.plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        records: List[Dict[str, Any]] = json.load(handle)

    category_totals: Dict[str, float] = {}
    for record in records:
        category, intensity = extract_intensity(record)
        category_totals[category] = category_totals.get(category, 0.0) + intensity

    # Remove zero entries to keep the chart clean.
    category_totals = {k: v for k, v in category_totals.items() if v > 0}

    if not category_totals:
        print("No emission intensity data available; skipping plot.")
        return

    labels = list(category_totals.keys())
    sizes = list(category_totals.values())

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Emission Intensity by Category (Factor × Certainty)")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("Emission intensity breakdown saved to", plot_path)


if __name__ == "__main__":
    main()

