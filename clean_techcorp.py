#!/usr/bin/env python3
"""
Load the TechCorp purchase workbook into a pandas DataFrame, normalize textual fields,
and export the cleaned data to JSON for downstream processing.

Usage example:
    python clean_techcorp_to_json.py \
        --input "Case-Study-Data/1.2. Sample Company Purchase Report.xlsx" \
        --output "Case-Study-Data/processed/techcorp_purchases_clean.json"
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List
import pandas as pd  # type: ignore

# Columns considered textual for normalization.
TEXT_COLUMNS: List[str] = [
    "supplier_name",
    "item_description",
    "description_level_2",
    "category",
    "unit",
    "department",
    "notes",
]

# Keep a few punctuation characters relevant for units and IDs.
ALLOWED_PUNCT = {"%", "/", "-", "_", ".", "(", ")", ","}

# Replacement map for common unicode punctuation variants.
UNICODE_REPLACEMENTS: Dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2014": "-",
    "\u2013": "-",
    "\u2012": "-",
    "\u2010": "-",
    "\u2212": "-",
    "\u2044": "/",
    "\u00b7": " ",
}


UNIT_MAP = {
    r"^(unit|units|piece|pieces|item|items|device|devices)$": "unit",
    r"^(set|sets|bundle|bundles|pack|packs)$": "set",
    r"^(month|months|monthly|units/month)$": "month",
    r"^(quarter|q[1-4])$": "quarter",
    r"^(day|days)$": "day",
    r"^(night|nights)$": "night",
    r"^(hour|hours)$": "hour",
    r"^(week|weeks)$": "week",
    r"^(ticket|tickets|journey|journeys|ride|rides|flight|flights)$": "ticket",
    r"^(vehicle|vehicles|vehicles/month)$": "vehicle",
    r"^(kwh)$": "kwh",
    r"^(litre|litres|l)$": "litre",
    r"^(kg|kilogram|kilograms)$": "kg",
    r"^(shipment|shipments|parcel|parcels|delivery|deliveries|order|orders)$": "shipment",
    r"^(box|boxes|boxes/month)$": "box",
    r"^(desk|desks|desks/month)$": "desk",
    r"^(chair|chairs)$": "chair",
    r"^(license|licenses|user|users)$": "license",
}


def normalize_text(value):
    """Apply unicode normalization and punctuation cleanup to a string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    for source, target in UNICODE_REPLACEMENTS.items():
        text = text.replace(source, target)

    text = text.lower()

    cleaned_chars = []
    for char in text:
        if char.isalnum() or char.isspace() or char in ALLOWED_PUNCT:
            cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned if cleaned else None


def normalise_unit(unit_value):
    """Normalize unit strings using UNIT_MAP."""
    if not isinstance(unit_value, str):
        return None
    candidate = unit_value.strip().lower()
    for pattern, normalised in UNIT_MAP.items():
        if re.match(pattern, candidate):
            return normalised
    return "unit"


def split_rate(unit_value):
    """Split rate units into base unit and time basis."""
    if not isinstance(unit_value, str):
        return pd.Series({"base_unit": None, "time_basis": None})
    parts = re.split(r"/", unit_value.lower())
    if len(parts) == 2:
        base = normalise_unit(parts[0])
        time_basis = normalise_unit(parts[1])
        return pd.Series({"base_unit": base, "time_basis": time_basis})
    return pd.Series({"base_unit": normalise_unit(unit_value), "time_basis": None})


def date_to_struct(value):
    """Convert a date-like value into a structured dictionary."""
    if value in (None, "", {}):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return {
        "year": int(parsed.year),
        "month": int(parsed.month),
        "day": int(parsed.day),
    }


def split_item_description(value):
    """Split the item description into core text and trailing qualifier."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None, None

    text = str(value)
    parts = re.split(r"\s*[-–—]\s*", text, maxsplit=1)
    if len(parts) == 2:
        core = parts[0].strip()
        qualifier = parts[1].strip()
        # Only keep the qualifier if it does not look like "foo -bar" (no space after dash).
        raw_match = re.search(r"[-–—](.+)", text)
        if raw_match:
            trailing_raw = raw_match.group(1)
            if trailing_raw.startswith(" "):
                return core, qualifier
        return text.strip(), None
    return text.strip(), None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean textual columns and standardize data types."""
    df = df.copy()

    text_columns = TEXT_COLUMNS.copy()

    if "item_description" in df.columns:
        splits = df["item_description"].apply(split_item_description)
        df["item_description"] = splits.apply(lambda pair: pair[0])
        insert_pos = (
            df.columns.get_loc("description_level_2") + 1
            if "description_level_2" in df.columns
            else len(df.columns)
        )
        df.insert(insert_pos, "description_level_3", splits.apply(lambda pair: pair[1]))
        text_columns.insert(text_columns.index("description_level_2") + 1, "description_level_3")

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_text)

    # Convert Excel serial dates if read as numbers.
    if "date" in df.columns:
        if pd.api.types.is_numeric_dtype(df["date"]):
            df["date"] = pd.to_datetime("1899-12-30") + pd.to_timedelta(df["date"], unit="D")
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.normalize()

    # Ensure numeric fields are floats where appropriate.
    numeric_columns = ["quantity", "unit_price_gbp", "total_spend_gbp"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "unit" in df.columns:
        df[["base_unit", "time_basis"]] = df["unit"].apply(split_rate)

    return df


def dataframe_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    for record in records:
        if "date" in record:
            record["date"] = date_to_struct(record["date"])
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TechCorp purchase Excel data and export to JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Case-Study-Data/1.2. Sample Company Purchase Report Annotated.xlsx"),
        help="Path to the source Excel workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Case-Study-Data/processed/techcorp_purchases_clean_annotated.json"),
        help="Destination path for the cleaned JSON file.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=0,
        help="Workbook sheet name/index to load.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    input_path = args.input
    if not input_path.is_absolute():
        input_path = base_dir / input_path
    input_path = input_path.resolve()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, sheet_name=args.sheet, dtype=str)
    df = df.apply(pd.to_numeric, errors="ignore")  # revert numeric columns automatically inferred.
    cleaned_df = normalize_dataframe(df)
    records = dataframe_to_records(cleaned_df)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    print("Saved {0} cleaned rows to {1}".format(len(records), output_path))


if __name__ == "__main__":
    main()
