#!/usr/bin/env python3
"""
Normalize textual fields in the DESNZ emission factors JSON so that key descriptors are
lowercase. Specifically targets `Level 1`, `Level 2`, `Level 3`, `Level 4`,
`Column Text`, and `UOM`.

Example:
    python clean_desnz_text.py \
        --input Case-Study-Data/processed/desnz_factors.json \
        --output Case-Study-Data/processed/desnz_factors_clean.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

TARGET_FIELDS = ["Level 1", "Level 2", "Level 3", "Level 4", "Column Text", "UOM"]


def normalize_field(record: Dict, field: str) -> None:
    value = record.get(field)
    if isinstance(value, str):
        record[field] = value.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lowercase key descriptor fields in DESNZ emission factors JSON.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Case-Study-Data/processed/desnz_factors.json"),
        help="Path to the original DESNZ factors JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Case-Study-Data/processed/desnz_factors_clean.json"),
        help="Path where the cleaned JSON will be written.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    input_path = args.input if args.input.is_absolute() else base_dir / args.input
    output_path = args.output if args.output.is_absolute() else base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        records: List[Dict] = json.load(handle)

    for record in records:
        for field in TARGET_FIELDS:
            normalize_field(record, field)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    print(f"Lowercased fields {TARGET_FIELDS} for {len(records)} DESNZ entries.")
    print(f"Cleaned JSON written to {output_path}")


if __name__ == "__main__":
    main()

