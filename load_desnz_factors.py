#!/usr/bin/env python3
"""
Utility to extract the DESNZ emission factors table from the case-study workbook
into a Python-friendly structure (list of dictionaries keyed by column names).

The script avoids third-party dependencies by parsing the XLSX archive directly.
Usage:
    python load_desnz_factors.py \
        --workbook "Case-Study-Data/1.1 UK DESNZ Emission Factors Database .xlsx" \
        --output "Case-Study-Data/processed/desnz_factors.json"
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from zipfile import ZipFile


NAMESPACE = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_letter_to_index(column: str) -> int:
    """Convert Excel column letters (e.g. 'A', 'AA') to a zero-based index."""
    index = 0
    for char in column.upper():
        if not ("A" <= char <= "Z"):
            raise ValueError("Unexpected column label: {0}".format(column))
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def extract_shared_strings(zf: ZipFile) -> List[str]:
    """Return the workbook's shared strings table (if present)."""
    try:
        with zf.open("xl/sharedStrings.xml") as stream:
            root = ET.parse(stream).getroot()
    except KeyError:
        return []

    strings: List[str] = []
    for si in root.findall("main:si", NAMESPACE):
        fragments: List[str] = []
        for node in si.iter():
            if node.tag.endswith("t"):
                fragments.append(node.text or "")
        strings.append("".join(fragments))
    return strings


def discover_sheets(zf: ZipFile) -> Dict[str, str]:
    """Map sheet names to their target XML worksheet paths."""
    with zf.open("xl/workbook.xml") as stream:
        workbook_root = ET.parse(stream).getroot()
    sheets_node = workbook_root.find("main:sheets", NAMESPACE)
    if sheets_node is None:
        raise ValueError("Workbook is missing sheet definitions.")

    with zf.open("xl/_rels/workbook.xml.rels") as stream:
        rels_root = ET.parse(stream).getroot()
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    rel_map = {
        rel.get("Id"): rel.get("Target")
        for rel in rels_root.findall("rel:Relationship", rel_ns)
    }

    sheets: Dict[str, str] = {}
    for sheet in sheets_node.findall("main:sheet", NAMESPACE):
        sheet_name = sheet.get("name")
        rel_id = sheet.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_map.get(rel_id)
        if sheet_name and target:
            sheets[sheet_name] = "xl/{0}".format(target)
    return sheets


def cell_value(cell: ET.Element, shared_strings: List[str]) -> Optional[str]:
    """Return the string value of a cell, resolving shared string references."""
    value_node = cell.find("main:v", NAMESPACE)
    if value_node is None:
        return None

    value = value_node.text or ""
    cell_type = cell.get("t")
    if cell_type == "s":
        return shared_strings[int(value)]
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return value


def numeric_or_text(value: Optional[str]) -> Optional[Union[float, str]]:
    """Convert numeric-looking strings to floats; keep others as-is."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    number_pattern = re.compile(r"^[+-]?((\d+(\.\d*)?)|(\.\d+))([Ee][+-]?\d+)?$")
    if number_pattern.match(text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def iter_sheet_rows(
    zf: ZipFile, sheet_target: str, shared_strings: List[str]
) -> Iterable[Dict[int, Optional[str]]]:
    """Iterate over rows as dictionaries keyed by zero-based column index."""
    with zf.open(sheet_target) as stream:
        sheet_root = ET.parse(stream).getroot()

    for row in sheet_root.findall("main:sheetData/main:row", NAMESPACE):
        cells: Dict[int, Optional[str]] = {}
        for cell in row.findall("main:c", NAMESPACE):
            ref = cell.get("r")
            if not ref:
                continue
            match = re.match(r"[A-Z]+", ref)
            if not match:
                continue
            column_index = column_letter_to_index(match.group())
            cells[column_index] = cell_value(cell, shared_strings)
        if cells:
            yield cells


def load_desnz_factors(
    workbook_path: Path, sheet_name: str = "Factors by Category"
) -> List[Dict[str, Optional[Union[float, str]]]]:
    """Return the DESNZ factors table as a list of dictionaries."""
    with ZipFile(workbook_path) as zf:
        shared_strings = extract_shared_strings(zf)
        sheets = discover_sheets(zf)
        if sheet_name not in sheets:
            raise ValueError("Sheet '{0}' not found. Available: {1}".format(sheet_name, list(sheets)))
        sheet_target = sheets[sheet_name]

        header: Dict[int, str] = {}
        data_rows: List[Dict[str, Optional[Union[float, str]]]] = []
        for row in iter_sheet_rows(zf, sheet_target, shared_strings):
            if not header:
                header_candidates = {idx: value for idx, value in row.items() if value}
                if "ID" in header_candidates.values():
                    header = {idx: str(label) for idx, label in header_candidates.items()}
                continue

            if not header:
                continue

            record: Dict[str, Optional[Union[float, str]]] = {}
            empty_count = 0
            for idx, label in header.items():
                value = numeric_or_text(row.get(idx))
                if value is None:
                    empty_count += 1
                record[label] = value

            if empty_count == len(header):
                continue
            data_rows.append(record)

    return data_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Load DESNZ emission factors into JSON-friendly form.")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("Case-Study-Data/1.1 UK DESNZ Emission Factors Database .xlsx"),
        help="Path to the DESNZ Excel workbook.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Factors by Category",
        help="Worksheet name to extract.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of rows to preview when printing (0 prints all rows).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the full dataset as JSON.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    workbook_path = args.workbook
    if not workbook_path.is_absolute():
        workbook_path = base_dir / workbook_path
    workbook_path = workbook_path.resolve()

    rows = load_desnz_factors(workbook_path, args.sheet)

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print("Saved {0} rows to {1}".format(len(rows), output_path))

    if args.limit and args.limit > 0:
        to_show = rows[: args.limit]
    else:
        to_show = rows
    print(json.dumps(to_show, indent=2, ensure_ascii=False))
    print("\nLoaded {0} rows from '{1}'.".format(len(rows), args.sheet))


if __name__ == "__main__":
    main()

