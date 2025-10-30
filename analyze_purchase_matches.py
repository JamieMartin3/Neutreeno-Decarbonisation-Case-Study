#!/usr/bin/env python3
"""
Augment purchase-to-emission match results with certainty scores derived from the
matching script's uncertainty values and produce diagnostic plots based only on the
top match per purchase.

Example:
    python analyze_purchase_matches.py \
        --input Case-Study-Data/processed/purchase_emission_matches.json \
        --output Case-Study-Data/processed/purchase_emission_matches_certainty.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def to_float(value: Any) -> float:
    try:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_valid_number(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def plot_bar(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filepath: Path,
    rotation: int = 60,
) -> None:
    plt.figure(figsize=(max(6, len(labels) * 0.4), 4))
    plt.bar(labels, values, color="#4C72B0")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def plot_certainty_metrics(records: List[Dict[str, Any]], max_score: float, plots_dir: Path) -> None:
    labels: List[str] = []
    certainties: List[float] = []
    ef_certainty: List[Tuple[str, float]] = []
    ef_spend_certainty: List[Tuple[str, float]] = []

    for idx, record in enumerate(records):
        matches = record.get("matches") or record.get("selected_matches") or []
        if not matches:
            continue

        label = record.get("purchase_id") or record.get("po_number") or f"purchase_{idx}"
        best_match = matches[0]
        certainty = best_match.get("certainty")
        if certainty is None:
            uncertainty = best_match.get("uncertainty")
            if isinstance(uncertainty, (int, float)):
                certainty = max(0.0, min(1.0, 1.0 - float(uncertainty)))
            else:
                certainty = 0.0
            best_match["certainty"] = certainty
        if not isinstance(certainty, (int, float)):
            continue

        labels.append(str(label))
        certainties.append(float(certainty))

        emission_ratio = to_float(best_match.get("emission_ratio"))
        ghg_value = emission_ratio
        if is_valid_number(ghg_value):
            ef_certainty.append((str(label), ghg_value * float(certainty)))

            spend = to_float(best_match.get("total_spend_gbp"))
            if is_valid_number(spend):
                ef_spend_certainty.append((str(label), ghg_value * spend * float(certainty)))

    if labels:
        plot_bar(
            labels,
            certainties,
            "Relative Certainty by Purchase",
            "Relative Certainty",
            plots_dir / "purchase_certainty.png",
        )

    if ef_certainty:
        labels_all, values_all = zip(*ef_certainty)
        plot_bar(
            list(labels_all),
            list(values_all),
            "Emission Factor × Certainty (All Purchases)",
            "kg CO2e × Certainty",
            plots_dir / "emission_certainty_all.png",
        )

        top10 = sorted(ef_certainty, key=lambda x: x[1], reverse=True)[:10]
        if top10:
            labels_top, values_top = zip(*top10)
            plot_bar(
                list(labels_top),
                list(values_top),
                "Emission Factor × Certainty (Top 10)",
                "kg CO2e × Certainty",
                plots_dir / "emission_certainty_top10.png",
                rotation=45,
            )

    if ef_spend_certainty:
        labels_all, values_all = zip(*ef_spend_certainty)
        plot_bar(
            list(labels_all),
            list(values_all),
            "Emission Factor × Spend × Certainty (All Purchases)",
            "kg CO2e × GBP × Certainty",
            plots_dir / "emission_spend_certainty_all.png",
        )

        top10_spend = sorted(ef_spend_certainty, key=lambda x: x[1], reverse=True)[:10]
        if top10_spend:
            labels_top, values_top = zip(*top10_spend)
            plot_bar(
                list(labels_top),
                list(values_top),
                "Emission Factor × Spend × Certainty (Top 10)",
                "kg CO2e × GBP × Certainty",
                plots_dir / "emission_spend_certainty_top10.png",
                rotation=45,
            )
        return


def inject_top_certainty(records: List[Dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        matches = record.get("matches") or record.get("selected_matches") or []
        if not matches:
            record["top_certainty"] = 0.0
            continue
        top_match = matches[0]
        uncertainty = top_match.get("uncertainty")
        if isinstance(uncertainty, (int, float)):
            certainty = max(0.0, min(1.0, 1.0 - float(uncertainty)))
        else:
            certainty = 0.0
        top_match["certainty"] = certainty
        record["top_certainty"] = certainty


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise certainty metrics from purchase-emission matches.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_emission_matches.json"),
        help="Path to the input matches JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_emission_matches_certainty.json"),
        help="Path to write the augmented JSON with relative certainty.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("Case-Study-Data/processed/plots"),
        help="Directory where diagnostic plots will be saved.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    input_path = args.input if args.input.is_absolute() else base_dir / args.input
    output_path = args.output if args.output.is_absolute() else base_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else base_dir / args.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(
            f"Input file '{input_path}' was not found. "
            "Run match_purchases_to_desnz.py first to generate the matches JSON."
        )
        return

    with input_path.open("r", encoding="utf-8") as handle:
        records: List[Dict[str, Any]] = json.load(handle)

    inject_top_certainty(records)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    plot_certainty_metrics(records, 1.0, plots_dir)

    print(
        "Annotated {0} records with certainty derived from match uncertainty. Output written to {1}".format(
            len(records), output_path
        )
    )


if __name__ == "__main__":
    main()
