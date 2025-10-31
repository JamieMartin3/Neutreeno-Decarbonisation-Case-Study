#!/usr/bin/env python3
"""
Compute semantic similarities between TechCorp purchase records and DESNZ emission
factors using Sentence-BERT embeddings, returning the closest emission factor for each
purchase.

Example:
    python match_purchases_to_desnz.py \
        --purchases Case-Study-Data/processed/techcorp_purchases_clean.json \
        --desnz Case-Study-Data/processed/desnz_factors.json \
        --output Case-Study-Data/processed/purchase_emission_matches.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


PURCHASE_FIELDS = [
    "item_description",
    "description_level_2",
    "category",
    "department",
    "notes",
]

DESNZ_FIELDS = [
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
]


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, found {type(data).__name__}")
        return data


def to_text(parts: List[Optional[Any]]) -> str:
    tokens: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, dict):
            # Flatten dicts (e.g., structured dates) into a readable form
            sub_tokens = [
                str(value)
                for key, value in part.items()
                if value not in (None, "", [], {})
            ]
            tokens.extend(sub_tokens)
            continue
        if isinstance(part, (list, tuple)):
            tokens.extend(str(item) for item in part if item not in (None, "", [], {}))
            continue
        text = str(part).strip()
        if text:
            tokens.append(text)
    return ", ".join(tokens)


def build_purchase_text(entry: Dict[str, Any]) -> str:
    parts = [entry.get(field) for field in PURCHASE_FIELDS]
    return to_text(parts)


def build_desnz_text(entry: Dict[str, Any]) -> str:
    parts = [entry.get(field) for field in DESNZ_FIELDS]
    return to_text(parts)


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Match TechCorp purchases to DESNZ emission factors via embeddings.")
    parser.add_argument(
        "--purchases",
        type=Path,
        default=Path("Case-Study-Data/processed/techcorp_purchases_clean_annotated.json"),
        help="Path to the cleaned TechCorp purchases JSON.",
    )
    parser.add_argument(
        "--desnz",
        type=Path,
        default=Path("Case-Study-Data/processed/desnz_factors_clean.json"),
        help="Path to the DESNZ emission factors JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_emission_matches.json"),
        help="Destination path for the selected matches JSON (1 or top 3 depending on confidence).",
    )
    parser.add_argument(
        "--output-all",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_emission_matches_all.json"),
        help="Destination path for the detailed matches JSON (top 3 for every purchase).",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchases_missing_emissions.json"),
        help="Path to list purchases whose retained matches lack emission factor values.",
    )
    parser.add_argument(
        "--ambiguous-output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchases_ambiguous.json"),
        help="Path to list purchases flagged as ambiguous (top scores within margin).",
    )
    parser.add_argument(
        "--top1-output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchases_confident_top1.json"),
        help="Path to list purchases confident enough to keep only the top match.",
    )
    parser.add_argument(
        "--top3-output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchases_needing_top3.json"),
        help="Path to list purchases requiring the top three matches.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("Case-Study-Data/processed/purchase_match_summary.json"),
        help="Path to write aggregate matching statistics.",
    )
    # Can use sentence-transformers/multi-qa-mpnet-base-dot-v1
    # Can use sentence-transformers/all-mpnet-base-v2
    # Can use BAAI/bge-small-en-v1.5
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-BERT model to use for embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size when encoding texts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of matches to retain per purchase (used when similarity falls below the threshold).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="If the top score meets or exceeds this threshold, only the best match is returned; otherwise the top-k matches are retained.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to write a log of uncertain matches (scores below threshold or missing emission data).",
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.005,
        help="Flag matches as ambiguous when the top-three scores are within this margin.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # ----------------------------------------------------------------------------------------------------------
    # Prepare Input and Output Paths
    # ----------------------------------------------------------------------------------------------------------

    purchases_path = args.purchases if args.purchases.is_absolute() else base_dir / args.purchases
    desnz_path = args.desnz if args.desnz.is_absolute() else base_dir / args.desnz
    output_path = args.output if args.output.is_absolute() else base_dir / args.output
    output_all_path = args.output_all if args.output_all.is_absolute() else base_dir / args.output_all
    missing_output_path = args.missing_output if args.missing_output.is_absolute() else base_dir / args.missing_output
    ambiguous_output_path = args.ambiguous_output if args.ambiguous_output.is_absolute() else base_dir / args.ambiguous_output
    top1_output_path = args.top1_output if args.top1_output.is_absolute() else base_dir / args.top1_output
    top3_output_path = args.top3_output if args.top3_output.is_absolute() else base_dir / args.top3_output
    summary_output_path = args.summary_output if args.summary_output.is_absolute() else base_dir / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_all_path.parent.mkdir(parents=True, exist_ok=True)
    missing_output_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguous_output_path.parent.mkdir(parents=True, exist_ok=True)
    top1_output_path.parent.mkdir(parents=True, exist_ok=True)
    top3_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = None

    if args.log_file:
        log_path = args.log_file if args.log_file.is_absolute() else base_dir / args.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)


    # ----------------------------------------------------------------------------------------------------------
    # Perform Embedding and Similarity Computation for each Purchase and DESNZ Entry
    # ----------------------------------------------------------------------------------------------------------

    print("[1/5] Loading data")
    purchase_records = load_json(purchases_path)
    desnz_records = load_json(desnz_path)
    filtered_desnz_records = [
        entry for entry in desnz_records if isinstance(entry.get("GHG/Unit"), str) and entry["GHG/Unit"].strip().lower() == "kg co2e"
    ]
    if not filtered_desnz_records:
        raise SystemExit("No DESNZ records found with GHG/Unit equal to 'kg CO2e'.")
    if len(filtered_desnz_records) != len(desnz_records):
        print(
            f"    Filtered out {len(desnz_records) - len(filtered_desnz_records)} DESNZ entries without 'kg CO2e' units."
        )
    desnz_records = filtered_desnz_records
    print(f"    Loaded {len(purchase_records)} purchase records and {len(desnz_records)} DESNZ entries")

    print("[2/5] Building text representations")
    purchase_texts = [build_purchase_text(entry) for entry in purchase_records]
    desnz_texts = [build_desnz_text(entry) for entry in desnz_records]

    # Embedding stage — map purchase and DESNZ text into the shared SBERT vector space.
    print("[3/5] Loading model and encoding texts")
    model = SentenceTransformer(args.model)
    desnz_embeddings = encode_texts(model, desnz_texts, batch_size=args.batch_size)
    purchase_embeddings = encode_texts(model, purchase_texts, batch_size=args.batch_size)

    # Similarity stage — once both corpora are embedded we can compare them via cosine similarity.
    print(f"[4/5] Computing similarity matrix")
    similarity_matrix = np.matmul(purchase_embeddings, desnz_embeddings.T)

    score_matrix = similarity_matrix

    print("[5/5] Preparing output")
    all_matches: List[Dict[str, Any]] = []
    selected_results: List[Dict[str, Any]] = []
    uncertain_logs: List[str] = []
    missing_emission_logs: List[str] = []
    missing_entries: List[Dict[str, Any]] = []
    ambiguous_entries: List[Dict[str, Any]] = []
    top1_entries: List[Dict[str, Any]] = []
    top3_entries: List[Dict[str, Any]] = []
    ambiguous_ids: List[Any] = []
    unmatched_ids: List[Any] = []

    requested_top_k = min(max(args.top_k, 1), len(desnz_records))
    # Always capture at least the top three matches so ambiguous reports can surface them.
    detail_top_k = min(max(requested_top_k, 3), len(desnz_records))

    low_bound = float(np.min(similarity_matrix))
    high_bound = float(np.max(similarity_matrix))
    # Denominator for confidence scaling; avoid zero division.
    denom = high_bound - low_bound if (high_bound - low_bound) != 0 else 1.0

    for idx, record in enumerate(purchase_records):
        score_vector = score_matrix[idx]
        cosine_vector = similarity_matrix[idx]
        sorted_indices = np.argsort(-score_vector)
        selected_idxs = sorted_indices[:detail_top_k]

        matches: List[Dict[str, Any]] = []
        for rank, emission_idx in enumerate(selected_idxs, start=1):
            emission_idx_int = int(emission_idx)
            emission = desnz_records[emission_idx_int]
            match_payload = {
                "rank": rank,
                "matched_emission_index": emission_idx_int,
                "matched_emission": emission,
                "score": float(score_vector[emission_idx]),
                "cosine_similarity": float(cosine_vector[emission_idx]),
                "desnz_id": emission.get("ID"),
            }
            matches.append(match_payload)

        primary_matches = matches[:requested_top_k]
        if not primary_matches:
            continue

        purchase_id = record.get("purchase_id")
        purchase_identifier = purchase_id if purchase_id else f"purchase_{idx}"
        top_score = primary_matches[0]["score"]
        second_score = primary_matches[1]["score"] if len(primary_matches) > 1 else high_bound
        margin = max(0.0, top_score - second_score)
        uniqueness_value = min(1.0, max(0.0, margin / 0.005))

        # ----------------------------------------------------------------------------------------------------------
        # Uncertainty Analysis
        # ----------------------------------------------------------------------------------------------------------

        # Uncertainty stage — derive interpretable diagnostics from the raw scores so analysts can vet results.
        # Confidence scales the cosine similarity against the observed range for that run.
        # Uniqueness measures the separation between the best and second-best scores; small margins reduce uniqueness.
        # Uncertainty is the remaining risk after combining both signals (high when confidence is low or options are tied).
        for match in matches:

            raw_score = float(match["score"])
            confidence = (raw_score - low_bound) / denom
            confidence = max(0.0, min(1.0, confidence))
            raw_conf = 0.7 * confidence + 0.3 * uniqueness_value
            match["confidence"] = confidence
            match["uniqueness"] = uniqueness_value
            match["uncertainty"] = max(0.0, min(1.0, 1.0 - raw_conf))

        best_confidence = primary_matches[0]["confidence"]
        ambiguous = uniqueness_value < 1.0 - 1e-9

        # ----------------------------------------------------------------------------------------------------------
        # Categorisation and Output Preparation
        # ----------------------------------------------------------------------------------------------------------
        # Categorises each match depending on uniqueness and confidence metrics, retaining either the top match
        # or the top three matches as appropriate. Logs uncertain matches for analyst review.

        if ambiguous:
            uncertain_logs.append(
                "Ambiguous match (uniqueness={0:.3f}) for purchase {1}".format(
                    uniqueness_value,
                    purchase_identifier,
                )
            )

        if best_confidence < 0.5:
            retained_matches = []
            needs_top3 = False
            uncertain_logs.append(
                "No match retained (confidence {0:.3f} < 0.50) for purchase {1}".format(
                    best_confidence, purchase_identifier
                )
            )
        elif best_confidence <= 0.6 and len(primary_matches) > 1:
            retain_count = min(3, len(primary_matches))
            retained_matches = primary_matches[:retain_count]
            needs_top3 = retain_count > 1
        else:
            retained_matches = primary_matches[:1]
            needs_top3 = False

        missing_for_purchase = False
        for match in retained_matches:
            emission = match["matched_emission"]
            ghg_val = emission.get("GHG Conversion Factor 2025")
            if ghg_val in (None, "", []):
                missing_for_purchase = True
                missing_emission_logs.append(
                    "Missing emission factor for purchase {0}, emission index {1}".format(
                        purchase_identifier,
                        match["matched_emission_index"],
                    )
                )

        spend_value = record.get("total_spend_gbp")
        export_matches = [
            {
                "desnz_id": match.get("desnz_id"),
                "similarity": match["score"],
                "confidence": match["confidence"],
                "uniqueness": match["uniqueness"],
                "uncertainty": match["uncertainty"],
                "emission_ratio": match.get("matched_emission", {}).get("GHG Conversion Factor 2025"),
                "total_spend_gbp": spend_value,
            }
            for match in retained_matches
        ]

        purchase_entry = {
            "purchase_id": purchase_identifier,
            "matches": [dict(m) for m in export_matches],
            "is_top3": bool(retained_matches and needs_top3),
            "is_top1": bool(retained_matches and not needs_top3),
            "is_ambiguous": bool(retained_matches and ambiguous),
            "no_match": not bool(retained_matches),
        }

        if missing_for_purchase and retained_matches:
            missing_entries.append(purchase_entry)
        if purchase_entry["is_ambiguous"]:
            top_matches_for_ambiguity = [
                {
                    "rank": match["rank"],
                    "score": match["score"],
                    "confidence": match["confidence"],
                    "uniqueness": match["uniqueness"],
                    "uncertainty": match["uncertainty"],
                    "cosine_similarity": match.get("cosine_similarity"),
                    "desnz_id": match.get("desnz_id"),
                    "matched_emission": match["matched_emission"],
                }
                for match in matches[: min(3, len(matches))]
            ]
            ambiguous_entries.append(
                {
                    "purchase_id": purchase_identifier,
                    "purchase_record": record,
                    "top_matches": top_matches_for_ambiguity,
                }
            )
            ambiguous_ids.append(purchase_identifier)
        if purchase_entry["is_top3"]:
            top3_entries.append(purchase_entry)
        elif purchase_entry["is_top1"]:
            top1_entries.append(purchase_entry)
        if purchase_entry["no_match"]:
            unmatched_ids.append(purchase_identifier)

        all_matches.append(purchase_entry)
        selected_results.append(purchase_entry)

    # ----------------------------------------------------------------------------------------------------------
    # Final Output
    # ----------------------------------------------------------------------------------------------------------
    # Write out the selected matches, detailed matches, and various categories of interest for analyst review

    with output_all_path.open("w", encoding="utf-8") as handle:
        json.dump(all_matches, handle, indent=2, ensure_ascii=False)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected_results, handle, indent=2, ensure_ascii=False)

    with missing_output_path.open("w", encoding="utf-8") as handle:
        json.dump(missing_entries, handle, indent=2, ensure_ascii=False)

    with ambiguous_output_path.open("w", encoding="utf-8") as handle:
        json.dump(ambiguous_entries, handle, indent=2, ensure_ascii=False)

    with top1_output_path.open("w", encoding="utf-8") as handle:
        json.dump(top1_entries, handle, indent=2, ensure_ascii=False)

    with top3_output_path.open("w", encoding="utf-8") as handle:
        json.dump(top3_entries, handle, indent=2, ensure_ascii=False)

    unmatched_count = len(unmatched_ids)
    total_purchases = len(purchase_records)
    summary_payload = {
        "total_purchases": total_purchases,
        "ambiguous_count": len(ambiguous_ids),
        "ambiguous_purchase_ids": ambiguous_ids,
        "unmatched_count": unmatched_count,
        "unmatched_percentage": (unmatched_count / total_purchases * 100.0) if total_purchases else 0.0,
        "top1_count": len(top1_entries),
        "top3_count": len(top3_entries),
    }
    with summary_output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False)

    if log_path:
        with log_path.open("w", encoding="utf-8") as log_handle:
            for line in uncertain_logs:
                log_handle.write(line + "\n")
            for line in missing_emission_logs:
                log_handle.write(line + "\n")
        print(f"Log written to {log_path}")
    else:
        for line in uncertain_logs:
            print(line)
        for line in missing_emission_logs:
            print(line)

    print(f"Summary: {len(uncertain_logs)} notices logged; {len(missing_emission_logs)} matches missing emission data.")
    print(
        "Category breakdown — ambiguous: {0}, missing_emissions: {1}, top1: {2}, top3: {3}".format(
            len(ambiguous_entries), len(missing_entries), len(top1_entries), len(top3_entries)
        )
    )
    print(
        "Unmatched purchases: {0} ({1:.1f}% of total)".format(
            unmatched_count, summary_payload["unmatched_percentage"]
        )
    )
    print(f"Top 1 matches retained: {len(top1_entries)}")
    print(f"Top 3 matches retained: {len(top3_entries)}")
    print(
        "Matching complete. Selected matches written to {0}, detailed matches to {1}".format(
            output_path, output_all_path
        )
    )
    if ambiguous_ids:
        formatted_ids = ", ".join(str(pid) for pid in ambiguous_ids)
        print(f"Ambiguous purchases: {formatted_ids}")
    else:
        print("Ambiguous purchases: none")


if __name__ == "__main__":
    main()
