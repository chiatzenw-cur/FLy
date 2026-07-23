#!/usr/bin/env python3

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable, List


COLLECTOR_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "fly" / "models" / "deferred_collector.py"
)
COLLECTOR_MODULE_SPEC = importlib.util.spec_from_file_location(
    "deferred_collector_for_inspection", COLLECTOR_MODULE_PATH
)
deferred_collector = importlib.util.module_from_spec(COLLECTOR_MODULE_SPEC)
COLLECTOR_MODULE_SPEC.loader.exec_module(deferred_collector)

POSITION_FEATURE_KEYS = deferred_collector.POSITION_FEATURE_KEYS
iter_raw_mask_records = deferred_collector.iter_raw_mask_records


def _quantile(values: List[float], probability: float):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _quantiles(values: Iterable[float]):
    materialized = [float(value) for value in values]
    return {
        "p10": _quantile(materialized, 0.10),
        "p25": _quantile(materialized, 0.25),
        "p50": _quantile(materialized, 0.50),
        "p75": _quantile(materialized, 0.75),
        "p90": _quantile(materialized, 0.90),
    }


def summarize(path: str):
    records = list(iter_raw_mask_records(path))
    total_positions = sum(record["response_length"] for record in records)
    total_mismatches = sum(record["num_mismatches"] for record in records)

    target_top1_disagreements = 0
    future_coverage = {}
    feature_values = {key: [] for key in POSITION_FEATURE_KEYS}
    mismatch_feature_values = {key: [] for key in POSITION_FEATURE_KEYS}

    for record in records:
        max_future = int(record["max_future_window"])
        coverage = future_coverage.setdefault(
            str(max_future), {"eligible": 0, "right_censored": 0}
        )

        response_length = record["response_length"]
        mismatch_set = set(record["mismatch_positions"])
        for position in mismatch_set:
            if response_length - position - 1 >= max_future:
                coverage["eligible"] += 1
            else:
                coverage["right_censored"] += 1

        target_top1_disagreements += sum(
            int(top1_id) != int(response_id)
            for top1_id, response_id in zip(
                record["target_top1_token_ids"],
                record["target_response_token_ids"],
            )
        )

        for key in POSITION_FEATURE_KEYS:
            values = [float(value) for value in record[key]]
            feature_values[key].extend(values)
            mismatch_feature_values[key].extend(
                value
                for position, value in enumerate(values)
                if position in mismatch_set
            )

    for coverage in future_coverage.values():
        total = coverage["eligible"] + coverage["right_censored"]
        coverage["eligible_rate"] = coverage["eligible"] / total if total else None
        coverage["right_censored_rate"] = (
            coverage["right_censored"] / total if total else None
        )

    return {
        "schema": "raw_mask_inspection_v1",
        "num_samples": len(records),
        "total_positions": total_positions,
        "total_mismatches": total_mismatches,
        "mismatch_rate": (
            total_mismatches / total_positions if total_positions else None
        ),
        "target_top1_response_disagreements": target_top1_disagreements,
        "future_window_coverage": future_coverage,
        "feature_quantiles_all_positions": {
            key: _quantiles(values) for key, values in feature_values.items()
        },
        "feature_quantiles_mismatch_positions": {
            key: _quantiles(values)
            for key, values in mismatch_feature_values.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and summarize target-teacher-forced raw mask JSONL."
    )
    parser.add_argument("input_path")
    parser.add_argument(
        "--output",
        help="Optional path for the JSON summary. The summary is always printed.",
    )
    args = parser.parse_args()

    summary = summarize(args.input_path)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")


if __name__ == "__main__":
    main()
