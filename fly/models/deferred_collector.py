from __future__ import annotations

import atexit
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # Allows schema inspection without the ML runtime.
    torch = None
    F = None


RAW_MASK_SCHEMA_VERSION = "target_teacher_forced_raw_mask_v1"

POSITION_FEATURE_KEYS = (
    "kl_target_to_draft",
    "target_entropy",
    "draft_entropy",
    "target_entropy_top3",
    "target_prob_of_draft",
    "draft_prob_of_draft",
    "target_top1_probability",
    "target_top1_top2_logit_margin",
)


def teacher_forced_response_logit_bounds(
    prompt_length: int, response_length: int
):
    """Return the causal-logit slice that predicts the response tokens."""
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    if response_length < 0:
        raise ValueError("response_length cannot be negative")
    start = prompt_length - 1
    return start, start + response_length


def compute_position_features(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    chunk_size: int = 8,
) -> Dict[str, List[float]]:
    """Compute aligned scalar features without retaining vocabulary logits.

    All tensors describe predictions at the same teacher-forced target prefix.
    The result contains one scalar per response position.
    """
    if torch is None or F is None:
        raise RuntimeError("torch is required to compute raw-mask position features")
    if target_logits.ndim != 3 or draft_logits.ndim != 3:
        raise ValueError("target_logits and draft_logits must have shape [1, T, V]")
    if draft_token_ids.ndim != 2:
        raise ValueError("draft_token_ids must have shape [1, T]")
    if target_logits.shape[0] != 1 or draft_logits.shape[0] != 1:
        raise ValueError("raw-mask collection currently supports batch size 1 only")
    if target_logits.shape[:2] != draft_logits.shape[:2]:
        raise ValueError(
            "target and draft logits must have the same batch and sequence dimensions"
        )
    if target_logits.shape[-1] != draft_logits.shape[-1]:
        raise ValueError(
            "target and draft vocabularies must be aligned to compute KL divergence"
        )
    if draft_token_ids.shape != target_logits.shape[:2]:
        raise ValueError("draft_token_ids must align with the logits sequence")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sequence_length = target_logits.shape[1]
    feature_chunks = {key: [] for key in POSITION_FEATURE_KEYS}

    for start in range(0, sequence_length, chunk_size):
        end = min(start + chunk_size, sequence_length)
        device = target_logits.device

        target_chunk = target_logits[:, start:end, :].float()
        draft_chunk = draft_logits[:, start:end, :].to(
            device=device, dtype=torch.float32
        )
        draft_ids_chunk = draft_token_ids[:, start:end].to(device=device)

        target_log_probs = F.log_softmax(target_chunk, dim=-1)
        draft_log_probs = F.log_softmax(draft_chunk, dim=-1)
        target_probs = target_log_probs.exp()
        draft_probs = draft_log_probs.exp()

        kl_target_to_draft = torch.sum(
            target_probs * (target_log_probs - draft_log_probs), dim=-1
        )
        target_entropy = -torch.sum(target_probs * target_log_probs, dim=-1)
        draft_entropy = -torch.sum(draft_probs * draft_log_probs, dim=-1)

        top3_log_probs = torch.topk(
            target_log_probs, min(3, target_log_probs.shape[-1]), dim=-1
        ).values
        target_entropy_top3 = -torch.sum(
            top3_log_probs.exp() * top3_log_probs, dim=-1
        )

        target_prob_of_draft = target_probs.gather(
            -1, draft_ids_chunk.unsqueeze(-1)
        ).squeeze(-1)
        draft_prob_of_draft = draft_probs.gather(
            -1, draft_ids_chunk.unsqueeze(-1)
        ).squeeze(-1)

        target_top_values = torch.topk(
            target_chunk, min(2, target_chunk.shape[-1]), dim=-1
        ).values
        target_top1_probability = target_probs.max(dim=-1).values
        if target_top_values.shape[-1] == 1:
            target_margin = torch.zeros_like(target_top_values[..., 0])
        else:
            target_margin = target_top_values[..., 0] - target_top_values[..., 1]

        chunk_values = {
            "kl_target_to_draft": kl_target_to_draft,
            "target_entropy": target_entropy,
            "draft_entropy": draft_entropy,
            "target_entropy_top3": target_entropy_top3,
            "target_prob_of_draft": target_prob_of_draft,
            "draft_prob_of_draft": draft_prob_of_draft,
            "target_top1_probability": target_top1_probability,
            "target_top1_top2_logit_margin": target_margin,
        }
        stacked_features = torch.stack(
            [chunk_values[key] for key in POSITION_FEATURE_KEYS], dim=-1
        )
        stacked_features = stacked_features.detach().float().cpu()
        if not torch.isfinite(stacked_features).all():
            raise ValueError("position features contain NaN or infinite values")
        for feature_index, key in enumerate(POSITION_FEATURE_KEYS):
            feature_chunks[key].extend(
                float(value)
                for value in stacked_features[0, :, feature_index].tolist()
            )

    if sequence_length:
        minimum_kl = min(feature_chunks["kl_target_to_draft"])
        if minimum_kl < -1e-3:
            raise ValueError(
                f"KL(target || draft) is unexpectedly negative: {minimum_kl}"
            )

    return feature_chunks


def build_raw_mask_record(
    *,
    sample_id: int,
    prompt_token_ids: Sequence[int],
    target_response_token_ids: Sequence[int],
    draft_top1_token_ids: Sequence[int],
    target_top1_token_ids: Sequence[int],
    match_mask: Sequence[bool],
    position_features: Dict[str, Sequence[float]],
    max_future_window: int,
) -> Dict:
    """Build one label-free record for one target-generated response."""
    if max_future_window <= 0:
        raise ValueError("max_future_window must be positive")

    response_length = len(target_response_token_ids)
    aligned_fields = {
        "draft_top1_token_ids": draft_top1_token_ids,
        "target_top1_token_ids": target_top1_token_ids,
        "match_mask": match_mask,
    }
    aligned_fields.update(position_features)

    for name, values in aligned_fields.items():
        if len(values) != response_length:
            raise ValueError(
                f"{name} has length {len(values)}, expected {response_length}"
            )

    normalized_match_mask = [bool(value) for value in match_mask]
    normalized_prompt_ids = [int(value) for value in prompt_token_ids]
    mismatch_positions = [
        position
        for position, is_match in enumerate(normalized_match_mask)
        if not is_match
    ]

    record = {
        "schema_version": RAW_MASK_SCHEMA_VERSION,
        "sample_id": int(sample_id),
        "prompt_token_sha256": hashlib.sha256(
            ",".join(str(value) for value in normalized_prompt_ids).encode("utf-8")
        ).hexdigest(),
        "max_future_window": int(max_future_window),
        "prompt_length": len(normalized_prompt_ids),
        "response_length": response_length,
        "prompt_token_ids": normalized_prompt_ids,
        "target_response_token_ids": [
            int(value) for value in target_response_token_ids
        ],
        "draft_top1_token_ids": [int(value) for value in draft_top1_token_ids],
        "target_top1_token_ids": [int(value) for value in target_top1_token_ids],
        "match_mask": normalized_match_mask,
        "mismatch_positions": mismatch_positions,
        "num_mismatches": len(mismatch_positions),
    }
    for key in POSITION_FEATURE_KEYS:
        if key not in position_features:
            raise ValueError(f"missing position feature: {key}")
        record[key] = [float(value) for value in position_features[key]]

    validate_raw_mask_record(record)
    return record


def validate_raw_mask_record(record: Dict) -> None:
    if record.get("schema_version") != RAW_MASK_SCHEMA_VERSION:
        raise ValueError("unexpected raw-mask schema version")

    response_length = int(record["response_length"])
    if int(record["prompt_length"]) != len(record["prompt_token_ids"]):
        raise ValueError("prompt_length is inconsistent with prompt_token_ids")
    expected_prompt_hash = hashlib.sha256(
        ",".join(str(int(value)) for value in record["prompt_token_ids"]).encode(
            "utf-8"
        )
    ).hexdigest()
    if record["prompt_token_sha256"] != expected_prompt_hash:
        raise ValueError("prompt_token_sha256 is inconsistent with prompt_token_ids")
    aligned_keys = (
        "target_response_token_ids",
        "draft_top1_token_ids",
        "target_top1_token_ids",
        "match_mask",
        *POSITION_FEATURE_KEYS,
    )
    for key in aligned_keys:
        if len(record[key]) != response_length:
            raise ValueError(
                f"{key} has length {len(record[key])}, expected {response_length}"
            )

    recomputed_mask = [
        int(draft_id) == int(target_id)
        for draft_id, target_id in zip(
            record["draft_top1_token_ids"],
            record["target_response_token_ids"],
        )
    ]
    if recomputed_mask != [bool(value) for value in record["match_mask"]]:
        raise ValueError("match_mask is inconsistent with draft and target token ids")

    recomputed_mismatches = [
        position
        for position, is_match in enumerate(recomputed_mask)
        if not is_match
    ]
    if recomputed_mismatches != record["mismatch_positions"]:
        raise ValueError("mismatch_positions is inconsistent with match_mask")
    if int(record["num_mismatches"]) != len(recomputed_mismatches):
        raise ValueError("num_mismatches is inconsistent with match_mask")

    for key in POSITION_FEATURE_KEYS:
        if not all(math.isfinite(float(value)) for value in record[key]):
            raise ValueError(f"{key} contains NaN or infinite values")


class RawMaskJSONLWriter:
    """Buffered JSONL writer with one complete response per line."""

    def __init__(self, output_path: str):
        if not output_path:
            raise ValueError("raw_mask_output_path is required")

        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._closed = False
        atexit.register(self.close)

    def write(self, record: Dict) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed raw-mask writer")
        validate_raw_mask_record(record)
        json.dump(record, self._handle, ensure_ascii=False, separators=(",", ":"))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.flush()
            self._handle.close()
            self._closed = True


def iter_raw_mask_records(path: str) -> Iterable[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                validate_raw_mask_record(record)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid record on line {line_number}: {error}") from error
            yield record
