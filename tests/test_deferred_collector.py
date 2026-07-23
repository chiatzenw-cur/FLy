import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "fly" / "models" / "deferred_collector.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "deferred_collector_under_test", MODULE_PATH
)
deferred_collector = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(deferred_collector)

INSPECT_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "inspect_raw_mask.py"
)
INSPECT_MODULE_SPEC = importlib.util.spec_from_file_location(
    "inspect_raw_mask_under_test", INSPECT_MODULE_PATH
)
inspect_raw_mask = importlib.util.module_from_spec(INSPECT_MODULE_SPEC)
INSPECT_MODULE_SPEC.loader.exec_module(inspect_raw_mask)

POSITION_FEATURE_KEYS = deferred_collector.POSITION_FEATURE_KEYS
RAW_MASK_SCHEMA_VERSION = deferred_collector.RAW_MASK_SCHEMA_VERSION
RawMaskJSONLWriter = deferred_collector.RawMaskJSONLWriter
build_raw_mask_record = deferred_collector.build_raw_mask_record
compute_position_features = deferred_collector.compute_position_features
iter_raw_mask_records = deferred_collector.iter_raw_mask_records
validate_raw_mask_record = deferred_collector.validate_raw_mask_record
teacher_forced_response_logit_bounds = (
    deferred_collector.teacher_forced_response_logit_bounds
)
resolve_aligned_vocab_size = deferred_collector.resolve_aligned_vocab_size


def _features(length):
    return {key: [0.1] * length for key in POSITION_FEATURE_KEYS}


class DeferredCollectorTests(unittest.TestCase):
    def test_teacher_forced_response_logit_bounds(self):
        self.assertEqual(teacher_forced_response_logit_bounds(10, 4), (9, 13))
        self.assertEqual(teacher_forced_response_logit_bounds(1, 1), (0, 1))

    def test_build_raw_mask_record_keeps_only_raw_observations(self):
        record = build_raw_mask_record(
            sample_id=7,
            prompt_token_ids=[1, 2],
            target_response_token_ids=[3, 4, 5, 6],
            draft_top1_token_ids=[3, 9, 5, 8],
            target_top1_token_ids=[3, 4, 5, 6],
            match_mask=[True, False, True, False],
            position_features=_features(4),
            max_future_window=64,
            target_logits_vocab_size=16,
            draft_logits_vocab_size=12,
            tokenizer_vocab_size=10,
            aligned_vocab_size=10,
        )

        self.assertEqual(record["schema_version"], RAW_MASK_SCHEMA_VERSION)
        self.assertEqual(record["mismatch_positions"], [1, 3])
        self.assertEqual(record["num_mismatches"], 2)
        self.assertNotIn("label", record)
        self.assertFalse(any("survive" in key for key in record))

    def test_record_validation_rejects_inconsistent_mask(self):
        record = build_raw_mask_record(
            sample_id=0,
            prompt_token_ids=[1],
            target_response_token_ids=[2, 3],
            draft_top1_token_ids=[2, 9],
            target_top1_token_ids=[2, 3],
            match_mask=[True, False],
            position_features=_features(2),
            max_future_window=64,
            target_logits_vocab_size=16,
            draft_logits_vocab_size=12,
            tokenizer_vocab_size=10,
            aligned_vocab_size=10,
        )
        record["match_mask"][1] = True

        with self.assertRaisesRegex(ValueError, "match_mask"):
            validate_raw_mask_record(record)

    @unittest.skipIf(torch is None, "torch is unavailable")
    def test_position_features_are_aligned_and_finite(self):
        target_logits = torch.tensor(
            [[[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]]], dtype=torch.float32
        )
        draft_logits = target_logits.clone()
        draft_ids = torch.tensor([[0, 1]], dtype=torch.long)

        features = compute_position_features(
            target_logits=target_logits,
            draft_logits=draft_logits,
            draft_token_ids=draft_ids,
            chunk_size=1,
        )

        self.assertEqual(set(features), set(POSITION_FEATURE_KEYS))
        self.assertTrue(all(len(values) == 2 for values in features.values()))
        for value in features["kl_target_to_draft"]:
            self.assertAlmostEqual(value, 0.0, places=6)
        self.assertTrue(
            all(value > 0 for value in features["target_prob_of_draft"])
        )

    def test_jsonl_writer_round_trip(self):
        record = build_raw_mask_record(
            sample_id=1,
            prompt_token_ids=[1],
            target_response_token_ids=[2],
            draft_top1_token_ids=[2],
            target_top1_token_ids=[2],
            match_mask=[True],
            position_features=_features(1),
            max_future_window=64,
            target_logits_vocab_size=16,
            draft_logits_vocab_size=12,
            tokenizer_vocab_size=10,
            aligned_vocab_size=10,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "raw.jsonl"
            writer = RawMaskJSONLWriter(str(output_path))
            writer.write(record)
            writer.close()

            loaded = list(iter_raw_mask_records(str(output_path)))
            self.assertEqual(loaded, [record])
            parsed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["sample_id"], 1)

    def test_inspector_reports_future_window_censoring(self):
        record = build_raw_mask_record(
            sample_id=2,
            prompt_token_ids=[1],
            target_response_token_ids=[2, 3, 4],
            draft_top1_token_ids=[2, 9, 4],
            target_top1_token_ids=[2, 3, 4],
            match_mask=[True, False, True],
            position_features=_features(3),
            max_future_window=64,
            target_logits_vocab_size=16,
            draft_logits_vocab_size=12,
            tokenizer_vocab_size=10,
            aligned_vocab_size=10,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "raw.jsonl"
            writer = RawMaskJSONLWriter(str(output_path))
            writer.write(record)
            writer.close()

            summary = inspect_raw_mask.summarize(str(output_path))

        self.assertEqual(summary["num_samples"], 1)
        self.assertEqual(summary["total_mismatches"], 1)
        coverage = summary["future_window_coverage"]["64"]
        self.assertEqual(coverage["eligible"], 0)
        self.assertEqual(coverage["right_censored"], 1)

    def test_resolve_aligned_vocab_size_uses_shared_tokenizer_prefix(self):
        self.assertEqual(resolve_aligned_vocab_size(152064, 151936, 151665), 151665)
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_aligned_vocab_size(10, 0, 8)


if __name__ == "__main__":
    unittest.main()
