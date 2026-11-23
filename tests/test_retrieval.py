import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from label_lookup.answering import Lookup
from label_lookup.corpus import Record, split_families
from label_lookup.evaluation import _metrics
from label_lookup.images import load_image, unit_pixels
from label_lookup.retrieval import BM25, reciprocal_rank_fusion


class ResearchChecks(unittest.TestCase):
    def test_reciprocal_rank_fusion_matches_hand_calculation(self):
        scores = reciprocal_rank_fusion([[11, 15], [15, 22]])
        self.assertAlmostEqual(scores[15], 1 / 62 + 1 / 61)
        self.assertEqual(sorted(scores, key=scores.get, reverse=True), [15, 11, 22])

    def test_bm25_ranks_specific_product_tokens(self):
        index = BM25(["two packages blue", "one carton green", "one bottle red"])
        self.assertEqual(int(np.argmax(index.score("green carton"))), 1)

    def test_splits_keep_product_families_together(self):
        records = tuple(
            Record(str(i), f"family-{i // 2}", "name", ("ndc",), ("box",), "", "", "", "", 1)
            for i in range(30)
        )
        splits = split_families(records)
        self.assertEqual(splits, split_families(tuple(reversed(records))))
        listed = [family for families in splits.values() for family in families]
        self.assertEqual(len(listed), 15)
        self.assertEqual(len(listed), len(set(listed)))

    def test_calculated_metrics(self):
        self.assertEqual(_metrics([1, 2, 4]), {
            "n": 3, "recall_at_1": 0.3333, "recall_at_3": 0.6667, "mrr": 0.5833,
        })

    def test_image_normalization(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example.png"
            Image.new("L", (10, 12), color=255).save(path)
            array = unit_pixels(load_image(path))
            self.assertEqual(array.shape, (12, 10, 3))
            self.assertEqual(array.dtype, np.float32)
            self.assertTrue((array == 1).all())

    def test_image_decoder_size_error_is_reported_as_invalid_input(self):
        with patch("label_lookup.images.Image.open", side_effect=Image.DecompressionBombError):
            with self.assertRaisesRegex(ValueError, "12-megapixel"):
                load_image(b"an image")

    def test_question_routing_reports_ambiguity(self):
        update = Lookup._classify({"question": "What are the product name and package codes?"})
        self.assertEqual(update["result"]["status"], "clarify")
        self.assertEqual(update["result"]["citations"], [])

    def test_unsupported_question_has_no_citation(self):
        update = Lookup._classify({"question": "Does it have a recycling symbol?"})
        self.assertEqual(update["result"]["status"], "abstain")
        self.assertEqual(update["result"]["citations"], [])


if __name__ == "__main__":
    unittest.main()
