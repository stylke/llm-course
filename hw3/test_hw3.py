"""Regression tests for labels, alignment and long-window coverage."""
from __future__ import annotations

import unittest

import numpy as np

from inference import Decoder, merge_weights, plan_windows
from modeling import encode_texts, load_tokenizer, metrics_from_confusion
from prepare import ROOT, label_text, read_jsonl, render


class MockDecoder(Decoder):
    def __init__(self):
        import torch
        self.device = torch.device("cpu")
        self.calls = 0
        self.processed_characters = 0

    def probabilities(self, texts):
        self.calls += 1
        self.processed_characters += sum(map(len, texts))
        result = []
        for text in texts:
            matrix = np.zeros((len(text), 8), dtype=np.float32)
            for i, char in enumerate(text):
                matrix[i, 2 if char == "终" else 0] = 1
            result.append(matrix)
        return result


class DataTests(unittest.TestCase):
    def test_cleanup(self):
        text, labels = label_text("１９９８ 年 ， 我 读{du2} 书 。 （ 附 图片 １ 张 ）")
        self.assertEqual(text, "1998年我读书")
        self.assertEqual(render(text, labels), "1998年，我读书。")

    def test_removed_structural_marks(self):
        text, labels = label_text("“你好！”他说：《书》很好。")
        self.assertEqual(render(text, labels), "你好！他说：书很好。")

    def test_lexical_numbers(self):
        text, _ = label_text("增长３．５％，１９９８－１９９９年。")
        self.assertEqual(text, "增长3.5%1998-1999年")

    def test_repeated_marks(self):
        text, labels = label_text("好吗？！好！")
        self.assertEqual(render(text, labels), "好吗？好！")

    def test_split_integrity(self):
        rows = read_jsonl(ROOT / "data/articles.jsonl")
        ids = {s: {r["id"] for r in rows if r["split"] == s}
               for s in ("train", "valid", "test")}
        self.assertFalse(ids["train"] & ids["valid"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["valid"] & ids["test"])
        self.assertTrue(all(len(r["text"]) == len(r["labels"]) for r in rows))
        self.assertLess(max(r["date"] for r in rows if r["split"] == "train"),
                        min(r["date"] for r in rows if r["split"] == "valid"))

    def test_token_alignment(self):
        tokenizer = load_tokenizer()
        texts = ["中国1998年增长3.5%", "生僻字𠮷"]
        encoded, positions = encode_texts(tokenizer, texts)
        self.assertEqual([len(p) for p in positions], [len(text) for text in texts])
        self.assertLessEqual(encoded["input_ids"].shape[1], 512)

    def test_window_coverage(self):
        for length in (0, 1, 300, 400, 401, 512, 700, 701, 3001):
            windows = plan_windows([length])
            coverage = np.zeros(length, dtype=int)
            for window in windows:
                coverage[window.start:window.end] += 1
                self.assertLessEqual(window.end - window.start, 400)
            self.assertTrue(np.all(coverage > 0))
            if windows:
                self.assertEqual(windows[-1].end, length)

    def test_batch_invariance_and_character_preservation(self):
        texts = ["甲" * 299 + "终" + "乙" * 2101, "", "短文"]
        decoder = MockDecoder()
        one, _ = decoder.fixed(texts, batch_size=1)
        eight, _ = decoder.fixed(texts, batch_size=8)
        self.assertEqual(one, eight)
        for text, labels in zip(texts, eight):
            self.assertEqual(len(text), len(labels))
            self.assertEqual(render(text, labels).replace("。", ""), text)

    def test_sequential_progress_without_stops(self):
        decoder = MockDecoder()
        text = "甲" * 3300
        predictions, timing = decoder.sequential([text])
        self.assertEqual(predictions[0], [0] * len(text))
        self.assertLess(timing["model_calls"], 20)

    def test_center_weights_and_invalid_sizes(self):
        weights = merge_weights(400, "center")
        self.assertEqual(weights[0], 1)
        self.assertEqual(weights[-1], 1)
        self.assertEqual(weights[200], 51)
        for window_size, stride in ((0, 1), (400, 401), (401, 300)):
            with self.assertRaises(ValueError):
                plan_windows([2000], window_size, stride)

    def test_boundary_and_type_metrics(self):
        matrix = np.zeros((8, 8), dtype=int)
        matrix[0, 0], matrix[1, 2], matrix[2, 2] = 100, 1, 1
        result = metrics_from_confusion(matrix)
        self.assertEqual(result["boundary_f1"], 1)
        self.assertEqual(result["punctuation_micro_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
