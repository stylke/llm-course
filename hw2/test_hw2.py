"""Small regression tests for preprocessing, splitting and representation."""
import unittest
from pathlib import Path

from gensim.models import FastText, Word2Vec
import numpy as np

from prepare import PICTURE, clean_token, january_articles, read_jsonl, token_sentences
from run_experiment import average_vectors, stable_hash

ROOT = Path(__file__).resolve().parent


class PreprocessTests(unittest.TestCase):
    def test_normalization_and_pronunciation(self):
        self.assertEqual(clean_token("上{shang5}"), "上")
        self.assertEqual(clean_token("ＡＢＣ１２"), "abc12")
        self.assertEqual(clean_token("。"), "")

    def test_sentence_boundary(self):
        words = ["学校", "举行", "比赛", "。", "国际", "新闻", "〓", "！"]
        self.assertEqual(list(token_sentences(words)),
                         [["学校", "举行", "比赛"], ["国际", "新闻"]])

    def test_segmented_picture_annotation(self):
        self.assertEqual(PICTURE.sub("", "冠军 （ 附 图片 １ 张 ）"), "冠军 ")
        self.assertEqual(PICTURE.sub("", "联赛 （ 图片 ）"), "联赛 ")

    def test_split_disjointness(self):
        rows = read_jsonl(ROOT / "data/january.jsonl")
        ids = {s: {r["id"] for r in rows if r["split"] == s}
               for s in ("train", "valid", "test")}
        self.assertFalse(ids["train"] & ids["valid"])
        self.assertFalse(ids["train"] & ids["test"])
        self.assertFalse(ids["valid"] & ids["test"])
        self.assertEqual(len({r["sha256"] for r in rows}), len(rows))
        for row in rows:
            self.assertEqual(row["tokens"], [w for s in row["sentences"] for w in s])
            self.assertFalse(any("{" in w or "}" in w for w in row["tokens"]))
            self.assertFalse(any(row["id"] in w for w in row["tokens"]))

    def test_source_matches_saved_preprocessing(self):
        source = ROOT.parent / "人民日报语料/1998-上半年-带音/1998-01-2003版-带音.txt"
        rows, _ = january_articles(source)
        self.assertEqual(rows, read_jsonl(ROOT / "data/january.jsonl"))


class VectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentences = [["足球", "比赛", "球队"], ["篮球", "比赛", "球队"],
                         ["银行", "金融", "贷款"]] * 20

    def test_fasttext_oov_and_saved_vectors(self):
        model = FastText(self.sentences, min_count=1, vector_size=12, min_n=1, max_n=3,
                         bucket=100, workers=1, seed=2026, epochs=2, hashfxn=stable_hash)
        word = "足球俱乐部"
        self.assertNotIn(word, model.wv.key_to_index)
        self.assertEqual(model.wv[word].shape, (12,))
        self.assertTrue(np.isfinite(model.wv[word]).all())
        vector, stats = average_vectors([{"tokens": [word]}], model.wv)
        self.assertEqual(stats["represented"], 1)
        self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 1.0, places=5)

    def test_word2vec_all_oov_document(self):
        model = Word2Vec(self.sentences, min_count=1, vector_size=12, workers=1,
                         seed=2026, epochs=2, hashfxn=stable_hash)
        matrix, stats = average_vectors([{"tokens": ["完全未见新词"]}], model.wv)
        self.assertEqual(stats["zero_documents"], 1)
        self.assertTrue(np.array_equal(matrix, np.zeros((1, 12))))

    def test_hash_stability(self):
        self.assertEqual(stable_hash("足球"), stable_hash("足球"))
        self.assertNotEqual(stable_hash("足球"), stable_hash("篮球"))


if __name__ == "__main__":
    unittest.main()
