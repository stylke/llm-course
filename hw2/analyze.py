"""Check saved results and record model probes and a few classification errors."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from gensim.models import FastText
from gensim.models.fasttext_inner import compute_ngrams_bytes, ft_hash_bytes
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from prepare import read_jsonl, write_json
from run_experiment import average_vectors, select_classifier, COMMON, SEED

ROOT = Path(__file__).resolve().parent


def verify_predictions(metrics, predictions):
    checked = 0
    for name, tasks in metrics.items():
        for task, record in tasks.items():
            rows = predictions[name][task]
            truth = [r["true"] for r in rows]
            predicted = [r["predicted"] for r in rows]
            assert np.isclose(accuracy_score(truth, predicted), record["test_accuracy"])
            assert np.isclose(f1_score(truth, predicted, labels=record["labels"],
                                      average="macro", zero_division=0),
                              record["test_macro_f1"])
            assert confusion_matrix(truth, predicted, labels=record["labels"]).tolist() == \
                record["confusion_matrix"]
            checked += 1
    return checked


def analyze():
    load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
    metrics = load("metrics.json")
    predictions = load("predictions.json")
    supervised = load("supervised_metrics.json")
    supervised_predictions = load("supervised_predictions.json")
    checked = verify_predictions(metrics, predictions)
    checked += verify_predictions(supervised, supervised_predictions)
    selection = load("selection.json")
    for task, record in selection.items():
        expected = max(metrics, key=lambda name: metrics[name][task]["valid_macro_f1"])
        assert record["selected_by_validation"] == expected
        assert record["test_macro_f1"] == metrics[expected][task]["test_macro_f1"]
    rows = read_jsonl(ROOT / "data/january.jsonl")
    manifest = load("split_manifest.json")
    assert len({r["id"] for r in manifest}) == len(rows)
    assert len({r["sha256"] for r in manifest}) == len(rows)
    assert [r["id"] for r in rows] == [r["id"] for r in manifest]
    dates = {s: {r["date"] for r in rows if r["split"] == s}
             for s in ("train", "valid", "test")}
    assert max(dates["train"]) < min(dates["valid"]) < min(dates["test"])

    model = FastText.load(str(ROOT / "models/ft_adapted.gensim"))
    for array in (model.wv.vectors, model.wv.vectors_vocab, model.wv.vectors_ngrams,
                  model.syn1neg):
        assert np.isfinite(array).all()
    test_rows = [r for r in rows if r["split"] == "test"]
    idf = json.loads((ROOT / "models/idf.json").read_text(encoding="utf-8"))
    features, _ = average_vectors(test_rows, model.wv, idf)
    clf = joblib.load(ROOT / "models/ft_adapted_idf_4class.joblib")
    reloaded_prediction = clf.predict(features)
    assert reloaded_prediction.tolist() == [
        r["predicted"] for r in predictions["ft_adapted_idf"]["4class"]]
    assert np.isfinite(clf.coef_).all()

    ngrams = set()
    for word in model.wv.index_to_key:
        ngrams.update(compute_ngrams_bytes(word, model.wv.min_n, model.wv.max_n))
    buckets = {ft_hash_bytes(ngram) % model.wv.bucket for ngram in ngrams}
    collisions = {
        "unique_ngram_strings": len(ngrams), "occupied_buckets": len(buckets),
        "bucket_count": model.wv.bucket,
        "extra_strings_sharing_buckets": len(ngrams) - len(buckets),
        "note": "A collision count is not a direct measure of semantic damage.",
    }
    subwords = {word: [ngram.decode("utf-8") for ngram in
                      compute_ngrams_bytes(word, 1, 3)]
                for word in ("足球", "互联网")}
    groups = {}
    for words in (["足球", "篮球", "排球"], ["银行", "金融", "贷款"]):
        if all(w in model.wv.key_to_index for w in words):
            groups["+".join(words)] = [[w, float(s)] for w, s in
                                       model.wv.most_similar(positive=words, topn=5)]
    analogies = {}
    for positive, negative in ((["美国", "北京"], ["中国"]),
                               (["篮球", "足球"], ["比赛"])):
        if all(w in model.wv.key_to_index for w in positive + negative):
            name = "+".join(positive) + "-" + "-".join(negative)
            analogies[name] = [[w, float(s)] for w, s in
                              model.wv.most_similar(positive=positive, negative=negative, topn=5)]
    errors = {}
    for name in ("tfidf", "ft_adapted_idf"):
        incorrect = [r for r in predictions[name]["4class"] if r["true"] != r["predicted"]]
        errors[name] = {
            "count": len(incorrect),
            "first_six_in_date_order": incorrect[:6],
            "pairs": dict(Counter(f"{r['true']}->{r['predicted']}" for r in incorrect)),
        }
    check = {
        "prediction_tables_checked": checked, "article_ids_unique": True,
        "article_hashes_unique": True, "date_splits_disjoint": True,
        "model_arrays_finite": True, "reloaded_predictions_match": True,
        "test_document_vectors_finite": True, "hash_collisions": collisions,
        "validation_selection_matches": True,
        "subword_examples": subwords, "group_neighbors": groups, "analogy_trials": analogies,
        "classification_errors": errors,
    }
    write_json(ROOT / "results/analysis.json", check)
    print(json.dumps(check, ensure_ascii=False, indent=2), flush=True)


def repeat_seeds():
    """Two extra seeds at fixed FastText settings; no test-set tuning."""
    rows = read_jsonl(ROOT / "data/january.jsonl")
    external = [r["tokens"] for r in read_jsonl(ROOT / "data/external_articles.jsonl")]
    sentences = external + [s for r in rows if r["split"] == "train" for s in r["sentences"]]
    from prepare import stable_hash
    baseline = json.loads((ROOT / "results/metrics.json").read_text())["ft_adapted"]["4class"]
    records = [{"seed": SEED, "C": baseline["C"],
                "valid_macro_f1": baseline["valid_macro_f1"],
                "test_macro_f1": baseline["test_macro_f1"],
                "test_accuracy": baseline["test_accuracy"]}]
    for seed in (7, 42):
        parameters = {**COMMON, "seed": seed, "sg": 1, "vector_size": 100,
                      "window": 5, "min_n": 1, "max_n": 3, "bucket": 50000}
        started = time.perf_counter()
        model = FastText(sentences, hashfxn=stable_hash, **parameters)
        features, _ = average_vectors(rows, model.wv)
        result, _, _ = select_classifier(
            features, rows, ["international", "sports", "economy", "education"])
        record = {"seed": seed, "C": result["C"], "valid_macro_f1": result["valid_macro_f1"],
                  "test_macro_f1": result["test_macro_f1"],
                  "test_accuracy": result["test_accuracy"],
                  "training_and_evaluation_seconds": time.perf_counter() - started}
        records.append(record)
        print(json.dumps(record), flush=True)
    values = [r["test_macro_f1"] for r in records]
    write_json(ROOT / "results/seed_repeats.json", {
        "configuration": "ft_adapted, uniform mean, 4class",
        "runs": records, "mean_test_macro_f1": float(np.mean(values)),
        "sample_std_test_macro_f1": float(np.std(values, ddof=1)),
        "note": "Descriptive seed variation, not a confidence interval; no seed selected on test.",
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-seeds", action="store_true")
    args = parser.parse_args()
    analyze()
    if args.repeat_seeds:
        repeat_seeds()
