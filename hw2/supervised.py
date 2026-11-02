"""Official fastText classification, separate from Gensim word embeddings."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import fasttext
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from prepare import read_jsonl, write_json

ROOT = Path(__file__).resolve().parent
TASKS = {
    "2class": ["international", "sports"],
    "3class": ["international", "sports", "economy"],
    "4class": ["international", "sports", "economy", "education"],
}
COMMON = {"dim": 100, "epoch": 25, "minCount": 1, "minn": 0, "maxn": 0,
          "loss": "softmax", "bucket": 50000, "thread": 1, "seed": 2026, "verbose": 0}


def predict_rows(model, rows):
    texts = [" ".join(r["tokens"]) for r in rows]
    # The batch API also avoids fasttext 0.9.3's scalar np.array(copy=False) on NumPy 2.
    labels, probabilities = model.predict(texts, k=len(model.labels))
    prediction = [row[0].removeprefix("__label__") for row in labels]
    records = []
    for row, names, probs in zip(rows, labels, probabilities):
        records.append({
            "id": row["id"], "title": row["title"], "true": row["label"],
            "predicted": names[0].removeprefix("__label__"),
            "probabilities": {n.removeprefix("__label__"): float(p)
                              for n, p in zip(names, probs)},
        })
    return prediction, records


def run():
    rows = read_jsonl(ROOT / "data/january.jsonl")
    (ROOT / "data/supervised").mkdir(exist_ok=True)
    metrics, predictions = {}, {}
    for task, labels in TASKS.items():
        subset = {s: [r for r in rows if r["label"] in labels and r["split"] == s]
                  for s in ("train", "valid", "test")}
        training_rows = list(subset["train"])
        random.Random(2026).shuffle(training_rows)
        train_file = ROOT / "data/supervised" / f"{task}.train.txt"
        with train_file.open("w", encoding="utf-8") as handle:
            for row in training_rows:
                handle.write(f"__label__{row['label']} " + " ".join(row["tokens"]) + "\n")
        metrics[task], predictions[task] = {}, {}
        for word_ngrams in (1, 2):
            trials = []
            best = None
            started = time.perf_counter()
            for lr in (0.1, 0.5):
                model = fasttext.train_supervised(
                    input=str(train_file), lr=lr, wordNgrams=word_ngrams, **COMMON)
                pred, _ = predict_rows(model, subset["valid"])
                score = f1_score([r["label"] for r in subset["valid"]], pred,
                                 labels=labels, average="macro", zero_division=0)
                trials.append({"lr": lr, "valid_macro_f1": float(score)})
                if best is None or score > best[0]:
                    best = (score, lr, model)
            score, lr, model = best
            prediction, records = predict_rows(model, subset["test"])
            truth = [r["label"] for r in subset["test"]]
            name = f"wordNgrams{word_ngrams}"
            model_path = ROOT / "models" / f"official_{task}_{name}.bin"
            model.save_model(str(model_path))
            metrics[task][name] = {
                "parameters": {**COMMON, "lr": lr, "wordNgrams": word_ngrams},
                "valid_macro_f1": float(score), "validation_trials": trials,
                "test_accuracy": float(accuracy_score(truth, prediction)),
                "test_macro_f1": float(f1_score(truth, prediction, labels=labels,
                                               average="macro", zero_division=0)),
                "classification_report": classification_report(
                    truth, prediction, labels=labels, output_dict=True, zero_division=0),
                "confusion_matrix": confusion_matrix(truth, prediction, labels=labels).tolist(),
                "labels": labels, "seconds_for_two_trials": time.perf_counter() - started,
                "model_bytes": model_path.stat().st_size,
                "split_sizes": {s: len(v) for s, v in subset.items()},
            }
            predictions[task][name] = records
            print(f"official {task} {name}: valid F1={score:.4f}, "
                  f"test F1={metrics[task][name]['test_macro_f1']:.4f}", flush=True)
            write_json(ROOT / "results/supervised_metrics.json", metrics)
            write_json(ROOT / "results/supervised_predictions.json", predictions)
    write_json(ROOT / "results/supervised_environment.json", {
        "version": version("fasttext"), "parameters": COMMON, "lr_grid": [0.1, 0.5],
        "notes": "January training data only; embeddings and classifier jointly trained. "
                 "Softmax is used because there are only 2-4 labels. No class reweighting.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
