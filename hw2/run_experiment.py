"""Train representations, select classifiers on validation data, and evaluate."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import Counter
import csv
import gc
import json
import logging
import platform
import random
import time

import gensim
from gensim import corpora, matutils
from gensim.models import FastText, LsiModel, TfidfModel, Word2Vec
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from prepare import read_jsonl, stable_hash, write_json

SEED = 2026
TASKS = {
    "2class": ["international", "sports"],
    "3class": ["international", "sports", "economy"],
    "4class": ["international", "sports", "economy", "education"],
}
CONFIGS = {
    "w2v_cbow": {"type": "w2v", "sg": 0, "vector_size": 100, "window": 5,
                 "corpus": "external"},
    "w2v_sg": {"type": "w2v", "sg": 1, "vector_size": 100, "window": 5,
               "corpus": "external"},
    "ft_external": {"type": "ft", "sg": 1, "vector_size": 100, "window": 5,
                    "min_n": 1, "max_n": 3, "corpus": "external"},
    "ft_3_6": {"type": "ft", "sg": 1, "vector_size": 100, "window": 5,
               "min_n": 3, "max_n": 6, "corpus": "external"},
    "ft_adapted": {"type": "ft", "sg": 1, "vector_size": 100, "window": 5,
                   "min_n": 1, "max_n": 3, "corpus": "external+train"},
    "ft_dim50": {"type": "ft", "sg": 1, "vector_size": 50, "window": 5,
                 "min_n": 1, "max_n": 3, "corpus": "external+train"},
    "ft_window3": {"type": "ft", "sg": 1, "vector_size": 100, "window": 3,
                   "min_n": 1, "max_n": 3, "corpus": "external+train"},
}
COMMON = {"min_count": 2, "negative": 5, "hs": 0, "sample": 1e-3, "epochs": 5,
          "workers": 1, "seed": SEED, "alpha": 0.025, "min_alpha": 0.0001,
          "ns_exponent": 0.75, "shrink_windows": True}


def average_vectors(rows, wv, idf=None):
    matrix = np.zeros((len(rows), wv.vector_size), dtype=np.float32)
    coverage = Counter()
    cache = {}
    for index, row in enumerate(rows):
        total_weight = 0.0
        for word in row["tokens"]:
            coverage["tokens"] += 1
            coverage["in_vocab"] += int(word in wv.key_to_index)
            if word not in cache:
                try:
                    cache[word] = wv[word]
                except KeyError:
                    cache[word] = None
            vector = cache[word]
            if vector is None:
                continue
            coverage["represented"] += 1
            weight = 1.0 if idf is None else idf.get(word, 1.0)
            matrix[index] += vector * weight
            total_weight += weight
        if total_weight:
            matrix[index] /= total_weight
        else:
            coverage["zero_documents"] += 1
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite document vectors")
    return normalize(matrix), dict(coverage)


def select_classifier(features, rows, labels):
    indices = {
        split: [i for i, row in enumerate(rows)
                if row["split"] == split and row["label"] in labels]
        for split in ("train", "valid", "test")
    }
    y = {split: [rows[i]["label"] for i in ids] for split, ids in indices.items()}
    trials = []
    best = None
    started = time.perf_counter()
    for c in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                 solver="lbfgs", random_state=SEED)
        clf.fit(features[indices["train"]], y["train"])
        prediction = clf.predict(features[indices["valid"]])
        score = f1_score(y["valid"], prediction, labels=labels, average="macro",
                         zero_division=0)
        trials.append({"C": c, "valid_macro_f1": float(score),
                       "iterations": int(clf.n_iter_.max())})
        # Keep the first (smaller C) when validation scores tie.
        if best is None or score > best[0]:
            best = (score, c, clf)
    score, c, clf = best
    prediction = clf.predict(features[indices["test"]])
    record = {
        "C": c, "valid_macro_f1": float(score),
        "test_accuracy": float(accuracy_score(y["test"], prediction)),
        "test_macro_f1": float(f1_score(y["test"], prediction, labels=labels,
                                       average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y["test"], prediction, labels=labels).tolist(),
        "classification_report": classification_report(
            y["test"], prediction, labels=labels, output_dict=True, zero_division=0),
        "labels": labels, "split_sizes": {s: len(v) for s, v in indices.items()},
        "validation_trials": trials, "classifier_seconds": time.perf_counter() - started,
    }
    probabilities = clf.predict_proba(features[indices["test"]])
    predictions = []
    for index, predicted, probs in zip(indices["test"], prediction, probabilities):
        row = rows[index]
        predictions.append({
            "id": row["id"], "title": row["title"], "true": row["label"],
            "predicted": str(predicted),
            "probabilities": {str(k): float(v) for k, v in zip(clf.classes_, probs)},
        })
    return record, clf, predictions


def semantic_probes(model):
    wv = model.wv
    queries = ("足球", "篮球", "经济", "银行", "教育", "科学", "中国", "美国")
    neighbors = {}
    for word in queries:
        if word in wv.key_to_index:
            neighbors[word] = [[w, float(s)] for w, s in wv.most_similar(word, topn=5)]
    pairs = []
    for a, b in (("足球", "篮球"), ("银行", "金融"), ("学校", "教育"),
                 ("足球", "银行"), ("美国", "中国")):
        if a in wv.key_to_index and b in wv.key_to_index:
            pairs.append({"a": a, "b": b, "cosine": float(wv.similarity(a, b))})
    oov = []
    for word in ("量子足球俱乐部", "互联网助学平台", "智能农业机器人"):
        if word in wv.key_to_index:
            raise ValueError(f"Probe unexpectedly in vocabulary: {word}")
        try:
            vector = wv[word]
            result = {"word": word, "in_vocab": False, "vector_available": True,
                      "norm": float(np.linalg.norm(vector)),
                      "neighbors": [[w, float(s)] for w, s in
                                    wv.similar_by_vector(vector, topn=3)]}
        except KeyError:
            result = {"word": word, "in_vocab": False, "vector_available": False}
        oov.append(result)
    return {"neighbors": neighbors, "pairs": pairs, "oov": oov}


def run(skip_training=False):
    random.seed(SEED)
    np.random.seed(SEED)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=ROOT / "results/logs/gensim.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = read_jsonl(ROOT / "data/january.jsonl")
    external = [r["tokens"] for r in read_jsonl(ROOT / "data/external_articles.jsonl")]
    training_rows = [r for r in rows if r["split"] == "train"]
    adapted = external + [s for r in training_rows for s in r["sentences"]]
    docs = [r["tokens"] for r in training_rows]
    idf_counts = Counter(w for doc in docs for w in set(doc))
    idf = {w: float(np.log((len(docs) + 1) / (count + 1)) + 1)
           for w, count in idf_counts.items()}
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "versions": {"gensim": gensim.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "seed": SEED, "common_embedding_parameters": COMMON,
        "classifier_C_grid": [0.1, 1.0, 10.0], "classifier_class_weight": "balanced",
        "external_sentences": len(external), "adapted_sentences": len(adapted),
        "external_tokens": sum(map(len, external)), "adapted_tokens": sum(map(len, adapted)),
        "hash_function": "first 8 bytes of SHA256, little endian",
        "fit_scope": "TF-IDF, LSI, IDF and classifiers: January train only; embeddings as named",
    }
    write_json(ROOT / "results/environment.json", environment)
    all_metrics = {}
    all_predictions = {}
    training = {}
    probes = {}
    coverage = {}
    csv_rows = []
    previous_training = {}
    if skip_training:
        previous_training = json.loads((ROOT / "results/training.json").read_text(encoding="utf-8"))

    def evaluate(name, features, feature_seconds):
        all_metrics[name], all_predictions[name] = {}, {}
        for task, labels in TASKS.items():
            record, clf, predictions = select_classifier(features, rows, labels)
            record["feature_seconds"] = feature_seconds
            all_metrics[name][task] = record
            all_predictions[name][task] = predictions
            joblib.dump(clf, ROOT / "models" / f"{name}_{task}.joblib")
            csv_rows.append({
                "representation": name, "task": task, "C": record["C"],
                "valid_macro_f1": record["valid_macro_f1"],
                "test_accuracy": record["test_accuracy"], "test_macro_f1": record["test_macro_f1"],
                "dimension": features.shape[1], "feature_seconds": feature_seconds,
                "classifier_seconds": record["classifier_seconds"],
            })
            print(f"{name:20s} {task}  valid F1={record['valid_macro_f1']:.4f}  "
                  f"test F1={record['test_macro_f1']:.4f}", flush=True)
        write_json(ROOT / "results/metrics.json", all_metrics)
        write_json(ROOT / "results/predictions.json", all_predictions)

    with threadpool_limits(limits=1):
        started = time.perf_counter()
        dictionary = corpora.Dictionary(docs)
        dictionary.filter_extremes(no_below=2, no_above=1.0, keep_n=30000)
        dictionary.compactify()
        train_bow = [dictionary.doc2bow(doc) for doc in docs]
        tfidf = TfidfModel(train_bow, dictionary=dictionary, normalize=True)
        all_bow = [dictionary.doc2bow(r["tokens"]) for r in rows]
        all_tfidf = list(tfidf[all_bow])
        features = matutils.corpus2csc(all_tfidf, num_terms=len(dictionary),
                                     num_docs=len(rows), dtype=np.float32).T.tocsr()
        dictionary.save(str(ROOT / "models/dictionary.gensim"))
        tfidf.save(str(ROOT / "models/tfidf.gensim"))
        training["tfidf"] = {"vocabulary": len(dictionary), "seconds": time.perf_counter() - started}
        evaluate("tfidf", features, training["tfidf"]["seconds"])
        started = time.perf_counter()
        lsi = LsiModel(tfidf[train_bow], id2word=dictionary, num_topics=100, random_seed=SEED)
        lsi.save(str(ROOT / "models/lsi.gensim"))
        lsi_features = matutils.corpus2dense(lsi[all_tfidf], num_terms=100, num_docs=len(rows)).T
        training["lsi"] = {"dimension": 100, "seconds": time.perf_counter() - started,
                           "topics": lsi.show_topics(num_topics=5, num_words=8)}
        evaluate("lsi", normalize(lsi_features), training["lsi"]["seconds"])

        for name, config in CONFIGS.items():
            model_path = ROOT / "models" / f"{name}.gensim"
            sentences = external if config["corpus"] == "external" else adapted
            kwargs = {k: v for k, v in config.items() if k not in ("type", "corpus")}
            if config["type"] == "ft":
                kwargs["bucket"] = 50000
            started = time.perf_counter()
            if skip_training and model_path.exists():
                cls = FastText if config["type"] == "ft" else Word2Vec
                model = cls.load(str(model_path))
                training[name] = previous_training[name]
            else:
                cls = FastText if config["type"] == "ft" else Word2Vec
                model = cls(sentences=sentences, hashfxn=stable_hash, **COMMON, **kwargs)
                seconds = time.perf_counter() - started
                model.save(str(model_path), separately=[])
                training[name] = {
                    "seconds": seconds, "parameters": {**COMMON, **kwargs},
                    "corpus": config["corpus"], "sentences": len(sentences),
                    "tokens": sum(map(len, sentences)), "vocabulary": len(model.wv),
                    "model_bytes": model_path.stat().st_size,
                }
            print(f"Trained {name}: {training[name]['seconds']:.2f}s", flush=True)
            probes[name] = semantic_probes(model)
            coverage[name] = {}
            started = time.perf_counter()
            dense, _ = average_vectors(rows, model.wv)
            for split in ("train", "valid", "test"):
                _, coverage[name][split] = average_vectors(
                    [r for r in rows if r["split"] == split], model.wv)
            evaluate(name, dense, time.perf_counter() - started)
            if name == "ft_adapted":
                started = time.perf_counter()
                weighted, _ = average_vectors(rows, model.wv, idf)
                evaluate("ft_adapted_idf", weighted, time.perf_counter() - started)
            write_json(ROOT / "results/training.json", training)
            write_json(ROOT / "results/semantic_probes.json", probes)
            write_json(ROOT / "results/coverage.json", coverage)
            del model
            gc.collect()

    with (ROOT / "results/metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_json(ROOT / "models/idf.json", idf)
    summaries = {}
    for task in TASKS:
        winner = max(all_metrics, key=lambda n: all_metrics[n][task]["valid_macro_f1"])
        summaries[task] = {"selected_by_validation": winner, **all_metrics[winner][task]}
    write_json(ROOT / "results/selection.json", summaries)
    draw_figures(all_metrics)


def draw_figures(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["tfidf", "lsi", "w2v_cbow", "w2v_sg", "ft_external", "ft_adapted",
             "ft_adapted_idf"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(names))
    for offset, task in enumerate(TASKS):
        values = [metrics[n][task]["test_macro_f1"] for n in names]
        ax.bar(x + (offset - 1) * 0.24, values, width=0.24, label=task)
    ax.set_xticks(x, ["TF-IDF", "LSI", "W2V-CBOW", "W2V-SG", "FT-ext", "FT-adapt", "FT-IDF"])
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(ROOT / "results/comparison.png", dpi=180)
    plt.close(fig)
    record = metrics["ft_adapted_idf"]["4class"]
    matrix = np.asarray(record["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Greens")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(4), ["International", "Sports", "Economy", "Education"])
    ax.set_yticks(range(4), ["International", "Sports", "Economy", "Education"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("FastText + IDF, 4 classes")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(ROOT / "results/confusion_fasttext.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    run(args.skip_training)
