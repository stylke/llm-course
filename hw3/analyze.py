"""Recompute saved results, check model reloads and draw experiment figures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForTokenClassification

from inference import Decoder
from modeling import add_confusion, metrics_from_confusion
from prepare import LABELS, ROOT, read_jsonl, render, write_json


def load(name):
    return json.loads((ROOT / f"results/{name}.json").read_text(encoding="utf-8"))


def analyze(device):
    rows = read_jsonl(ROOT / "data/articles.jsonl")
    test = [row for row in rows if row["split"] == "test"]
    saved = read_jsonl(ROOT / "results/test_predictions.jsonl")
    assert [r["id"] for r in saved] == [r["id"] for r in test]
    matrix = np.zeros((8, 8), dtype=np.int64)
    gold, predicted = [], []
    for row, prediction in zip(test, saved):
        assert prediction["reference"] == row["labels"]
        assert len(prediction["prediction"]) == len(row["text"])
        assert all(isinstance(label, int) and 0 <= label < 8 for label in prediction["prediction"])
        add_confusion(matrix, row["labels"], prediction["prediction"])
        gold.extend(row["labels"])
        predicted.extend(prediction["prediction"])
    metrics = metrics_from_confusion(matrix)
    reference = load("test_metrics")["bert"]
    assert metrics == reference, "Saved predictions and reported metrics differ"
    sklearn_f1 = f1_score(gold, predicted, labels=list(range(1, 8)), average="micro",
                         zero_division=0)
    assert np.isclose(sklearn_f1, reference["punctuation_micro_f1"], rtol=0, atol=1e-12)
    split_hashes = [{r["sha256"] for r in rows if r["split"] == split}
                    for split in ("train", "valid", "test")]
    assert not split_hashes[0] & split_hashes[1]
    assert not split_hashes[0] & split_hashes[2]
    assert not split_hashes[1] & split_hashes[2]
    manifest = load("pretrained_manifest")
    for item in manifest["files"]:
        actual = hashlib.sha256((ROOT / "models/pretrained" / item["name"]).read_bytes()).hexdigest()
        assert actual == item["sha256"], "Pretrained checkpoint checksum mismatch"

    model = AutoModelForTokenClassification.from_pretrained(
        ROOT / "models/punctuation", local_files_only=True)
    finite = all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
    assert finite
    del model
    decoder = Decoder(device=device)
    sample = test[:5]
    merge = load("merge_selection")["chosen"]
    reloaded, _ = decoder.fixed([r["text"] for r in sample], batch_size=8, merge=merge)
    assert reloaded == [r["prediction"] for r in saved[:5]], "Reloaded model changed predictions"
    training = load("training_metrics")
    assert training["encoder_first_query_max_change"] > 0
    history = load("training_history")
    assert len(history) == training["epochs"]
    assert all(np.isfinite(row["train_loss"]) for row in history)
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()}
    checks = {
        "saved_prediction_articles": len(saved),
        "saved_prediction_characters": len(gold),
        "metrics_recomputed": True,
        "sklearn_punctuation_micro_f1": float(sklearn_f1),
        "article_split_and_hashes_disjoint": True,
        "pretrained_checksums_match": True,
        "model_parameters_finite": finite,
        "encoder_updated": True,
        "reload_prediction_articles_match": len(sample),
        "batch_prediction_differences": {
            str(r["batch_size"]): r["differences_from_fixed_batch1"]
            for r in load("benchmark")["records"] if r["method"] == "fixed"
        },
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
    write_json(ROOT / "results/checks.json", checks)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], "o-", color="#25786b")
    axes[0].set(xlabel="Epoch", ylabel="Weighted training loss", xticks=epochs)
    axes[1].plot(epochs, [row["valid"]["punctuation_micro_f1"] for row in history],
                 "o-", color="#a54b67", label="Micro-F1")
    axes[1].plot(epochs, [row["valid"]["punctuation_macro_f1"] for row in history],
                 "s-", color="#386d9e", label="Macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Validation punctuation F1", xticks=epochs, ylim=(0, 1))
    axes[1].legend()
    fig.savefig(ROOT / "results/training.png", dpi=160)
    plt.close(fig)

    records = load("benchmark")["records"]
    names = ["Sequential\nbatch 1"] + [f"Fixed\nbatch {r['batch_size']}" for r in records[1:]]
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    bars = ax.bar(names, [r["characters_per_second"] for r in records],
                  color=["#8b8e93", "#386d9e", "#25786b", "#a54b67", "#c39e30"])
    for bar, row in zip(bars, records):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{row['characters_per_second']:.0f}", ha="center", va="bottom")
    ax.set(ylabel="Characters / second (median of 3 runs)")
    ax.margins(y=0.18)
    fig.savefig(ROOT / "results/throughput.png", dpi=160)
    plt.close(fig)

    names = ["Comma", "Period", "List", "Colon", "Semicolon", "Question", "Exclamation"]
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    ax.bar(names, [reference["per_label"][label]["f1"] for label in LABELS[1:]],
           color=["#25786b", "#386d9e", "#a54b67", "#c39e30", "#767f40", "#b35e49", "#777984"])
    ax.set(ylabel="Test F1", ylim=(0, 1))
    fig.savefig(ROOT / "results/label_f1.png", dpi=160)
    plt.close(fig)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    analyze(parser.parse_args().device)
