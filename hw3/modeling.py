"""Shared token alignment and device helpers."""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from prepare import LABELS, ROOT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def load_tokenizer(path=None):
    return AutoTokenizer.from_pretrained(path or ROOT / "models/pretrained",
                                        use_fast=True, local_files_only=True)


def encode_texts(tokenizer, texts):
    encoded = tokenizer([list(text) for text in texts], is_split_into_words=True,
                        padding=True, truncation=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > 512:
        raise ValueError("Encoded window exceeds BERT's 512-token context")
    positions = []
    for index, text in enumerate(texts):
        last = None
        first = []
        for token, word in enumerate(encoded.word_ids(batch_index=index)):
            if word is not None and word != last:
                first.append(token)
            last = word
        if len(first) != len(text):
            raise ValueError("Some input characters have no tokenizer alignment")
        positions.append(first)
    return dict(encoded), positions


class PunctuationDataset:
    def __init__(self, rows, tokenizer):
        self.items = []
        for row in rows:
            encoded, positions = encode_texts(tokenizer, [row["text"]])
            labels = torch.full_like(encoded["input_ids"][0], -100)
            labels[positions[0]] = torch.tensor(row["labels"])
            self.items.append({**{k: v[0] for k, v in encoded.items()}, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


@dataclass
class PadBatch:
    pad_id: int

    def __call__(self, items):
        length = max(len(item["input_ids"]) for item in items)
        output = {}
        for key in items[0]:
            fill = -100 if key == "labels" else self.pad_id if key == "input_ids" else 0
            tensor = torch.full((len(items), length), fill, dtype=torch.long)
            for index, item in enumerate(items):
                tensor[index, :len(item[key])] = item[key]
            output[key] = tensor
        return output


def metrics_from_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = matrix.diagonal().astype(float)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=precision + recall != 0)
    p_tp = tp[1:].sum()
    p_precision = float(p_tp / max(1, predicted[1:].sum()))
    p_recall = float(p_tp / max(1, support[1:].sum()))
    p_f1 = 2 * p_precision * p_recall / max(1e-12, p_precision + p_recall)
    # Boundary detection ignores which of the seven marks was predicted.
    boundary_tp = int(matrix[1:, 1:].sum())
    boundary_p = boundary_tp / max(1, int(predicted[1:].sum()))
    boundary_r = boundary_tp / max(1, int(support[1:].sum()))
    return {
        "characters": int(matrix.sum()), "accuracy": float(tp.sum() / max(1, matrix.sum())),
        "punctuation_precision": p_precision, "punctuation_recall": p_recall,
        "punctuation_micro_f1": p_f1, "punctuation_macro_f1": float(f1[1:].mean()),
        "boundary_f1": 2 * boundary_p * boundary_r / max(1e-12, boundary_p + boundary_r),
        "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                              "f1": float(f1[i]), "support": int(support[i])}
                      for i, label in enumerate(LABELS)},
        "confusion_matrix": matrix.tolist(),
    }


def add_confusion(matrix, gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Prediction length differs from reference")
    flat = np.asarray(gold, dtype=np.int64) * len(LABELS) + np.asarray(predicted, dtype=np.int64)
    matrix += np.bincount(flat, minlength=len(LABELS) ** 2).reshape(len(LABELS), len(LABELS))
