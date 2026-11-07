"""Batched overlapping-window decoding for long documents."""
from __future__ import annotations

from dataclasses import dataclass
import json
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification

from modeling import encode_texts, get_device, load_tokenizer, synchronize
from prepare import LABELS, ROOT


@dataclass(frozen=True)
class Window:
    document: int
    start: int
    end: int


def plan_windows(lengths, window_size=400, stride=300):
    if not 1 <= stride <= window_size <= 400:
        raise ValueError("Expected 1 <= stride <= window_size <= 400")
    windows = []
    for document, length in enumerate(lengths):
        start = 0
        while start < length:
            end = min(start + window_size, length)
            windows.append(Window(document, start, end))
            if end == length:
                break
            start += stride
    return windows


def merge_weights(length, mode):
    if mode == "mean":
        return np.ones(length, dtype=np.float32)
    if mode == "center":
        position = np.arange(length)
        return np.minimum(np.minimum(position + 1, length - position), 51).astype(np.float32)
    raise ValueError("Unknown merge mode")


class Decoder:
    def __init__(self, model_path=None, device="auto", threads=4):
        torch.set_num_threads(threads)
        self.device = get_device(device)
        self.path = model_path or ROOT / "models/punctuation"
        self.tokenizer = load_tokenizer(self.path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.path, local_files_only=True, attn_implementation="sdpa").to(self.device).eval()
        calibration = ROOT / "results/calibration.json"
        self.none_bias = (json.loads(calibration.read_text(encoding="utf-8"))["chosen_none_bias"]
                          if calibration.exists() else 0.0)
        self.calls = 0
        self.processed_characters = 0

    @torch.inference_mode()
    def probabilities(self, texts):
        if not texts or any(not text for text in texts):
            raise ValueError("Model batches must contain nonempty texts")
        encoded, positions = encode_texts(self.tokenizer, texts)
        logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
        logits[..., 0] += self.none_bias
        probabilities = logits.softmax(-1).cpu().numpy()
        self.calls += 1
        self.processed_characters += sum(len(text) for text in texts)
        return [probabilities[index, tokens] for index, tokens in enumerate(positions)]

    def fixed(self, texts, batch_size=8, window_size=400, stride=300, merge="center"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        windows = plan_windows([len(text) for text in texts], window_size, stride)
        # Group similar lengths so padding does not dominate short tail windows.
        ordered = sorted(windows, key=lambda w: w.end - w.start, reverse=True)
        scores = [np.zeros((len(text), len(LABELS)), dtype=np.float32) for text in texts]
        totals = [np.zeros(len(text), dtype=np.float32) for text in texts]
        calls_before, chars_before = self.calls, self.processed_characters
        synchronize(self.device)
        started = time.perf_counter()
        for start in range(0, len(ordered), batch_size):
            group = ordered[start:start + batch_size]
            predictions = self.probabilities([texts[w.document][w.start:w.end] for w in group])
            for window, probability in zip(group, predictions):
                weights = merge_weights(window.end - window.start, merge)
                scores[window.document][window.start:window.end] += probability * weights[:, None]
                totals[window.document][window.start:window.end] += weights
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        predictions = []
        for score, total in zip(scores, totals):
            if len(total) and not np.all(total > 0):
                raise ValueError("A character was left uncovered")
            # The denominator is shared by all labels at a position.
            predictions.append(score.argmax(-1).tolist())
        return predictions, {
            "method": f"fixed_{merge}", "batch_size": batch_size,
            "window_size": window_size, "stride": stride, "windows": len(windows),
            "model_calls": self.calls - calls_before,
            "processed_characters": self.processed_characters - chars_before,
            "seconds": elapsed, "characters": sum(map(len, texts)),
            "characters_per_second": sum(map(len, texts)) / max(elapsed, 1e-12),
        }

    def sequential(self, texts, window_size=400, commit=300):
        if not 1 <= commit <= window_size <= 400:
            raise ValueError("Expected 1 <= commit <= window_size <= 400")
        calls_before, chars_before = self.calls, self.processed_characters
        synchronize(self.device)
        started = time.perf_counter()
        results = []
        for text in texts:
            position, labels = 0, []
            while position < len(text):
                end = min(position + window_size, len(text))
                predicted = self.probabilities([text[position:end]])[0].argmax(-1).tolist()
                if end == len(text):
                    take = len(predicted)
                else:
                    # Keep the last predicted full stop in the first ~300 characters.
                    stops = [i + 1 for i, label in enumerate(predicted[:commit]) if label == 2]
                    take = stops[-1] if stops else min(commit, len(predicted))
                labels.extend(predicted[:take])
                position += take
            results.append(labels)
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        return results, {
            "method": "sequential_stop", "batch_size": 1, "window_size": window_size,
            "commit": commit, "model_calls": self.calls - calls_before,
            "processed_characters": self.processed_characters - chars_before,
            "seconds": elapsed, "characters": sum(map(len, texts)),
            "characters_per_second": sum(map(len, texts)) / max(elapsed, 1e-12),
        }
