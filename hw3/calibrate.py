"""Select a no-punctuation logit bias on validation data."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from modeling import (PadBatch, PunctuationDataset, add_confusion, get_device,
                      load_tokenizer, metrics_from_confusion)
from prepare import LABELS, ROOT, read_jsonl, write_json


@torch.inference_mode()
def calibrate(device_name):
    torch.set_num_threads(4)
    device = get_device(device_name)
    tokenizer = load_tokenizer(ROOT / "models/punctuation")
    dataset = PunctuationDataset(read_jsonl(ROOT / "data/valid.jsonl"), tokenizer)
    loader = DataLoader(dataset, batch_size=32, collate_fn=PadBatch(tokenizer.pad_token_id))
    model = AutoModelForTokenClassification.from_pretrained(
        ROOT / "models/punctuation", local_files_only=True,
        attn_implementation="sdpa").to(device).eval()
    candidates = (0.0, 0.5, 1.0, 1.5, 2.0)
    matrices = {bias: np.zeros((len(LABELS), len(LABELS)), dtype=np.int64) for bias in candidates}
    for batch in loader:
        labels = batch.pop("labels").numpy()
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu().numpy()
        valid = labels != -100
        for bias, matrix in matrices.items():
            adjusted = logits.copy()
            adjusted[..., 0] += bias
            predicted = adjusted.argmax(-1)
            add_confusion(matrix, labels[valid], predicted[valid])
    records = [{"none_bias": bias, "valid": metrics_from_confusion(matrix)}
               for bias, matrix in matrices.items()]
    best = max(records, key=lambda r: r["valid"]["punctuation_micro_f1"])
    result = {
        "selection_data": "validation chunks only",
        "chosen_none_bias": best["none_bias"], "candidates": records,
    }
    write_json(ROOT / "results/calibration.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    calibrate(parser.parse_args().device)
