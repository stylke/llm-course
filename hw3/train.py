"""Fine-tune a pretrained Chinese BERT for punctuation restoration."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import platform
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForTokenClassification, get_linear_schedule_with_warmup

from modeling import (PadBatch, PunctuationDataset, add_confusion, get_device,
                      load_tokenizer, metrics_from_confusion, set_seed, synchronize)
from prepare import LABELS, ROOT, read_jsonl, write_json


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for batch in loader:
        labels = batch.pop("labels").numpy()
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        predicted = logits.argmax(-1).cpu().numpy()
        valid = labels != -100
        add_confusion(matrix, labels[valid], predicted[valid])
    return metrics_from_confusion(matrix)


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    tokenizer = load_tokenizer()
    rows = {split: read_jsonl(ROOT / f"data/{split}.jsonl")
            for split in ("train", "valid", "test")}
    print(f"Device: {device}; encoding datasets", flush=True)
    datasets = {split: PunctuationDataset(data, tokenizer) for split, data in rows.items()}
    collate = PadBatch(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=split == "train", collate_fn=collate,
                                generator=generator if split == "train" else None)
               for split, dataset in datasets.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        ROOT / "models/pretrained", num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)},
        local_files_only=True, attn_implementation="sdpa",
    ).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    encoder_before = model.bert.encoder.layer[0].attention.self.query.weight.detach().cpu().clone()
    counts = Counter(label for row in rows["train"] for label in row["labels"])
    weights = torch.tensor([math.sqrt(counts[0] / counts[i]) for i in range(len(LABELS))],
                           dtype=torch.float32, device=device).clamp(max=8)
    weights = weights / weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    output = ROOT / "models/punctuation"
    output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch = [], -1, None
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for step, batch in enumerate(loaders["train"], 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits.reshape(-1, len(LABELS)), labels.reshape(-1))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Training loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0 or step == len(loaders["train"]):
                print(f"Epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss={np.mean(losses[-50:]):.4f} "
                      f"elapsed={time.perf_counter() - epoch_start:.1f}s", flush=True)
        valid = evaluate(model, loaders["valid"], device)
        synchronize(device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "seconds": time.perf_counter() - epoch_start, "valid": valid}
        history.append(record)
        print(f"Epoch {epoch}: valid punctuation micro-F1 "
              f"{valid['punctuation_micro_f1']:.4f}, macro-F1 "
              f"{valid['punctuation_macro_f1']:.4f}", flush=True)
        if valid["punctuation_micro_f1"] > best_score:
            best_score, best_epoch = valid["punctuation_micro_f1"], epoch
            model.save_pretrained(output, safe_serialization=True)
            tokenizer.save_pretrained(output)
        write_json(ROOT / "results/training_history.json", history)

    synchronize(device)
    training_seconds = time.perf_counter() - started
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    best = AutoModelForTokenClassification.from_pretrained(
        output, local_files_only=True, attn_implementation="sdpa").to(device)
    encoder_change = float((best.bert.encoder.layer[0].attention.self.query.weight.detach().cpu()
                            - encoder_before).abs().max())
    valid = evaluate(best, loaders["valid"], device)
    test = evaluate(best, loaders["test"], device)
    checkpoint_size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    result = {
        "seed": args.seed, "device": str(device), "python": platform.python_version(),
        "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "parameters": parameters,
        "trainable_parameters": parameters, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "optimizer": "AdamW", "weight_decay": 0.01, "warmup_ratio": 0.1,
        "class_weights": weights.cpu().tolist(), "gradient_clip": 1.0,
        "best_epoch": best_epoch, "training_seconds": training_seconds,
        "encoder_first_query_max_change": encoder_change,
        "checkpoint_bytes": checkpoint_size, "valid": valid, "test_chunks": test,
    }
    write_json(ROOT / "results/training_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.threads < 1:
        parser.error("epochs, batch size, learning rate and threads must be positive")
    train(args)
