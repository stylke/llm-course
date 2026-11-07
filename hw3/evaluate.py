"""Evaluate article-level restoration and benchmark long-document decoding."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from inference import Decoder
from modeling import add_confusion, metrics_from_confusion
from prepare import LABELS, ROOT, read_jsonl, render, write_json, write_jsonl


def score(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction count differs from document count")
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for row, predicted in zip(rows, predictions):
        add_confusion(matrix, row["labels"], predicted)
    return metrics_from_confusion(matrix)


def error_samples(rows, predictions, limit=12):
    samples = []
    for row, predicted in zip(rows, predictions):
        errors = [i for i, (gold, guess) in enumerate(zip(row["labels"], predicted)) if gold != guess]
        if not errors:
            continue
        center = errors[0]
        start, end = max(0, center - 35), min(len(row["text"]), center + 65)
        samples.append({
            "id": row["id"], "start": start, "end": end,
            "reference": render(row["text"][start:end], row["labels"][start:end]),
            "prediction": render(row["text"][start:end], predicted[start:end]),
            "errors": len(errors), "characters": len(row["text"]),
        })
        if len(samples) == limit:
            break
    return samples


def validation_merge(decoder, rows, batch_size):
    results = []
    # Same checkpoint, window plan and batch size; vary only the fusion rule.
    for mode in ("mean", "center"):
        predicted, timing = decoder.fixed([r["text"] for r in rows], batch_size=batch_size,
                                          merge=mode)
        result = {"merge": mode, "metrics": score(rows, predicted), "timing": timing}
        results.append(result)
        print(f"Validation merge={mode}: F1={result['metrics']['punctuation_micro_f1']:.4f}",
              flush=True)
    chosen = max(results, key=lambda r: r["metrics"]["punctuation_micro_f1"])["merge"]
    write_json(ROOT / "results/merge_selection.json", {"chosen": chosen, "candidates": results})
    return chosen


def evaluate(args):
    decoder = Decoder(device=args.device)
    articles = read_jsonl(ROOT / "data/articles.jsonl")
    validation = [r for r in articles if r["split"] == "valid"]
    test = [r for r in articles if r["split"] == "test"]
    decoder.probabilities(["人民日报标点恢复测试"])
    merge = validation_merge(decoder, validation, args.batch_size)
    predicted, timing = decoder.fixed([r["text"] for r in test],
                                      batch_size=args.batch_size, merge=merge)
    test_score = score(test, predicted)
    write_jsonl(ROOT / "results/test_predictions.jsonl",
                [{"id": r["id"], "reference": r["labels"], "prediction": p}
                 for r, p in zip(test, predicted)])
    write_json(ROOT / "results/error_samples.json", error_samples(test, predicted))
    zero = score(test, [[0] * len(r["text"]) for r in test])
    # Lightweight context baseline fitted solely from training characters.
    char_counts = {}
    for row in articles:
        if row["split"] == "train":
            for char, label in zip(row["text"], row["labels"]):
                char_counts.setdefault(char, Counter())[label] += 1
    lookup = {char: count.most_common(1)[0][0] for char, count in char_counts.items()}
    char_baseline = score(test, [[lookup.get(char, 0) for char in r["text"]] for r in test])
    write_json(ROOT / "results/test_metrics.json", {
        "device": str(decoder.device), "merge": merge, "bert": test_score,
        "no_punctuation": zero, "character_majority": char_baseline, "timing": timing,
    })
    print(f"Test articles: {len(test)}, punctuation F1 "
          f"{test_score['punctuation_micro_f1']:.4f}", flush=True)

    # First 24 qualifying articles in source order: reproducible, independent of errors.
    long_rows = [r for r in test if len(r["text"]) >= 1500][:args.benchmark_articles]
    if not long_rows:
        raise ValueError("No long test articles")
    texts = [r["text"] for r in long_rows]
    records, reference_predictions = [], None
    for method, batch_size in [("sequential", 1), ("fixed", 1),
                               ("fixed", 4), ("fixed", 8), ("fixed", 16)]:
        # Warm up each batch shape before synchronized timing.
        decoder.probabilities(["标点恢复" * 100] * batch_size)
        runs = []
        for repeat in range(args.repeats):
            if method == "sequential":
                output, run = decoder.sequential(texts)
            else:
                output, run = decoder.fixed(texts, batch_size=batch_size, merge=merge)
            runs.append(run)
            if method == "fixed":
                if reference_predictions is None:
                    reference_predictions = output
                differences = sum(sum(a != b for a, b in zip(left, right))
                                  for left, right in zip(reference_predictions, output))
            else:
                differences = None
        seconds = float(np.median([r["seconds"] for r in runs]))
        record = {
            "method": method, "batch_size": batch_size, "repeats": args.repeats,
            "seconds_median": seconds, "seconds_runs": [r["seconds"] for r in runs],
            "characters_per_second": sum(map(len, texts)) / seconds,
            "model_calls": runs[-1]["model_calls"],
            "processed_characters": runs[-1]["processed_characters"],
            "metrics": score(long_rows, output),
            "differences_from_fixed_batch1": differences,
        }
        records.append(record)
        print(f"Benchmark {method} batch={batch_size}: {seconds:.3f}s, "
              f"{record['characters_per_second']:.0f} chars/s, "
              f"calls={record['model_calls']}", flush=True)
    for record in records:
        record["speedup_vs_sequential"] = records[0]["seconds_median"] / record["seconds_median"]
    write_json(ROOT / "results/benchmark.json", {
        "device": str(decoder.device), "window_size": 400, "stride": 300, "merge": merge,
        "article_ids": [r["id"] for r in long_rows], "articles": len(long_rows),
        "characters": sum(map(len, texts)), "lengths": [len(t) for t in texts],
        "records": records,
    })

    examples = [
        ("现代文", "全国人民代表大会常务委员会今天举行会议审议有关法律草案"
         "与会人员认真讨论提出修改意见会议决定将草案提交下一次会议继续审议"),
        ("古文", "学而时习之不亦说乎有朋自远方来不亦乐乎人不知而不愠不亦君子乎"),
        ("古文", "庆历四年春滕子京谪守巴陵郡越明年政通人和百废具兴"
         "乃重修岳阳楼增其旧制刻唐贤今人诗赋于其上属予作文以记之"),
    ]
    outputs, _ = decoder.fixed([text for _, text in examples], batch_size=args.batch_size, merge=merge)
    write_json(ROOT / "results/demo.json",
               [{"domain": domain, "input": text, "output": render(text, labels)}
                for (domain, text), labels in zip(examples, outputs)])
    write_json(ROOT / "results/long_example.json", {
        "id": long_rows[0]["id"], "characters": len(long_rows[0]["text"]),
        "input": long_rows[0]["text"],
        "reference": render(long_rows[0]["text"], long_rows[0]["labels"]),
        "output": render(long_rows[0]["text"], reference_predictions[0]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benchmark-articles", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.benchmark_articles, args.repeats) < 1:
        parser.error("batch size, article count and repeats must be positive")
    evaluate(args)
