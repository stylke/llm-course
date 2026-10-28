"""训练、验证选阶、最终测试并保存续写与运行日志。"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import kenlm

from generate import generate
from prepare import prepare

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(model, path):
    start = time.perf_counter()
    log10_sum, tokens, sentences = 0.0, 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            log10_sum += model.score(line, bos=True, eos=True)
            tokens += len(line.split()) + 1  # EOS计入分母，BOS不计入。
            sentences += 1
    return {"sentences": sentences, "predicted_tokens_with_eos": tokens,
            "log10_probability": log10_sum,
            "nll_per_token_nats": -log10_sum * math.log(10) / tokens,
            "perplexity": 10 ** (-log10_sum / tokens),
            "seconds": time.perf_counter() - start}


def run_logged(command, path):
    start = time.perf_counter()
    with path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(map(str, command)) + "\n")
        log.flush()
        subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path,
                   default=ROOT.parent / "人民日报语料" / "1998-上半年-带音")
    p.add_argument("--kenlm-bin", type=Path, default=ROOT / "tools")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-generation", action="store_true")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats必须大于0")
    data, models, results = ROOT / "data", ROOT / "models", ROOT / "results"
    logs = results / "logs"
    for directory in [models, results, logs, models / "temp"]:
        directory.mkdir(exist_ok=True, parents=True)
    if not args.skip_prepare:
        prepare(args.source)
    cpu = platform.processor()
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            cpu = result.stdout.strip()
    save_json(results / "environment.json", {
        "run_time_local": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": cpu,
        "python": sys.version, "python_executable": sys.executable,
        "kenlm_version": importlib.metadata.version("kenlm"),
        "lmplz": str((args.kenlm_bin / "lmplz").resolve()),
        "smoothing": "interpolated modified Kneser-Ney (KenLM default)",
        "memory_budget": "512M", "pruning": "none", "repeats": args.repeats})

    measurements = []
    if not args.skip_training:
        for order in [2, 3, 4]:
            arpa = models / f"{order}gram.arpa"
            train_times, build_times = [], []
            for repeat in range(1, args.repeats + 1):
                # 不启用discount_fallback：如果频数不足以估计折扣，应直接暴露失败。
                command = [args.kenlm_bin / "lmplz", "-o", order, "-S", "512M",
                           "-T", models / "temp", "--text", data / "train.txt",
                           "--arpa", arpa]
                seconds = run_logged(command, logs / f"train_{order}gram_{repeat}.log")
                train_times.append(seconds)
                print(f"{order}-gram 第{repeat}次训练：{seconds:.3f}s", flush=True)
                seconds = run_logged([args.kenlm_bin / "build_binary", "trie", arpa,
                                      models / f"{order}gram.bin"],
                                     logs / f"build_{order}gram_{repeat}.log")
                build_times.append(seconds)
            counts = {}
            with arpa.open(encoding="utf-8") as stream:
                for line in stream:
                    match = re.match(r"ngram (\d+)=(\d+)", line)
                    if match:
                        counts[int(match[1])] = int(match[2])
                    if line.strip() == "\\1-grams:":
                        break
            measurements.append({"order": order, "train_seconds": train_times,
                                 "binary_build_seconds": build_times,
                                 "train_median_seconds": statistics.median(train_times),
                                 "ngram_counts": counts, "arpa_bytes": arpa.stat().st_size,
                                 "binary_bytes": (models / f"{order}gram.bin").stat().st_size})
        save_json(results / "training.json", measurements)
    else:
        measurements = json.loads((results / "training.json").read_text(encoding="utf-8"))

    # 先完成所有验证集评测，再选定阶数，最后统一报告测试结果。
    for row in measurements:
        model = kenlm.Model(str(models / f"{row['order']}gram.bin"))
        row["valid"] = evaluate(model, data / "valid.txt")
    selected = min(measurements, key=lambda r: r["valid"]["perplexity"])["order"]
    save_json(results / "selection.json", {"criterion": "lowest validation perplexity",
                                            "selected_order": selected})
    for row in measurements:
        order = row["order"]
        model = kenlm.Model(str(models / f"{order}gram.bin"))
        row["test"] = evaluate(model, data / "test.txt")
        print(f"{order}-gram valid PPL={row['valid']['perplexity']:.3f}, "
              f"test PPL={row['test']['perplexity']:.3f}", flush=True)
    save_json(results / "metrics.json", measurements)
    with (results / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["order", "train_median_seconds", "train_runs_seconds", "binary_mib",
                  "highest_order_ngrams", "valid_ppl", "test_ppl", "test_nll_nats"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in measurements:
            writer.writerow({"order": row["order"],
                             "train_median_seconds": row["train_median_seconds"],
                             "train_runs_seconds": ";".join(f"{s:.6f}" for s in row["train_seconds"]),
                             "binary_mib": row["binary_bytes"] / 1024 ** 2,
                             "highest_order_ngrams": row["ngram_counts"].get(str(row["order"]),
                                                                         row["ngram_counts"].get(row["order"])),
                             "valid_ppl": row["valid"]["perplexity"],
                             "test_ppl": row["test"]["perplexity"],
                             "test_nll_nats": row["test"]["nll_per_token_nats"]})

    if args.skip_generation:
        return
    vocabulary = json.loads((data / "vocab.json").read_text(encoding="utf-8"))
    configs = [
        ("greedy", {"greedy": True}),
        ("T0.7_k20", {"temperature": 0.7, "top_k": 20}),
        ("T1.0_k20", {"temperature": 1.0, "top_k": 20}),
        ("T1.0_k100", {"temperature": 1.0, "top_k": 100}),
        ("T1.0_p0.9", {"temperature": 1.0, "top_k": 0, "top_p": 0.9})]
    generations = []
    for order in [2, 3, 4]:
        model = kenlm.Model(str(models / f"{order}gram.bin"))
        for name, config in configs:
            start = time.perf_counter()
            sample = generate(model, vocabulary, seed=42, **config)
            sample.update(order=order, config=name, seconds=time.perf_counter() - start)
            generations.append(sample)
            print(f"{order}-gram {name}: {sample['text']}", flush=True)
    # 在验证集选定的模型上增加固定种子，不凭好坏筛选输出。
    model = kenlm.Model(str(models / f"{selected}gram.bin"))
    for seed in [7, 2026]:
        sample = generate(model, vocabulary, seed=seed, temperature=0.7, top_k=20)
        sample.update(order=selected, config="T0.7_k20")
        generations.append(sample)
    save_json(results / "generations.json", generations)
    with (results / "generations.md").open("w", encoding="utf-8") as stream:
        stream.write("# 原始续写结果\n\n所有输出按预设顺序保存，未人工润色。最长新增60个词，句末标点或EOS停止。\n\n")
        for sample in generations:
            stream.write(f"## {sample['order']}-gram / {sample['config']} / seed={sample['seed']}\n\n")
            stream.write(sample["text"] + "\n\n")
            stream.write(f"停止原因：`{sample['stop_reason']}`；开头未登录词：`{sample['prompt_oov']}`。\n\n")


if __name__ == "__main__":
    main()
