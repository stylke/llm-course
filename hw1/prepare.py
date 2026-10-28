"""清理带音人民日报语料，按月份划分，保留原有词边界。仅依赖标准库。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ID = re.compile(r"^\s*(\d{8}-\d{2}-\d{3}-\d{3})\s+")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")
END = set("。！？!?")
CLOSERS = set("”’」』）》】〕〉")
# 小写<unk>是lmplz的保留符号，不能直接出现在训练文本中。
# 用__RARE__作为可学习的低频词类别，评测时也明确映射到该类别。
UNKNOWN = "__RARE__"


def sentences(words):
    """句末标点保留，紧随其后的右引号留在同一句；〓只用作边界。"""
    current = []
    ended = False
    for word in words:
        if word == "〓":
            if current:
                yield current
            current, ended = [], False
            continue
        if ended and word not in CLOSERS and word not in END:
            yield current
            current, ended = [], False
        current.append(word)
        ended = ended or word in END
    if current:
        yield current


def prepare(source: Path, min_count: int = 2):
    started = time.perf_counter()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    stats = {"source": str(source.resolve()), "encoding": "gb18030",
             "min_count": min_count, "files": [], "splits": {}}
    seen = set()
    split_sentences = {name: [] for name in ("train", "valid", "test")}
    files = sorted(source.glob("*.txt"))
    if len(files) != 6:
        raise ValueError(f"预期6个月份文件，实际找到{len(files)}个")
    for path in files:
        raw = path.read_bytes()
        # 先用严格模式确认编码异常，然后记录带替代字符的整行并剔除。
        decode_error = None
        try:
            text = raw.decode("gb18030")
        except UnicodeDecodeError as error:
            decode_error = {"first_byte_offset": error.start, "reason": error.reason}
            text = raw.decode("gb18030", errors="replace")
        month = int(path.name[5:7])
        split = "train" if month <= 4 else "valid" if month == 5 else "test"
        record = {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                  "bytes": len(raw), "split": split, "nonblank_lines": 0,
                  "pronunciation_annotations": 0, "separator_count": 0,
                  "replacement_characters": text.count("\ufffd"),
                  "decode_error": decode_error, "dropped_damaged_lines": [],
                  "duplicates_removed": 0, "short_fragments_removed": 0,
                  "kept_sentences": 0}
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            record["nonblank_lines"] += 1
            record["pronunciation_annotations"] += len(PRONUNCIATION.findall(line))
            record["separator_count"] += line.count("〓")
            match = ID.match(line)
            if not match:
                raise ValueError(f"{path.name}:{line_number} 缺少预期的行编号")
            if "\ufffd" in line:
                record["dropped_damaged_lines"].append({"line": line_number, "id": match[1]})
                continue
            body = PRONUNCIATION.sub("", line[match.end():])
            words = body.split()
            if any("{" in w or "}" in w for w in words):
                raise ValueError(f"{path.name}:{line_number} 残留读音标记")
            for sentence in sentences(words):
                if sum(any(c.isalnum() for c in w) for w in sentence) < 2:
                    record["short_fragments_removed"] += 1
                    continue
                key = " ".join(sentence)
                if key in seen:
                    record["duplicates_removed"] += 1
                    continue
                seen.add(key)
                split_sentences[split].append(sentence)
                record["kept_sentences"] += 1
        stats["files"].append(record)

    counts = Counter(w for sentence in split_sentences["train"] for w in sentence)
    vocabulary = sorted(w for w, count in counts.items() if count >= min_count)
    vocab_set = set(vocabulary)
    (data / "vocab.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    for split, rows in split_sentences.items():
        mapped = [[w if w in vocab_set else UNKNOWN for w in s] for s in rows]
        raw_path = data / f"{split}.raw.txt"
        model_path = data / f"{split}.txt"
        raw_path.write_text("".join(" ".join(s) + "\n" for s in rows), encoding="utf-8")
        model_path.write_text("".join(" ".join(s) + "\n" for s in mapped), encoding="utf-8")
        token_count = sum(map(len, rows))
        unknowns = sum(w not in vocab_set for s in rows for w in s)
        stats["splits"][split] = {
            "sentences": len(rows), "tokens": token_count,
            "unknown_tokens": unknowns, "unknown_rate": unknowns / token_count,
            "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    stats["train_types_before_cutoff"] = len(counts)
    stats["vocabulary_size_without_specials"] = len(vocabulary)
    stats["preprocess_seconds"] = time.perf_counter() - started
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "data_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["splits"], ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT.parent / "人民日报语料" / "1998-上半年-带音")
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()
    if args.min_count < 1:
        parser.error("min-count必须大于0")
    prepare(args.source, args.min_count)
