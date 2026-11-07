"""Prepare character-level punctuation labels from People's Daily 9801."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
LABELS = ["O", "，", "。", "、", "：", "；", "？", "！"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
PUNCTUATION = {
    ",": "，", "，": "，", "。": "。", "、": "、", ":": "：", "：": "：",
    ";": "；", "；": "；", "?": "？", "？": "？", "!": "！", "！": "！",
}
LINE_ID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)$")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")
PICTURE = re.compile(r"[（(]\s*附?\s*图片[^）)]*[）)]")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_body(body):
    body = PICTURE.sub("", PRONUNCIATION.sub("", body))
    return "".join(unicodedata.normalize("NFKC", body).split())


def is_content(text, index):
    char = text[index]
    if char.isalnum() or char in "%‰·":
        return True
    # Preserve decimal separators and signs inside numbers as lexical content.
    return (char in ".-" and 0 < index < len(text) - 1
            and text[index - 1].isdigit() and text[index + 1].isdigit())


def label_text(body):
    text = clean_body(body)
    chars, labels = [], []
    for index, char in enumerate(text):
        if char in PUNCTUATION:
            if labels:
                current = LABEL_TO_ID[PUNCTUATION[char]]
                # A sentence-ending mark wins in a run such as "？！".
                if labels[-1] not in (2, 6, 7):
                    labels[-1] = current
        elif is_content(text, index):
            chars.append(char)
            labels.append(0)
    return "".join(chars), labels


def render(text, labels):
    if len(text) != len(labels):
        raise ValueError("Text and labels have different lengths")
    return "".join(char + (LABELS[label] if label else "")
                   for char, label in zip(text, labels))


def prepare(source, chunk_size=256):
    if not 1 <= chunk_size <= 400:
        raise ValueError("chunk_size must be in 1..400")
    raw = source.read_bytes()
    decoded = raw.decode("gb18030", errors="strict")
    grouped = defaultdict(list)
    for line_no, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            continue
        match = LINE_ID.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"Malformed line {line_no}")
        date, page, article, paragraph, body = match.groups()
        grouped[f"{date}-{page}-{article}"].append((int(paragraph), body))

    articles, segments, chunks = [], [], []
    removed, seen = Counter(), set()
    for article_id, paragraphs in sorted(grouped.items()):
        date = article_id[:8]
        day = int(date[-2:])
        split = "train" if day <= 20 else "valid" if day <= 24 else "test"
        kept = []
        for paragraph, body in sorted(paragraphs):
            # Headings, author lines and captions have no full-stop supervision.
            if not any(mark in body for mark in "。？！!?"):
                removed["heading_or_caption"] += 1
                continue
            text, labels = label_text(body)
            if len(text) < 20 or not any(labels):
                removed["short_paragraph"] += 1
                continue
            kept.append({"id": f"{article_id}-{paragraph:03d}", "article_id": article_id,
                         "date": date, "split": split, "text": text, "labels": labels})
        if not kept:
            removed["empty_article"] += 1
            continue
        text = "".join(row["text"] for row in kept)
        labels = [label for row in kept for label in row["labels"]]
        signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if signature in seen:
            removed["duplicate_article"] += 1
            continue
        seen.add(signature)
        articles.append({"id": article_id, "date": date, "split": split, "text": text,
                         "labels": labels, "sha256": signature})
        segments.extend(kept)
        for segment in kept:
            for start in range(0, len(segment["text"]), chunk_size):
                end = min(start + chunk_size, len(segment["text"]))
                chunks.append({
                    "id": f"{segment['id']}:{start}", "article_id": article_id,
                    "split": split, "text": segment["text"][start:end],
                    "labels": segment["labels"][start:end],
                })

    for split in ("train", "valid", "test"):
        write_jsonl(ROOT / f"data/{split}.jsonl", [r for r in chunks if r["split"] == split])
    write_jsonl(ROOT / "data/articles.jsonl", articles)
    write_jsonl(ROOT / "data/paragraphs.jsonl", segments)
    splits = {}
    for split in ("train", "valid", "test"):
        subset = [r for r in articles if r["split"] == split]
        counts = Counter(label for r in subset for label in r["labels"])
        splits[split] = {
            "articles": len(subset),
            "paragraphs": sum(r["split"] == split for r in segments),
            "chunks": sum(r["split"] == split for r in chunks),
            "characters": sum(len(r["text"]) for r in subset),
            "dates": sorted({r["date"] for r in subset}),
            "labels": {label: counts[i] for i, label in enumerate(LABELS)},
            "long_articles_over_512": sum(len(r["text"]) > 512 for r in subset),
        }
    stats = {
        "source": source.name, "encoding": "gb18030", "source_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "raw_articles": len(grouped),
        "labels": LABELS, "chunk_size": chunk_size, "removed": dict(removed),
        "splits": splits,
    }
    write_json(ROOT / "results/data_stats.json", stats)
    write_json(ROOT / "results/split_manifest.json",
               [{key: r[key] for key in ("id", "date", "split", "sha256")} for r in articles])
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "人民日报语料" /
                        "1998-上半年-带音/1998-01-2003版-带音.txt")
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    prepare(args.source, args.chunk_size)
