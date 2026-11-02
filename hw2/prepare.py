"""Prepare the external news titles and the January People's Daily articles."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
PAGE_LABELS = {"02": "economy", "05": "education", "06": "international", "08": "sports"}
LABEL_NAMES = {
    "international": "国际", "sports": "体育", "economy": "国内经济", "education": "科教文卫"
}
LINE_ID = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)$")
PRONUNCIATION = re.compile(r"\{[^{}]*\}")
PICTURE = re.compile(r"[（(]\s*附?\s*图片[^）)]*[）)]")
CONTENT = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
SENTENCE_END = set("。！？!?〓")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_token(word):
    word = unicodedata.normalize("NFKC", PRONUNCIATION.sub("", word)).lower()
    return word if CONTENT.search(word) else ""


def token_sentences(words):
    sentence = []
    for word in words:
        if word in SENTENCE_END:
            if len(sentence) >= 2:
                yield sentence
            sentence = []
        else:
            token = clean_token(word)
            if token:
                sentence.append(token)
    if len(sentence) >= 2:
        yield sentence


def january_articles(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("gb18030")
        decode_error = None
    except UnicodeDecodeError as error:
        text = raw.decode("gb18030", errors="replace")
        decode_error = {"offset": error.start, "reason": error.reason}
    grouped = {}
    damaged = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = LINE_ID.fullmatch(line.strip())
        if not match:
            raise ValueError(f"Unexpected line ID at {number}")
        date, page, article, paragraph, body = match.groups()
        key = f"{date}-{page}-{article}"
        if "\ufffd" in body:
            damaged.add(key)
        grouped.setdefault(key, []).append((int(paragraph), body))

    rows = []
    removed = Counter()
    seen = set()
    for key, paragraphs in sorted(grouped.items()):
        date, page, _ = key.split("-")
        day = int(date[-2:])
        if page not in PAGE_LABELS or not 5 <= day <= 27:
            removed["outside_date_or_page"] += 1
            continue
        if datetime.strptime(date, "%Y%m%d").weekday() >= 5:
            removed["weekend"] += 1
            continue
        if key in damaged:
            removed["damaged_article"] += 1
            continue
        paragraphs.sort()
        clean = [PICTURE.sub("", PRONUNCIATION.sub("", body)) for _, body in paragraphs]
        sentences = [s for body in clean for s in token_sentences(body.split())]
        tokens = [word for sentence in sentences for word in sentence]
        if len(tokens) < 20:
            removed["short_article"] += 1
            continue
        signature = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
        if signature in seen:
            removed["duplicate_article"] += 1
            continue
        seen.add(signature)
        split = "train" if day <= 16 else "valid" if day <= 21 else "test"
        rows.append({
            "id": key, "date": date, "page": page, "label": PAGE_LABELS[page],
            "split": split, "title": "".join(clean[0].split())[:160],
            "tokens": tokens, "sentences": sentences, "sha256": signature,
        })
    return rows, {
        "file": path.name, "encoding": "gb18030", "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "raw_articles": len(grouped), "decode_error": decode_error,
        "removed": dict(removed), "damaged_ids": sorted(damaged),
        "split_by_date": {"train": "1998-01-05..16", "valid": "1998-01-19..21",
                          "test": "1998-01-22..27"},
        "page_labels": PAGE_LABELS, "label_names": LABEL_NAMES,
    }


def external_articles(path, limit):
    import jieba

    jieba.dt.tmp_dir = str(ROOT / "data")
    jieba.initialize()
    rows = []
    counts = Counter()
    seen = set()
    removed = Counter()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            title, label = line.rstrip("\n").rsplit("\t", 1)
            if counts[label] >= limit:
                continue
            title = unicodedata.normalize("NFKC", title)
            key = "".join(title.split()).lower()
            if key in seen:
                removed["duplicate"] += 1
                continue
            seen.add(key)
            tokens = [clean_token(w) for w in jieba.lcut(title, HMM=False)]
            tokens = [w for w in tokens if w]
            if len(tokens) < 3:
                removed["short"] += 1
                continue
            counts[label] += 1
            rows.append({"id": index, "title": title, "tokens": tokens})
    return rows, {
        "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size, "max_per_source_category": limit,
        "kept_per_source_category": dict(sorted(counts.items())),
        "articles": len(rows), "tokens": sum(len(r["tokens"]) for r in rows),
        "removed": dict(removed),
        "url": "https://raw.githubusercontent.com/649453932/"
               "Chinese-Text-Classification-Pytorch/master/THUCNews/data/train.txt",
        "note": "THUCNews title subset; source category labels are discarded before training.",
    }


def prepare(source, external, limit=5000):
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    january, january_stats = january_articles(source)
    titles, external_stats = external_articles(external, limit)
    # Exclude exact title overlaps, without using evaluation labels or fitting a vocabulary.
    january_titles = {"".join(r["title"].split()).lower() for r in january}
    before = len(titles)
    titles = [r for r in titles if "".join(r["title"].split()).lower() not in january_titles]
    external_stats["exact_january_title_overlaps_removed"] = before - len(titles)
    external_stats["articles_after_overlap_check"] = len(titles)
    external_stats["tokens_after_overlap_check"] = sum(len(r["tokens"]) for r in titles)
    write_jsonl(ROOT / "data/january.jsonl", january)
    write_jsonl(ROOT / "data/external_articles.jsonl", titles)
    with (ROOT / "data/embedding_sentences.txt").open("w", encoding="utf-8") as handle:
        for row in titles:
            handle.write(" ".join(row["tokens"]) + "\n")
    splits = {}
    for split in ("train", "valid", "test"):
        subset = [r for r in january if r["split"] == split]
        splits[split] = {
            "articles": len(subset), "tokens": sum(len(r["tokens"]) for r in subset),
            "labels": dict(Counter(r["label"] for r in subset)),
            "dates": sorted({r["date"] for r in subset}),
        }
    manifest = [{k: r[k] for k in ("id", "date", "page", "label", "split", "sha256")}
                for r in january]
    write_json(ROOT / "results/split_manifest.json", manifest)
    stats = {"external": external_stats, "january": january_stats, "splits": splits}
    write_json(ROOT / "results/data_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "人民日报语料" /
                        "1998-上半年-带音/1998-01-2003版-带音.txt")
    parser.add_argument("--external", type=Path, default=ROOT / "data/raw/thucnews_train.txt")
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    prepare(args.source, args.external, args.limit)
