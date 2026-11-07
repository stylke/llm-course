"""Restore punctuation in an unpunctuated Chinese text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference import Decoder
from prepare import ROOT, label_text, render


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--merge", choices=("mean", "center"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if bool(args.text) == bool(args.file):
        parser.error("Supply either text or --file")
    source = args.file.read_text(encoding="utf-8") if args.file else args.text
    text, _ = label_text(source)
    decoder = Decoder(device=args.device)
    selection = ROOT / "results/merge_selection.json"
    merge = args.merge or (json.loads(selection.read_text(encoding="utf-8"))["chosen"]
                           if selection.exists() else "center")
    labels, _ = decoder.fixed([text], batch_size=args.batch_size, merge=merge)
    restored = render(text, labels[0])
    if args.output:
        args.output.write_text(restored + "\n", encoding="utf-8")
    print(restored)
