"""Download a pinned, pretrained Chinese BERT checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

from prepare import ROOT, write_json

MODEL_ID = "uer/chinese_roberta_L-4_H-256"
REVISION = "68d674689f3c9d425e5436149c8acece077187f1"
WEIGHT_SHA256 = "c60f6fde023a817b263dab52a4d0767ae139be98971da4a1a070c695ad24612e"
FILES = ["config.json", "vocab.txt", "tokenizer_config.json",
         "special_tokens_map.json", "pytorch_model.bin", "README.md"]


def download(endpoint):
    directory = ROOT / "models/pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results/pretrained_manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected = {item["name"]: item["sha256"] for item in old.get("files", [])}
    expected.setdefault("pytorch_model.bin", WEIGHT_SHA256)
    files = []
    for name in FILES:
        target = directory / name
        url = f"{endpoint.rstrip('/')}/{MODEL_ID}/resolve/{REVISION}/{name}"
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            partial = target.with_suffix(target.suffix + ".part")
            subprocess.run(["curl", "--location", "--fail", "--retry", "3",
                            "--connect-timeout", "30", "--max-time", "600",
                            url, "--output", str(partial)], check=True)
            partial.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if name in expected and digest != expected[name]:
            raise ValueError(f"SHA256 mismatch: {name}")
        files.append({"name": name, "bytes": target.stat().st_size,
                      "sha256": digest, "download_url": url})
        print(f"{name}: {target.stat().st_size:,} bytes {digest}", flush=True)
    write_json(manifest_path, {"model_id": MODEL_ID, "revision": REVISION, "files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    download(parser.parse_args().endpoint)
