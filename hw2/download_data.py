"""Download the news title file, checking the exact bytes used in this homework."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = ("https://raw.githubusercontent.com/649453932/"
       "Chinese-Text-Classification-Pytorch/master/THUCNews/data/train.txt")
EXPECTED_SHA256 = "ba6fcbbb8fb809a9eb2be443b6b4db533c147861d94ebe0931b8c6092d8abe94"


def download():
    destination = ROOT / "data/raw/thucnews_train.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == EXPECTED_SHA256:
        print("Existing news file matches the expected SHA256.")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            request = urllib.request.Request(URL, headers={"User-Agent": "hw2-course-experiment"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    handle.write(block)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Source file has changed or download is incomplete: {actual}")
        temporary.replace(destination)
        print(f"Downloaded {destination.stat().st_size} bytes; SHA256 verified.")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    download()
