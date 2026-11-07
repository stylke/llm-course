"""Run experiment stages and retain their actual console output."""
from __future__ import annotations

import argparse
from datetime import datetime
import subprocess
import sys

from prepare import ROOT

STAGES = {
    "prepare": ["prepare.py"],
    "download": ["download_model.py"],
    "train": ["train.py"],
    "calibrate": ["calibrate.py"],
    "evaluate": ["evaluate.py"],
    "test": ["-m", "unittest", "discover", "-s", str(ROOT), "-p", "test_*.py", "-v"],
    "analyze": ["analyze.py"],
}


def run(stage, extra):
    arguments = STAGES[stage]
    command = [sys.executable, str(ROOT / arguments[0]), *arguments[1:]] \
        if arguments[0].endswith(".py") else [sys.executable, *arguments]
    command.extend(extra)
    log = ROOT / f"results/logs/{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        header = f"{datetime.now().isoformat(timespec='seconds')}\n{' '.join(command)}\n"
        handle.write(header)
        handle.flush()
        print(header, end="", flush=True)
        process = subprocess.Popen(command, cwd=ROOT.parent, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        footer = f"\nexit_code={returncode}\n"
        handle.write(footer)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    args, extra = parser.parse_known_args()
    run(args.stage, extra)
