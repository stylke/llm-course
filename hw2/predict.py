"""Predict the four broad page categories with the trained embedding pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from gensim.models import FastText
import jieba
import joblib

from prepare import LABEL_NAMES, clean_token
from run_experiment import average_vectors

ROOT = Path(__file__).resolve().parent


def predict(text, segmented=False):
    import json

    jieba.dt.tmp_dir = str(ROOT / "data")
    words = text.split() if segmented else jieba.lcut(text, HMM=False)
    words = [clean_token(w) for w in words]
    words = [w for w in words if w]
    if not words:
        raise ValueError("Input must contain at least one content token")
    model = FastText.load(str(ROOT / "models/ft_adapted.gensim"))
    idf = json.loads((ROOT / "models/idf.json").read_text(encoding="utf-8"))
    clf = joblib.load(ROOT / "models/ft_adapted_idf_4class.joblib")
    features, _ = average_vectors([{"tokens": words}], model.wv, idf)
    probs = clf.predict_proba(features)[0]
    for label, probability in sorted(zip(clf.classes_, probs), key=lambda x: -x[1]):
        print(f"{LABEL_NAMES[label]}\t{probability:.4f}")
    print("tokens:", " / ".join(words))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--segmented", action="store_true",
                        help="Use space-separated tokens instead of jieba")
    args = parser.parse_args()
    predict(args.text, args.segmented)
