"""补充检查：未见n-gram比例、条件分布、生成复现与评分一致性。"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import kenlm

from generate import PROMPT_WORDS, generate

ROOT = Path(__file__).resolve().parent


def grams(line, order):
    words = ["<s>"] + line.split() + ["</s>"]
    for i in range(1, len(words)):
        yield tuple(words[max(0, i - order + 1):i + 1])


def state_after(model, words):
    state, out = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state)
    for word in words:
        model.BaseScore(state, word, out)
        state, out = out, state
    return state


def main():
    data, results = ROOT / "data", ROOT / "results"
    summary = {"unseen_mle_events": [], "conditional_top10": {}, "checks": []}
    for order in [2, 3, 4]:
        events = Counter(g for line in (data / "test.txt").read_text(encoding="utf-8").splitlines()
                         for g in grams(line, order))
        total = sum(events.values())
        unseen = set(events)
        with (data / "train.txt").open(encoding="utf-8") as stream:
            for line in stream:
                for g in grams(line, order):
                    unseen.discard(g)
        missing = sum(events[g] for g in unseen)
        summary["unseen_mle_events"].append({"order": order, "total": total,
                                            "unseen": missing, "rate": missing / total})
        print(f"{order}-gram 未见事件比例：{missing / total:.4%}", flush=True)
    vocab = json.loads((data / "vocab.json").read_text(encoding="utf-8"))
    for order in [2, 3, 4]:
        model = kenlm.Model(str(ROOT / "models" / f"{order}gram.bin"))
        state, out = state_after(model, PROMPT_WORDS), kenlm.State()
        scored = [(model.BaseScore(state, w, out), w) for w in vocab + ["__RARE__", "<unk>", "</s>"]]
        probability_sum = sum(10 ** s for s, w in scored)
        assert abs(probability_sum - 1) < 1e-5, probability_sum
        summary["conditional_top10"][str(order)] = [
            {"word": w, "probability": 10 ** s} for s, w in sorted(scored, reverse=True)[:10]]
        other = state_after(model, ["昨天", "政府", "胜利", "召开", "了"])
        max_delta = max(abs(model.BaseScore(state, w, out) - model.BaseScore(other, w, out))
                        for w in vocab)
        assert max_delta == 0  # 三种阶数都只保留相同的末尾局部上下文。
        sentence = "我们 学校 召开 了 会议 。"
        state = kenlm.State()
        model.BeginSentenceWrite(state)
        step_score = 0.0
        for word in sentence.split() + ["</s>"]:
            step_score += model.BaseScore(state, word, out)
            state, out = out, state
        assert math.isclose(step_score, model.score(sentence), abs_tol=2e-5)
        sample = generate(model, vocab, seed=42, temperature=0.7, top_k=20)
        saved = json.loads((results / "generations.json").read_text(encoding="utf-8"))
        existing = next(s for s in saved if s["order"] == order and s["seed"] == 42
                        and s["config"] == "T0.7_k20")
        assert sample["tokens"] == existing["tokens"]
        assert not sample["prompt_oov"]
        summary["checks"].append({"order": order, "probability_sum": probability_sum,
                                  "different_prefix_same_suffix_max_log10_delta": max_delta,
                                  "incremental_score_matches_sentence_score": True,
                                  "saved_generation_reproduced": True,
                                  "prompt_has_no_oov": True})
    (results / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    print("补充分析及一致性检查通过。")


if __name__ == "__main__":
    main()
