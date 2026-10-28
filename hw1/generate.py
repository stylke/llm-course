"""通过KenLM给整个词表打分，进行贪心或temperature/top-k/top-p续写。"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from pathlib import Path

import kenlm

ROOT = Path(__file__).resolve().parent
PROMPT = "在阳光明媚的五月，我们学校胜利召开了"
# 手工按语料的粒度切分给定开头，不额外引入分词工具。
PROMPT_WORDS = ["在", "阳光", "明媚", "的", "五月", "，", "我们", "学校", "胜利", "召开", "了"]


def generate(model, vocabulary, *, prompt_words=None, temperature=1.0,
             top_k=50, top_p=1.0, greedy=False, seed=42, max_tokens=60):
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("要求temperature>0、top_k>=0、0<top_p<=1")
    prompt_words = PROMPT_WORDS if prompt_words is None else prompt_words
    rng = random.Random(seed)
    state, next_state = kenlm.State(), kenlm.State()
    model.BeginSentenceWrite(state)
    for word in prompt_words:
        model.BaseScore(state, word if word in model else "__RARE__", next_state)
        state, next_state = next_state, state
    # BOS、UNK不作为输出；EOS参与打分。排除UNK后对候选重新归一化。
    candidates = [w for w in vocabulary if w not in {"<s>", "<unk>", "__RARE__", "</s>"}] + ["</s>"]
    output = []
    stop_reason = "max_tokens"
    for _ in range(max_tokens):
        scores = [(model.BaseScore(state, word, next_state), word) for word in candidates]
        if greedy:
            _, selected = max(scores)
        else:
            ranked = (heapq.nlargest(top_k, scores) if top_k else
                      sorted(scores, reverse=True))
            peak = ranked[0][0]
            weights = [math.exp((score - peak) * math.log(10) / temperature)
                       for score, _ in ranked]
            if top_p < 1:
                target, cumulative, keep = sum(weights) * top_p, 0.0, 0
                for weight in weights:
                    cumulative += weight
                    keep += 1
                    if cumulative >= target:
                        break
                ranked, weights = ranked[:keep], weights[:keep]
            selected = rng.choices([w for _, w in ranked], weights=weights, k=1)[0]
        if selected == "</s>":
            stop_reason = "eos"
            break
        output.append(selected)
        model.BaseScore(state, selected, next_state)
        state, next_state = next_state, state
        if selected in {"。", "！", "？", "!", "?"}:
            stop_reason = "sentence_punctuation"
            break
    return {"seed": seed, "temperature": temperature, "top_k": top_k,
            "top_p": top_p, "greedy": greedy, "max_tokens": max_tokens,
            "prompt_tokens": list(prompt_words),
            "prompt_oov": [w for w in prompt_words if w not in model],
            "tokens": output, "text": "".join(prompt_words + output),
            "stop_reason": stop_reason}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=ROOT / "models" / "3gram.bin")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()
    vocab = json.loads((ROOT / "data" / "vocab.json").read_text(encoding="utf-8"))
    result = generate(kenlm.Model(str(args.model)), vocab,
                      temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, seed=args.seed, greedy=args.greedy,
                      max_tokens=args.max_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))
