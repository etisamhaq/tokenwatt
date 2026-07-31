#!/usr/bin/env python3
"""Output quality, so energy savings can be priced.

An energy number on its own is meaningless for inference: you can always spend
fewer joules by producing worse output. The whole point of the study is the
exchange rate between the two, which requires a quality measure that is
comparable across configurations.

Two measures are used, because each hides something the other catches:

  perplexity   Cheap, continuous, and sensitive to small degradations. But it
               scores the model's fit to a corpus, not whether the model does
               anything useful, and it is not comparable across tokenizers.

  needle       A retrieval probe: a fact is buried at a known depth in filler
               text and the model is asked for it. Coarse (pass/fail) but it
               catches the failure mode perplexity misses -- a quantized model
               whose perplexity barely moves while its ability to attend to
               distant context collapses.

Perplexity comparisons here are only ever made BETWEEN precisions of the SAME
model, on the SAME corpus, with the SAME tokenizer. Cross-model perplexity
comparison would be meaningless and the runner never does it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "corpus.txt"

# Needle probes: (filler_repeats, depth_fraction, fact, question, expected)
# Depth is where in the filler the fact is inserted, as a fraction of length.
NEEDLE_FACT = "The secret access code for the north gate is 47823."
NEEDLE_ASK = ("What is the secret access code for the north gate? "
              "Reply with only the number.")
# Plain-continuation fallback for base models with no chat template.
NEEDLE_QUESTION = ("\n\nQuestion: What is the secret access code for the north "
                   "gate?\nAnswer:")
NEEDLE_EXPECTED = "47823"


@dataclass
class QualityResult:
    perplexity: float
    ppl_tokens: int
    needle_pass: int
    needle_total: int

    @property
    def needle_rate(self) -> float:
        return self.needle_pass / self.needle_total if self.needle_total else 0.0


def load_corpus(path: Path | None = None, max_chars: int | None = None) -> str:
    p = path or DEFAULT_CORPUS
    if not p.exists():
        raise FileNotFoundError(
            f"corpus not found at {p}. Provide --corpus or restore corpus.txt.")
    text = p.read_text(encoding="utf-8")
    return text[:max_chars] if max_chars else text


def perplexity(backend, text: str, stride: int = 512) -> tuple[float, int]:
    """exp(mean negative log-likelihood per token)."""
    total_ll, n = backend.logprobs(text, stride=stride)
    if n == 0:
        return float("nan"), 0
    return math.exp(-total_ll / n), n


def _as_chat(backend, user_text: str) -> str:
    """Wrap `user_text` in the model's chat template when it has one.

    This is not cosmetic. Given a bare continuation prompt, an instruction-tuned
    model treats the filler as text to continue and simply produces more filler,
    scoring 0 regardless of whether it could retrieve the fact. With the template
    applied the same model and same context answers correctly. Measuring the
    prompt format instead of the model's retrieval ability would have made every
    needle number in this study meaningless."""
    tok = getattr(backend, "tok", None)
    if tok is None or not getattr(tok, "chat_template", None):
        return None
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return None


def _build_needle(filler_sentence: str, repeats: int, depth: float) -> str:
    """Insert the fact at `depth` through `repeats` copies of filler."""
    n_before = max(0, min(repeats, int(round(repeats * depth))))
    parts = [filler_sentence] * n_before + [NEEDLE_FACT] + \
            [filler_sentence] * (repeats - n_before)
    return " ".join(parts)


def needle_probe(backend, repeats: int = 40,
                 depths: tuple[float, ...] = (0.1, 0.5, 0.9),
                 max_new_tokens: int = 12) -> tuple[int, int]:
    """Ask for a buried fact at several depths. Returns (passes, total).

    Depth is varied because degradation is rarely uniform: quantization often
    damages attention to the middle of the context first, which a single
    end-of-context probe would miss entirely."""
    filler = ("The garden was quiet and the afternoon light fell across the "
              "gravel path without hurry.")
    passes = 0
    for depth in depths:
        context = _build_needle(filler, repeats, depth)
        chat = _as_chat(backend, context + "\n\n" + NEEDLE_ASK)
        prompt = chat if chat is not None else context + NEEDLE_QUESTION
        out = backend.generate(prompt, max_new_tokens=max_new_tokens)
        if NEEDLE_EXPECTED in out.text:
            passes += 1
    return passes, len(depths)


def evaluate(backend, corpus: str | None = None, stride: int = 512,
             needle_repeats: int = 40, run_needle: bool = True) -> QualityResult:
    text = corpus if corpus is not None else load_corpus()
    ppl, n = perplexity(backend, text, stride=stride)
    if run_needle:
        p, t = needle_probe(backend, repeats=needle_repeats)
    else:
        p, t = 0, 0
    return QualityResult(perplexity=ppl, ppl_tokens=n, needle_pass=p,
                         needle_total=t)


if __name__ == "__main__":
    import argparse
    import backends

    ap = argparse.ArgumentParser(description="Quality of one configuration")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--precision", default="fp32", choices=backends.PRECISIONS)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--corpus-chars", type=int, default=6000)
    ap.add_argument("--no-needle", action="store_true")
    a = ap.parse_args()

    b = backends.make_backend("cpu", a.model, a.precision, threads=a.threads)
    b.load()
    q = evaluate(b, corpus=load_corpus(max_chars=a.corpus_chars),
                 run_needle=not a.no_needle)
    print(f"{a.model}  {a.precision}")
    print(f"  perplexity  {q.perplexity:.4f}  ({q.ppl_tokens} tokens)")
    if q.needle_total:
        print(f"  needle      {q.needle_pass}/{q.needle_total} "
              f"({q.needle_rate * 100:.0f}%)")
