#!/usr/bin/env python3
"""Perplexity for the GGUF schemes, measured separately from the energy sweep.

Separate on purpose. Reading per-position logits requires `logits_all=True`,
which costs roughly half the decode throughput (68.6 -> 32.7 tok/s measured on
this machine), so any energy figure taken under it would be wrong. The energy
sweeps therefore run with --no-quality and quality is measured here.

Every perplexity in the paper -- PyTorch and llama.cpp alike -- is computed over
the same corpus slice with the same tokenizer and the same window size, which is
what makes the cross-engine quality comparison meaningful even though the
cross-engine *energy* comparison is not.

Usage:  python3 kquant_perplexity.py [--corpus-chars 4000] [--stride 512]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import backends

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus-chars", type=int, default=4000,
                    help="must match the value used for the PyTorch runs, or "
                         "the perplexities are not comparable")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default=str(HERE / "runs" / "kquant" / "perplexity.json"))
    a = ap.parse_args()

    corpus = (HERE / "corpus.txt").read_text()[:a.corpus_chars]
    results: dict[str, dict] = {}

    for precision in backends.GGUF_PRECISIONS:
        backend = backends.make_backend("llamacpp", "x", precision,
                                        threads=a.threads, seed=0)
        backend.load()
        try:
            total, n = backend.logprobs(corpus, stride=a.stride)
            results[precision] = {
                "logprob_sum": total,
                "n_tokens": n,
                "perplexity": math.exp(-total / n) if n else float("nan"),
                "corpus_chars": a.corpus_chars,
                "stride": a.stride,
            }
            print(f"{precision:14} ppl={results[precision]['perplexity']:.3f} "
                  f"over {n} tokens", flush=True)
        finally:
            backend.unload()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
