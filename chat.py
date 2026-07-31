#!/usr/bin/env python3
"""Interactive chat with a model, at a chosen precision and core placement.

Not part of the measurement pipeline -- this is for feeling what the numbers in
RESULTS.md actually mean. Running the same prompt at `--precision bf16` and then
`--precision int8-dynamic` makes the perplexity difference visceral in a way a
table does not.

Examples
--------
    python3 chat.py                                  # SmolLM2-135M, fp32, all cores
    python3 chat.py --model Qwen/Qwen2.5-0.5B-Instruct --precision bf16
    python3 chat.py --precision int8-dynamic         # see the damage first-hand
    python3 chat.py --placement E --threads 8        # pinned to efficiency cores
    python3 chat.py --prompt "Explain KV caching." --max-new-tokens 80

Conversation history is not kept: each prompt is independent. That keeps the
timings comparable to what runner.py measures.
"""

from __future__ import annotations

import argparse
import sys

import backends
import energy as energy_mod
import topology


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--precision", default="fp32", choices=backends.PRECISIONS)
    ap.add_argument("--placement", default="all",
                    help="core island: all, P-1t, P-smt, E, LP-E "
                         "(see: python3 runner.py --list-placements)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt", default=None,
                    help="answer one prompt and exit instead of looping")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="send the raw prompt, as a base model would see it")
    a = ap.parse_args()

    import runner  # for the shared placement map
    places = runner.placements()
    if a.placement not in places:
        print(f"unknown placement {a.placement!r}; available: "
              f"{', '.join(sorted(places))}", file=sys.stderr)
        return 2
    cpus = places[a.placement]
    threads = a.threads or (len(cpus) if cpus else None)

    print(f"loading {a.model} ({a.precision}) ...", flush=True)
    with topology.Affinity(cpus):
        b = backends.make_backend("cpu", a.model, a.precision, threads=threads)
        b.load()

        gb = b.param_bytes / 1e9
        print(f"ready — {gb:.2f} GB resident, placement={a.placement}"
              + (f", {threads} threads" if threads else ""))
        rapl_ok, _ = energy_mod.check_readable()
        if not rapl_ok:
            print("      (energy not shown: RAPL unreadable — see README)")
        print()

        def answer(text: str) -> None:
            prompt = text
            if not a.no_chat_template:
                import quality
                chat = quality._as_chat(b, text)
                if chat is not None:
                    prompt = chat

            meter = None
            if rapl_ok:
                try:
                    meter = energy_mod.EnergyMeter()
                except (PermissionError, RuntimeError):
                    meter = None

            if meter is not None:
                with meter:
                    r = b.generate(prompt, a.max_new_tokens)
            else:
                r = b.generate(prompt, a.max_new_tokens)

            print(r.text.strip())
            stats = (f"\n[{r.n_new_tokens} tok · "
                     f"prefill {r.prefill_s * 1000:.0f} ms · "
                     f"decode {r.decode_tok_per_s:.1f} tok/s · "
                     f"KV {r.kv_bytes_final / 1024:.0f} KiB")
            if meter is not None and r.n_new_tokens:
                j = meter.joules.get("package-0")
                if j is not None:
                    stats += f" · {j / r.n_new_tokens:.3f} J/tok"
            print(stats + "]\n")

        if a.prompt:
            answer(a.prompt)
            return 0

        print("Type a prompt, or Ctrl-D / 'quit' to exit.\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text.lower() in {"quit", "exit"}:
                return 0
            if text:
                answer(text)


if __name__ == "__main__":
    sys.exit(main())
