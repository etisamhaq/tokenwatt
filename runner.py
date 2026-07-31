#!/usr/bin/env python3
"""Experiment runner: sweeps a configuration matrix and writes results.jsonl.

Each row is one repeat of one configuration, with energy, timing, memory and
quality. Analysis is deliberately a separate step (analyze.py) so the expensive
measurement is never re-run to change a plot.

Measurement discipline
----------------------
A laptop is a hostile place to measure energy, and results that ignore that get
dismissed. So:

  idle baseline   Measured PER CONFIGURATION, immediately after load and warmup
                  and under the same affinity mask, then subtracted to give net
                  energy. At small batch sizes idle draw is a large fraction of
                  the total, so gross joules would mostly measure "the machine
                  is switched on".

                  A single run-level baseline is not good enough: the non-core
                  part of the package (uncore, and the iGPU on integrated parts)
                  varies with whatever the desktop is doing, and subtracting a
                  baseline captured in a different system state produced net
                  package energy LOWER than net core energy in an earlier run -
                  arithmetically impossible for a subset domain. Per-config
                  baselines under matched thermal state and affinity fix that.

                  Net energy is recorded UNCLAMPED. A negative value means the
                  baseline exceeded the measurement and is a signal that the
                  baseline is wrong; clamping it to zero would hide exactly the
                  error this is meant to expose.
  thermal soak    A warmup generation per configuration, discarded. The first
                  run after load sees cold caches and an unthrottled turbo
                  budget; the throttled steady state is the honest number.
  repeats         Every configuration is run N times. analyze.py reports median
                  and IQR. A single laptop number is not a measurement.
  fixed inputs    Same prompts, same seed, same corpus for every configuration.

Energy is recorded as null rather than estimated when RAPL is unreadable, so a
run without permissions still yields valid timing and quality data and is never
silently passed off as having measured energy.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import backends
import energy as energy_mod
import quality as quality_mod
import topology

HERE = Path(__file__).resolve().parent

PROMPTS = [
    "Explain in two sentences why memory bandwidth limits language model inference.",
    "The capital of France is",
    "Write a short function that reverses a linked list.",
    "Summarise the causes of the 1929 financial crash.",
]


@dataclass
class Config:
    engine: str
    precision: str
    placement: str          # island name, "+SMT" variants, or "all"
    threads: int
    cpus: list[int] | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.engine}/{self.precision}/{self.placement}/t{self.threads}"


def placements() -> dict[str, list[int] | None]:
    """Named CPU placements available on this machine.

    For SMT islands both a one-thread-per-core and an all-hyperthreads variant
    are offered, because whether SMT helps memory-bound decode is exactly the
    kind of question this is for."""
    isl = topology.get_islands()
    out: dict[str, list[int] | None] = {"all": None}
    for name, tier in isl.items():
        if tier.smt:
            out[f"{name}-1t"] = tier.one_thread_per_core()
            out[f"{name}-smt"] = tier.cpus
        else:
            out[name] = tier.cpus
    return out


def build_matrix(engines: list[str], precisions: list[str],
                 place_names: list[str], thread_counts: list[int] | None
                 ) -> list[Config]:
    avail = placements()
    unknown = [p for p in place_names if p not in avail]
    if unknown:
        raise SystemExit(
            f"unknown placement(s) {unknown}. available: {sorted(avail)}")

    matrix: list[Config] = []
    for engine in engines:
        for prec in precisions:
            for pname in place_names:
                cpus = avail[pname]
                # Default thread count is the number of physical cores in the
                # placement -- more threads than cores just adds contention.
                width = len(cpus) if cpus else len(topology.online_cpus())
                for t in (thread_counts or [width]):
                    if t > width:
                        continue      # cannot use more threads than pinned CPUs
                    matrix.append(Config(engine, prec, pname, t, cpus))
    return matrix


def measure_idle(seconds: float) -> dict[str, float] | None:
    """Mean idle power per RAPL domain, in watts."""
    try:
        with energy_mod.EnergyMeter() as m:
            time.sleep(seconds)
        return m.watts
    except (PermissionError, RuntimeError):
        return None


def measure_idle_settled(seconds: float, settle: float = 15.0,
                         tolerance: float = 0.10, attempts: int = 3,
                         domain: str = "package-0"
                         ) -> tuple[dict[str, float] | None, bool]:
    """Idle power measured only once it has stopped moving.

    Returns (watts, stable). A baseline taken immediately after a burst of work
    captures the CPU decaying out of turbo rather than true idle, and how far it
    has decayed depends on timing nobody controls. Measured that way, per-config
    baselines on this machine ranged 5.5-13.2 W -- a 2.2x spread against gross
    power that varied only 10% -- and the resulting net energy was dominated by
    baseline artifact rather than signal. See runs/failed-perconfig-baseline/.

    So: wait `settle` seconds doing nothing, then take two consecutive windows
    and require them to agree within `tolerance` before believing either."""
    time.sleep(settle)
    prev = measure_idle(seconds)
    if prev is None:
        return None, False
    for _ in range(attempts - 1):
        cur = measure_idle(seconds)
        if cur is None:
            return prev, False
        a, b = prev.get(domain), cur.get(domain)
        if a and b and abs(a - b) / max(a, b) <= tolerance:
            return cur, True
        prev = cur
    return prev, False


def peak_rss_bytes() -> int:
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_one(cfg: Config, model: str, max_new_tokens: int, repeats: int,
            run_idle_watts: dict[str, float] | None, corpus: str,
            run_quality: bool, seed: int, needle_repeats: int = 40,
            measure_energy: bool = True, idle_seconds: float = 10.0,
            baseline_mode: str = "run"
            ) -> tuple[list[dict], dict[str, float] | None]:
    """Load one configuration, warm it up, baseline it, then measure `repeats`.

    Returns (rows, per-config idle watts) so the caller can report drift."""
    rows: list[dict] = []

    with topology.Affinity(cfg.cpus):
        backend = backends.make_backend(cfg.engine, model, cfg.precision,
                                        threads=cfg.threads, seed=seed)
        t_load = time.perf_counter()
        backend.load()
        load_s = time.perf_counter() - t_load

        # Thermal soak / cache warm. Discarded.
        backend.generate(PROMPTS[0], max_new_tokens=min(8, max_new_tokens))

        # Baseline. "run" reuses one baseline measured on a quiet machine before
        # any load; "per-config" re-measures here, but only after settling, since
        # an unsettled baseline is worse than a slightly stale one.
        idle_stable = True
        if not measure_energy:
            idle_watts = None
        elif baseline_mode == "per-config":
            idle_watts, idle_stable = measure_idle_settled(idle_seconds)
        else:
            idle_watts = run_idle_watts

        qual = None
        if run_quality:
            qual = quality_mod.evaluate(backend, corpus=corpus,
                                        needle_repeats=needle_repeats,
                                        run_needle=True)

        for rep in range(repeats):
            prompt = PROMPTS[rep % len(PROMPTS)]
            # Energy availability is fixed for the whole run (see main()).
            # Probing per repeat means a permission change mid-run silently
            # yields a file where some configurations have joules and others do
            # not, which makes the Pareto frontier compare incomparable things.
            meter = None
            if measure_energy:
                try:
                    meter = energy_mod.EnergyMeter()
                except (PermissionError, RuntimeError):
                    meter = None

            if meter is not None:
                with meter:
                    gen = backend.generate(prompt, max_new_tokens)
                joules = dict(meter.joules)
                window_s = meter.seconds
            else:
                t0 = time.perf_counter()
                gen = backend.generate(prompt, max_new_tokens)
                window_s = time.perf_counter() - t0
                joules = {}

            # Unclamped on purpose -- see the module docstring.
            net_joules = {}
            negative_domains = []
            if joules and idle_watts:
                for k, v in joules.items():
                    net = v - idle_watts.get(k, 0.0) * window_s
                    net_joules[k] = net
                    if net < 0:
                        negative_domains.append(k)

            total_tokens = gen.n_prompt_tokens + gen.n_new_tokens
            rows.append({
                "config": cfg.label,
                "engine": cfg.engine,
                "precision": cfg.precision,
                "placement": cfg.placement,
                "threads": cfg.threads,
                "repeat": rep,
                "prompt_index": rep % len(PROMPTS),
                "model": model,
                "load_s": round(load_s, 3),
                "param_bytes": backend.param_bytes,
                "peak_rss_bytes": peak_rss_bytes(),
                "n_prompt_tokens": gen.n_prompt_tokens,
                "n_new_tokens": gen.n_new_tokens,
                "prefill_s": gen.prefill_s,
                "decode_s": gen.decode_s,
                "decode_tok_per_s": gen.decode_tok_per_s,
                "prefill_tok_per_s": gen.prefill_tok_per_s,
                "decode_step_s_median": (statistics.median(gen.decode_step_s)
                                         if gen.decode_step_s else None),
                "kv_bytes_final": gen.kv_bytes_final,
                "kv_bytes_by_step": gen.kv_bytes_by_step,
                "window_s": window_s,
                "energy_measured": bool(joules),
                "joules_gross": joules or None,
                "joules_net": net_joules or None,
                "idle_watts_config": idle_watts or None,
                "baseline_mode": baseline_mode,
                "baseline_stable": idle_stable,
                "net_negative_domains": negative_domains or None,
                "joules_per_new_token": (
                    {k: v / gen.n_new_tokens for k, v in
                     (net_joules or joules).items()}
                    if (net_joules or joules) and gen.n_new_tokens else None),
                "total_tokens": total_tokens,
                "perplexity": qual.perplexity if qual else None,
                "ppl_tokens": qual.ppl_tokens if qual else None,
                "needle_pass": qual.needle_pass if qual else None,
                "needle_total": qual.needle_total if qual else None,
            })

        backend.unload()
    return rows, idle_watts


def machine_info() -> dict:
    isl = topology.get_islands()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "islands": {n: {"physical": t.n_physical, "logical": t.n_logical,
                        "max_ghz": round(t.max_khz / 1e6, 2), "cpus": t.cpus}
                    for n, t in isl.items()},
        "rapl_readable": energy_mod.check_readable()[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--engines", default="cpu")
    ap.add_argument("--precisions", default="fp32,bf16,int8-dynamic")
    ap.add_argument("--placements", default="all",
                    help="comma list, or 'ALL' to sweep every island")
    ap.add_argument("--threads", default="",
                    help="comma list of thread counts; default = cores in placement")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--baseline", choices=("run", "per-config"), default="run",
                    help="'run' (default) uses one baseline measured before any "
                         "load; 'per-config' re-measures per configuration after "
                         "a settle period. per-config is more principled but "
                         "noisier in practice - see FINDINGS.md")
    ap.add_argument("--corpus-chars", type=int, default=4000)
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--needle-repeats", type=int, default=40,
                    help="filler sentences in the needle probe (context length)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--list-placements", action="store_true")
    a = ap.parse_args()

    if a.list_placements:
        for name, cpus in sorted(placements().items()):
            print(f"{name:10s} {cpus if cpus else 'unpinned (all cpus)'}")
        return 0

    place_names = (sorted(k for k in placements() if k != "all")
                   if a.placements == "ALL" else a.placements.split(","))
    thread_counts = [int(x) for x in a.threads.split(",")] if a.threads else None

    matrix = build_matrix(a.engines.split(","), a.precisions.split(","),
                          place_names, thread_counts)
    if not matrix:
        raise SystemExit("empty configuration matrix")

    info = machine_info()
    print(f"machine  : {info['processor'] or platform.machine()}")
    print(f"islands  : "
          + ", ".join(f"{n}({v['physical']}c/{v['logical']}t)"
                      for n, v in info["islands"].items()))
    print(f"RAPL     : {'readable' if info['rapl_readable'] else 'NOT readable'}")
    if not info["rapl_readable"]:
        print("           energy will be recorded as null - see README to fix")
    print(f"configs  : {len(matrix)} x {a.repeats} repeats")

    # One decision, applied uniformly. If RAPL is readable a baseline is
    # mandatory: without it only gross joules exist, and gross joules at these
    # batch sizes are dominated by the machine simply being powered on.
    measure_energy = info["rapl_readable"]
    idle = measure_idle(a.idle_seconds) if measure_energy else None
    if measure_energy and not idle:
        raise SystemExit("RAPL readable but idle baseline failed - aborting "
                         "rather than record gross-only energy")
    if idle:
        print("idle     : " + ", ".join(f"{k}={v:.2f}W" for k, v in
                                        sorted(idle.items())))
        print(f"baseline : {a.baseline}")

    corpus = quality_mod.load_corpus(max_chars=a.corpus_chars)
    baselines: dict[str, dict[str, float]] = {}
    out_path = Path(a.out)
    with out_path.open("w") as f:
        f.write(json.dumps({"_meta": True, "machine": info,
                            "idle_watts_reference": idle, "model": a.model,
                            "max_new_tokens": a.max_new_tokens,
                            "repeats": a.repeats, "seed": a.seed}) + "\n")
        for i, cfg in enumerate(matrix, 1):
            print(f"[{i}/{len(matrix)}] {cfg.label} ... ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                rows, cfg_idle = run_one(
                    cfg, a.model, a.max_new_tokens, a.repeats,
                    idle, corpus, not a.no_quality, a.seed,
                    needle_repeats=a.needle_repeats,
                    measure_energy=measure_energy,
                    idle_seconds=a.idle_seconds,
                    baseline_mode=a.baseline)
            except NotImplementedError as e:
                print(f"skipped ({e})")
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if cfg_idle:
                baselines[cfg.label] = cfg_idle
            med = statistics.median(r["decode_tok_per_s"] for r in rows)
            neg = sum(1 for r in rows if r.get("net_negative_domains"))
            warn = f"  [{neg} repeat(s) with negative net energy]" if neg else ""
            print(f"{time.perf_counter() - t0:5.1f}s  {med:5.1f} tok/s{warn}")

    # Baseline drift across configurations is diagnostic: a wide spread means
    # the machine was not in a comparable state throughout, and absolute
    # joules-per-token figures should be treated with corresponding suspicion.
    if baselines:
        dom = "package-0"
        vals = [b[dom] for b in baselines.values() if dom in b]
        if len(vals) > 1:
            lo, hi = min(vals), max(vals)
            spread = (hi - lo) / statistics.median(vals) * 100
            print(f"\nbaseline {dom}: {lo:.2f}-{hi:.2f} W across "
                  f"{len(vals)} configs (spread {spread:.0f}% of median)")
            if spread > 25:
                print("  WARNING: wide baseline drift - absolute joule figures "
                      "are unreliable, though relative ordering may survive")

    print(f"\nwrote {out_path}")
    print(f"next: python3 analyze.py --results {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
