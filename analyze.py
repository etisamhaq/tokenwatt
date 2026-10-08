#!/usr/bin/env python3
"""Aggregate results.jsonl into statistics, a Pareto frontier and plots.

Kept separate from runner.py so plots can be regenerated without repeating the
expensive measurement.

Reporting choices
-----------------
Median and IQR rather than mean and standard deviation. Laptop timing
distributions are right-skewed -- a thermal event or a background process adds a
long tail but nothing pulls the other way -- so the mean drifts upward with
outliers while the median describes the typical run.

The Pareto frontier is over (energy per token, quality loss), both
minimised. A configuration on the frontier is one where you cannot reduce energy
without losing quality. Everything off the frontier is dominated and should
simply never be chosen, which is the actionable output of the whole study.

When RAPL was unreadable the frontier falls back to (seconds per token, quality)
and says so, rather than quietly presenting latency as though it were energy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Whole-package is the number that matters for a device: it includes the uncore
# and, on integrated parts, the iGPU. Override with --domain to isolate the cores.
DEFAULT_DOMAIN = "package-0"
ENERGY_DOMAIN = DEFAULT_DOMAIN

# Validated default categorical palette, light mode. Only the first three slots
# clear the all-pairs colour-vision-deficiency floors, which is what a scatter
# needs -- so colour carries the 3-level variable (precision) and marker shape
# carries the 4-level one (core placement). Identity is therefore never
# colour-alone, and the palette is never extended past its validated cap.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
PRECISION_COLOR = {
    "fp32": "#2a78d6",           # slot 1, blue
    "bf16": "#eb6834",           # slot 2, orange
    "int8-dynamic": "#1baf7a",   # slot 3, aqua
}
PLACEMENT_MARKER = {
    "P-1t": "o", "P-smt": "s", "E": "^", "LP-E": "D", "all": "P",
}


def load(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            meta = obj
        else:
            rows.append(obj)
    return meta, rows


def iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def bootstrap_ci(values: list[float], statistic=statistics.median,
                 confidence: float = 0.95, resamples: int = 10_000,
                 seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap interval for `statistic`.

    IQR describes the spread of the sample; it says nothing about how precisely
    the sample locates the median. Those are different questions and the claims
    in FINDINGS.md are about the second. Returns None below 3 values, where a
    resampled interval would be arithmetic rather than evidence."""
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = []
    for _ in range(resamples):
        stats.append(statistic([values[rng.randrange(n)] for _ in range(n)]))
    stats.sort()
    lo_i = int((1.0 - confidence) / 2.0 * resamples)
    hi_i = min(resamples - 1, int((1.0 + confidence) / 2.0 * resamples))
    return stats[lo_i], stats[hi_i]


def by_prompt(rows: list[dict], key) -> dict[int, float]:
    """Median of `key` per prompt_index, collapsing true repeats.

    This is the unit of the paired analysis: repeats are exchangeable
    measurements of one thing, prompts are not."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        v = key(r)
        if v is not None:
            buckets[r.get("prompt_index", 0)].append(v)
    return {p: statistics.median(v) for p, v in buckets.items() if v}


def paired_ratio(rows_a: list[dict], rows_b: list[dict], key,
                 confidence: float = 0.95) -> dict | None:
    """Ratio a/b computed within each prompt, then aggregated across prompts.

    Every configuration sees the same prompts, so differencing inside a prompt
    removes prompt-to-prompt variation -- which on this workload is comparable to
    the effects being measured, so the unpaired comparison buries a real effect
    in the prompt spread. Reports the geometric mean of the per-prompt ratios
    with a bootstrap interval, and the raw ratios, so a reader can see whether an
    effect is consistent or carried by a single prompt."""
    a, b = by_prompt(rows_a, key), by_prompt(rows_b, key)
    shared = sorted(set(a) & set(b))
    ratios = [a[p] / b[p] for p in shared if b[p]]
    if not ratios:
        return None
    gmean = math.exp(statistics.fmean(math.log(r) for r in ratios))
    ci = bootstrap_ci([math.log(r) for r in ratios], statistic=statistics.fmean,
                      confidence=confidence)
    return {
        "n_prompts": len(ratios),
        "ratio": gmean,
        "ci": (math.exp(ci[0]), math.exp(ci[1])) if ci else None,
        "per_prompt": {p: a[p] / b[p] for p in shared if b[p]},
        "consistent": all(r > 1 for r in ratios) or all(r < 1 for r in ratios),
    }


def determinism_report(rows: list[dict]) -> list[dict]:
    """Where identical (precision, prompt) gave different output by placement.

    Greedy decode at a fixed seed should be placement-invariant. On this SoC it
    is not always: islands can dispatch to different kernels, the arithmetic
    differs, and the generated text diverges. Downstream that appears only as an
    unexplained perplexity difference, which reads as noise. Comparing digests
    names it instead."""
    groups: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        sha = r.get("output_sha")
        if sha:
            groups[(r["precision"], r.get("prompt_index", 0))][r["placement"]].add(sha)

    out = []
    for (prec, prompt), per_place in sorted(groups.items()):
        all_shas = set().union(*per_place.values())
        if len(all_shas) > 1:
            out.append({
                "precision": prec,
                "prompt_index": prompt,
                "placements": {p: sorted(s) for p, s in sorted(per_place.items())},
                "n_distinct": len(all_shas),
                "unstable_within_placement": sorted(
                    p for p, s in per_place.items() if len(s) > 1),
            })
    return out


def summarise(rows: list[dict]) -> list[dict]:
    """One record per configuration, with median and IQR over repeats."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["config"]].append(r)

    out = []
    for cfg, rs in groups.items():
        tok_s = [r["decode_tok_per_s"] for r in rs]
        prefill = [r["prefill_s"] for r in rs]

        jpt_vals = [r["joules_per_new_token"][ENERGY_DOMAIN]
                    for r in rs
                    if r.get("joules_per_new_token")
                    and ENERGY_DOMAIN in r["joules_per_new_token"]]

        # Net power is the quantity that makes the island result interpretable:
        # if it is flat across placements then energy differences are duration
        # differences, which is a mechanism rather than an observation.
        watt_vals = [r["joules_net"][ENERGY_DOMAIN] / r["window_s"]
                     for r in rs
                     if r.get("joules_net") and r.get("window_s")
                     and ENERGY_DOMAIN in r["joules_net"]]

        first = rs[0]
        out.append({
            "config": cfg,
            "engine": first["engine"],
            "precision": first["precision"],
            "placement": first["placement"],
            "threads": first["threads"],
            "n": len(rs),
            "decode_tok_per_s_med": statistics.median(tok_s),
            "decode_tok_per_s_iqr": iqr(tok_s),
            "prefill_s_med": statistics.median(prefill),
            "s_per_token_med": 1.0 / statistics.median(tok_s) if statistics.median(tok_s) else None,
            "joules_per_token_med": statistics.median(jpt_vals) if jpt_vals else None,
            "joules_per_token_iqr": iqr(jpt_vals) if jpt_vals else None,
            "joules_per_token_ci": bootstrap_ci(jpt_vals) if jpt_vals else None,
            "decode_tok_per_s_ci": bootstrap_ci(tok_s),
            "net_watts_med": statistics.median(watt_vals) if watt_vals else None,
            "net_watts_ci": bootstrap_ci(watt_vals) if watt_vals else None,
            "n_prompts": len({r.get("prompt_index", 0) for r in rs}),
            "energy_measured": bool(jpt_vals),
            "param_bytes": first["param_bytes"],
            "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rs),
            "kv_bytes_final": first["kv_bytes_final"],
            "perplexity": first.get("perplexity"),
            "needle_pass": first.get("needle_pass"),
            "needle_total": first.get("needle_total"),
            "idle_w": ((first.get("idle_watts_config") or {})
                       .get(ENERGY_DOMAIN)),
            # Split by domain. The uncore baseline is ~0.25 W and dominated by
            # noise, so it goes slightly negative routinely and says nothing
            # about the result. Reporting one combined count made a clean
            # dataset read as 237-of-240 broken, when the domain every number
            # is computed from never went negative at all.
            "n_negative_reported": sum(
                1 for r in rs
                if ENERGY_DOMAIN in (r.get("net_negative_domains") or [])),
            "n_negative_other": sum(
                1 for r in rs
                if (r.get("net_negative_domains")
                    and ENERGY_DOMAIN not in r["net_negative_domains"])),
        })
    out.sort(key=lambda d: (d["precision"], d["placement"], d["threads"]))
    return out


def baseline_mode_of(rows: list[dict]) -> str:
    """Which baseline method the DATA actually used.

    Read from the rows rather than assumed, because a report that asserts a
    methodology the data did not use is worse than one that says nothing. Rows
    predating the `baseline_mode` field came from the single run-level baseline."""
    modes = {r.get("baseline_mode") for r in rows if r.get("baseline_mode")}
    if not modes:
        return "run"
    if len(modes) > 1:
        return "mixed(" + ",".join(sorted(modes)) + ")"
    return modes.pop()


def baseline_report(summary: list[dict]) -> str | None:
    """Spread of the per-configuration idle baseline.

    A wide spread means the machine was not in a comparable state across
    configurations, which puts a systematic error on every absolute
    joules-per-token figure. The relative ordering of configurations usually
    survives it -- large duration differences dominate -- but a gap of a few
    percent between two configurations does not."""
    vals = [s["idle_w"] for s in summary if s.get("idle_w") is not None]
    if len(vals) < 2:
        return None
    lo, hi, med = min(vals), max(vals), statistics.median(vals)
    spread = (hi - lo) / med * 100 if med else 0.0
    line = (f"Idle baseline for `{ENERGY_DOMAIN}` across {len(vals)} "
            f"configurations: {lo:.2f}-{hi:.2f} W, median {med:.2f} W "
            f"(spread {spread:.0f}% of median).")
    if spread > 25:
        line += (" **Wide drift** — absolute joule figures carry a systematic "
                 "error of about this size; treat differences smaller than it "
                 "as unresolved.")
    return line


def quality_loss(summary: list[dict]) -> dict[str, float]:
    """Relative perplexity increase against the best (lowest) configuration.

    Expressed as a fraction so it reads as 'you paid 6% quality for that'.
    Configurations without a perplexity measurement are omitted."""
    ppls = {s["config"]: s["perplexity"] for s in summary
            if s.get("perplexity") is not None}
    if not ppls:
        return {}
    best = min(ppls.values())
    return {c: (p - best) / best for c, p in ppls.items()}


def pareto(points: list[tuple[str, float, float]]) -> list[str]:
    """Non-dominated configs. points = (label, cost, quality_loss), both minimised."""
    frontier = []
    for label, c, q in points:
        dominated = any(
            (oc <= c and oq <= q) and (oc < c or oq < q)
            for ol, oc, oq in points if ol != label
        )
        if not dominated:
            frontier.append(label)
    return frontier


def fmt_table(summary: list[dict], loss: dict[str, float],
              frontier: set[str], energy: bool) -> str:
    cost_hdr = "J/token [95% CI]" if energy else "s/token"
    lines = [
        f"| config | n | tok/s [95% CI] | {cost_hdr} | perplexity | Δquality | needle | on frontier |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        cost = (s["joules_per_token_med"] if energy else s["s_per_token_med"])
        if cost is None:
            cost_s = "-"
        elif energy and s.get("joules_per_token_ci"):
            lo, hi = s["joules_per_token_ci"]
            cost_s = f"{cost:.4f} [{lo:.4f}, {hi:.4f}]"
        else:
            cost_s = f"{cost:.4f}"
        tps = f"{s['decode_tok_per_s_med']:.1f}"
        if s.get("decode_tok_per_s_ci"):
            lo, hi = s["decode_tok_per_s_ci"]
            tps += f" [{lo:.1f}, {hi:.1f}]"
        ppl = f"{s['perplexity']:.3f}" if s.get("perplexity") is not None else "-"
        dq = f"{loss[s['config']] * 100:+.1f}%" if s["config"] in loss else "-"
        nd = (f"{s['needle_pass']}/{s['needle_total']}"
              if s.get("needle_total") else "-")
        lines.append(
            f"| `{s['config']}` | {s['n']} | {tps} | {cost_s} | {ppl} | {dq} | {nd} "
            f"| {'yes' if s['config'] in frontier else ''} |"
        )
    return "\n".join(lines)


def _fmt_ratio(res: dict) -> str:
    ci = (f" [{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]" if res["ci"] else "")
    flag = "" if res["consistent"] else " ⚠ sign varies by prompt"
    return f"{res['ratio']:.3f}x{ci}{flag}"


def paired_section(rows: list[dict], energy: bool) -> str:
    """Paired per-prompt comparisons for the two claims the paper rests on."""
    cost_key = ((lambda r: (r.get("joules_per_new_token") or {}).get(ENERGY_DOMAIN))
                if energy else
                (lambda r: 1.0 / r["decode_tok_per_s"] if r["decode_tok_per_s"] else None))
    cost_name = "J/token" if energy else "s/token"
    tps_key = lambda r: r["decode_tok_per_s"]

    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r["precision"], r["placement"])].append(r)
    if not by_cfg:
        return ""

    n_prompts = len({r.get("prompt_index", 0) for r in rows})
    out = [
        "## Paired comparisons",
        "",
        f"Ratios computed within each of the {n_prompts} prompts and then "
        "aggregated (geometric mean, bootstrap 95% CI over prompts). Pairing "
        "removes prompt-to-prompt variation, which is otherwise large enough to "
        "swamp these effects. A ratio below 1.000 favours the first config; an "
        "interval spanning 1.000 means the data does not resolve the direction.",
        "",
    ]

    precisions = sorted({p for p, _ in by_cfg})
    placements = sorted({pl for _, pl in by_cfg})
    ref_prec = "bf16" if "bf16" in precisions else precisions[0]
    ref_place = "P-1t" if "P-1t" in placements else placements[0]

    # --- island effect, precision held fixed ---
    others = [pl for pl in placements if pl != ref_place
              and (ref_prec, pl) in by_cfg]
    if (ref_prec, ref_place) in by_cfg and others:
        out += [f"### Core placement (precision fixed at `{ref_prec}`)", "",
                f"| comparison | {cost_name} ratio | tok/s ratio |",
                "| --- | --- | --- |"]
        base = by_cfg[(ref_prec, ref_place)]
        for pl in others:
            c = paired_ratio(by_cfg[(ref_prec, pl)], base, cost_key)
            t = paired_ratio(by_cfg[(ref_prec, pl)], base, tps_key)
            out.append(f"| `{pl}` vs `{ref_place}` | "
                       f"{_fmt_ratio(c) if c else '-'} | "
                       f"{_fmt_ratio(t) if t else '-'} |")
        out.append("")

    # --- precision effect, placement held fixed ---
    others = [p for p in precisions if p != ref_prec and (p, ref_place) in by_cfg]
    if (ref_prec, ref_place) in by_cfg and others:
        out += [f"### Precision (placement fixed at `{ref_place}`)", "",
                f"| comparison | {cost_name} ratio | tok/s ratio |",
                "| --- | --- | --- |"]
        base = by_cfg[(ref_prec, ref_place)]
        for p in others:
            c = paired_ratio(by_cfg[(p, ref_place)], base, cost_key)
            t = paired_ratio(by_cfg[(p, ref_place)], base, tps_key)
            out.append(f"| `{p}` vs `{ref_prec}` | "
                       f"{_fmt_ratio(c) if c else '-'} | "
                       f"{_fmt_ratio(t) if t else '-'} |")
        out.append("")
    return "\n".join(out)


def determinism_section(rows: list[dict]) -> str:
    div = determinism_report(rows)
    if not any(r.get("output_sha") for r in rows):
        return ""
    out = ["## Output determinism across placements", ""]
    if not div:
        out += ["Greedy decode produced byte-identical output on every core "
                "island, for every precision and prompt. Placement affected "
                "cost only, not the tokens generated.", ""]
        return "\n".join(out)
    out += [
        "> **Placement changed the generated text.** Greedy decode at a fixed "
        "seed should not depend on which cores run it. Where it does, the "
        "islands are dispatching to different kernels and the arithmetic "
        "differs — which also makes any perplexity difference between these "
        "rows a real numerical effect rather than measurement noise.",
        "",
        "| precision | prompt | distinct outputs | by placement |",
        "| --- | --- | --- | --- |",
    ]
    for d in div:
        per = "; ".join(f"`{p}`={','.join(s)[:8]}" for p, s in d["placements"].items())
        out.append(f"| `{d['precision']}` | {d['prompt_index']} | "
                   f"{d['n_distinct']} | {per} |")
    unstable = sorted({p for d in div for p in d["unstable_within_placement"]})
    out.append("")
    if unstable:
        out += [f"Output was also unstable *within* a single placement for: "
                f"{', '.join('`' + p + '`' for p in unstable)}. That is "
                "non-determinism independent of placement and needs separate "
                "investigation.", ""]
    return "\n".join(out)


def place_labels(ax, items: list[tuple[float, float, str]], fontsize: int = 8,
                 pad_px: float = 3.0) -> None:
    """Annotate points, nudging labels that would overlap.

    Frontier points can sit almost on top of one another -- which is the whole
    reason they are both on the frontier -- so a fixed offset puts their labels
    in the same place and neither is readable. Each label is tried at a series of
    offsets and takes the first that clears every label already placed; a leader
    line is drawn when the offset is large enough that the pairing is no longer
    obvious. Boxes are estimated from the font metrics rather than measured, so
    this is approximate by design and only has to be good enough to separate a
    handful of labels."""
    if not items:
        return
    # Candidate offsets in points, tried in order: right, left, then stacked.
    candidates = [(9, 4), (9, -12), (-9, 4), (-9, -12),
                  (9, 18), (9, -26), (-9, 18), (-9, -26),
                  (0, 24), (0, -32)]
    placed: list[tuple[float, float, float, float]] = []
    inv = ax.transData

    def overlaps(box, others):
        x0, y0, x1, y1 = box
        return any(not (x1 < o[0] or o[2] < x0 or y1 < o[1] or o[3] < y0)
                   for o in others)

    for x, y, text in sorted(items, key=lambda t: (-t[1], t[0])):
        px, py = inv.transform((x, y))
        w = len(text) * fontsize * 0.55 + 2 * pad_px
        h = fontsize * 1.5 + 2 * pad_px
        chosen = candidates[0]
        for dx, dy in candidates:
            left = px + dx if dx >= 0 else px + dx - w
            box = (left - pad_px, py + dy - pad_px,
                   left + w + pad_px, py + dy + h + pad_px)
            if not overlaps(box, placed):
                chosen = (dx, dy)
                placed.append(box)
                break
        else:
            left = px + chosen[0]
            placed.append((left - pad_px, py + chosen[1] - pad_px,
                           left + w + pad_px, py + chosen[1] + h + pad_px))
        far = abs(chosen[1]) > 14 or chosen[0] < 0
        ax.annotate(
            text, (x, y), fontsize=fontsize, color=INK,
            xytext=chosen, textcoords="offset points", zorder=5,
            ha="left" if chosen[0] >= 0 else "right",
            arrowprops=(dict(arrowstyle="-", linewidth=0.6, color=GRID,
                             shrinkA=0, shrinkB=2) if far else None),
        )


def make_plots(summary: list[dict], loss: dict[str, float],
               frontier: set[str], rows: list[dict], energy: bool,
               outdir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # --- 1. cost vs quality, frontier highlighted ---
    pts = [(s, s["joules_per_token_med"] if energy else s["s_per_token_med"],
            loss.get(s["config"]))
           for s in summary]
    pts = [(s, x, q) for s, x, q in pts if x is not None and q is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        for s, x, q in pts:
            on = s["config"] in frontier
            ax.scatter(x, q * 100,
                       s=170 if on else 95,
                       c=PRECISION_COLOR.get(s["precision"], INK_MUTED),
                       marker=PLACEMENT_MARKER.get(s["placement"], "o"),
                       # 2px surface ring so overlapping marks stay separable
                       edgecolors=SURFACE, linewidths=2.0,
                       zorder=4 if on else 3)

        fr = sorted((x, q * 100) for s, x, q in pts if s["config"] in frontier)
        if len(fr) > 1:
            ax.plot([a for a, _ in fr], [b for _, b in fr], linestyle="--",
                    linewidth=1.2, color=GRID, zorder=1)

        # Label only the frontier -- the points the reader has to act on. A value
        # beside all twelve is what made this plot unreadable before.
        place_labels(ax, [(x, q * 100, f"{s['precision']} / {s['placement']}")
                          for s, x, q in pts if s["config"] in frontier])

        # Quality loss spans 0% to ~900%; symlog keeps the near-zero cluster
        # readable without hiding the collapse. Linear below 1%, log above.
        # q is a fraction; the axis is plotted in percent. Compare in percent.
        if max(q for _, _, q in pts) * 100 > 20:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylabel("quality loss vs best\n(% perplexity increase, symlog)",
                          color=INK_MUTED, fontsize=9)
        else:
            ax.set_ylabel("quality loss vs best (% perplexity increase)",
                          color=INK_MUTED, fontsize=9)
        ax.set_xlabel("joules per generated token" if energy
                      else "seconds per generated token",
                      color=INK_MUTED, fontsize=9)
        ax.set_title("Cost vs quality — labelled points are non-dominated"
                     + ("" if energy else "   (energy unmeasured)"),
                     color=INK, fontsize=11, loc="left")

        precisions = sorted({s["precision"] for s, _, _ in pts})
        placements = sorted({s["placement"] for s, _, _ in pts})
        col_handles = [plt.Line2D([], [], marker="o", linestyle="",
                                  markersize=8, markeredgecolor=SURFACE,
                                  markerfacecolor=PRECISION_COLOR.get(p, INK_MUTED),
                                  label=p) for p in precisions]
        mk_handles = [plt.Line2D([], [], marker=PLACEMENT_MARKER.get(p, "o"),
                                 linestyle="", markersize=8, color=INK_MUTED,
                                 markeredgecolor=SURFACE, label=p)
                      for p in placements]
        leg1 = ax.legend(handles=col_handles, title="precision", loc="upper left",
                         bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8,
                         title_fontsize=8, labelcolor=INK_MUTED)
        leg1.get_title().set_color(INK_MUTED)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=mk_handles, title="core placement",
                         loc="upper left", bbox_to_anchor=(1.02, 0.55),
                         frameon=False, fontsize=8, title_fontsize=8,
                         labelcolor=INK_MUTED)
        leg2.get_title().set_color(INK_MUTED)

        ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)

        p = outdir / "pareto.png"
        fig.tight_layout(); fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)

    # --- 2. throughput by configuration ---
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.42 * len(summary) + 1)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = [s["config"].replace("cpu/", "") for s in summary]
    vals = [s["decode_tok_per_s_med"] for s in summary]
    errs = [s["decode_tok_per_s_iqr"] / 2 for s in summary]
    colors = [PRECISION_COLOR.get(s["precision"], INK_MUTED) for s in summary]
    ax.barh(labels, vals, xerr=errs, capsize=3, color=colors,
            # 2px surface gap between adjacent bars
            edgecolor=SURFACE, linewidth=2.0,
            error_kw={"ecolor": INK_MUTED, "elinewidth": 1})
    ax.set_xlabel("decode tokens/sec (median, error bars = IQR/2)",
                  color=INK_MUTED, fontsize=9)
    ax.set_title("Decode throughput", color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          markeredgecolor=SURFACE,
                          markerfacecolor=PRECISION_COLOR[k], label=k)
               for k in sorted({s["precision"] for s in summary}
                               & set(PRECISION_COLOR))]
    if handles:
        leg = ax.legend(handles=handles, title="precision", frameon=False,
                        fontsize=8, title_fontsize=8, loc="lower right",
                        labelcolor=INK_MUTED)
        leg.get_title().set_color(INK_MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    p = outdir / "throughput.png"
    fig.tight_layout(); fig.savefig(p, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(p)

    # --- 3. KV cache growth ---
    series: dict[str, list[int]] = {}
    for r in rows:
        if r.get("kv_bytes_by_step") and r["config"] not in series:
            series[r["config"]] = r["kv_bytes_by_step"]
    if series:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        # One line per precision: the curve depends on dtype (cache element size),
        # not on where the threads run, so plotting all twelve would draw the same
        # three lines four times over.
        by_prec: dict[str, list[int]] = {}
        for cfg, ks in series.items():
            prec = cfg.split("/")[1] if "/" in cfg else cfg
            by_prec.setdefault(prec, ks)

        # Precisions whose cache is byte-identical share one line and one legend
        # entry. fp32 and int8-dynamic coincide because dynamic quantization only
        # touches Linear weights -- the KV cache stays fp32 -- so drawing both
        # would hide one series under the other and put a phantom in the legend.
        merged: dict[tuple, list[str]] = {}
        for prec, ks in by_prec.items():
            merged.setdefault(tuple(ks), []).append(prec)
        for ks, precs in merged.items():
            precs.sort()
            label = " = ".join(precs)
            ax.plot(range(1, len(ks) + 1), [k / 1024 for k in ks],
                    label=label, linewidth=2.0,
                    color=PRECISION_COLOR.get(precs[0], INK_MUTED))
        ax.set_xlabel("decode step", color=INK_MUTED, fontsize=9)
        ax.set_ylabel("KV cache (KiB)", color=INK_MUTED, fontsize=9)
        ax.set_title("KV cache growth during decode", color=INK, fontsize=11,
                     loc="left")
        leg = ax.legend(title="precision", frameon=False, fontsize=8,
                        title_fontsize=8, labelcolor=INK_MUTED)
        leg.get_title().set_color(INK_MUTED)
        ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        p = outdir / "kv_growth.png"
        fig.tight_layout(); fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)

    # --- 4. net power vs throughput: the race-to-idle mechanism ---
    # If power is flat along y while x spans several-fold, then energy per token
    # is set by duration alone. Stating that as a sentence invites doubt; showing
    # the two axes together settles it. Iso-energy hyperbolae (W = J/tok * tok/s)
    # turn "same energy" into a visible curve rather than arithmetic.
    pw = [s for s in summary if s.get("net_watts_med") and s.get("decode_tok_per_s_med")]
    if energy and len(pw) > 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)

        xs = [s["decode_tok_per_s_med"] for s in pw]
        ys = [s["net_watts_med"] for s in pw]
        # A handful of round iso-energy curves, not one per configuration --
        # twelve near-identical dotted lines are clutter, not a reference grid.
        jpts = [s["joules_per_token_med"] for s in pw if s.get("joules_per_token_med")]
        lo, hi = min(jpts), max(jpts)
        levels = [v for v in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
                  if lo * 0.7 <= v <= hi * 1.3]
        gx = [min(xs) * 0.85 + i * (max(xs) * 1.15 - min(xs) * 0.85) / 60
              for i in range(61)]
        for jpt in levels:
            ax.plot(gx, [jpt * g for g in gx], color=GRID, linewidth=0.7,
                    linestyle=":", zorder=1)
            ax.annotate(f"{jpt:g} J/tok", (gx[-1], jpt * gx[-1]), fontsize=6,
                        color=GRID, ha="right", va="bottom", zorder=1)

        for s in pw:
            ax.errorbar(
                s["decode_tok_per_s_med"], s["net_watts_med"],
                yerr=([[s["net_watts_med"] - s["net_watts_ci"][0]],
                       [s["net_watts_ci"][1] - s["net_watts_med"]]]
                      if s.get("net_watts_ci") else None),
                fmt="none", ecolor=GRID, elinewidth=1.0, capsize=2, zorder=2)
            ax.scatter(s["decode_tok_per_s_med"], s["net_watts_med"], s=110,
                       c=PRECISION_COLOR.get(s["precision"], INK_MUTED),
                       marker=PLACEMENT_MARKER.get(s["placement"], "o"),
                       edgecolors=SURFACE, linewidths=2.0, zorder=3)

        place_labels(ax, [(s["decode_tok_per_s_med"], s["net_watts_med"],
                           s["placement"]) for s in pw], fontsize=7)
        ax.set_xlabel("decode tokens/sec", color=INK_MUTED, fontsize=9)
        ax.set_ylabel("net package power (W)", color=INK_MUTED, fontsize=9)
        ax.set_title("Power is nearly flat across core islands; throughput is not.\n"
                     "Dotted curves are constant joules per token.",
                     color=INK, fontsize=10, loc="left")
        # Bound the axis by the data, not by where the iso-energy curves happen
        # to exit the frame. Letting the curves set the limit left three
        # quarters of the panel empty and squashed the very flatness the figure
        # exists to show. Zero stays on the axis because this is a power scale.
        ax.set_ylim(0, max(ys) * 1.35)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        handles = [plt.Line2D([], [], marker=PLACEMENT_MARKER.get(p, "o"), ls="",
                              color=INK_MUTED, markersize=7, label=p)
                   for p in sorted({s["placement"] for s in pw})]
        handles += [plt.Line2D([], [], marker="o", ls="", markersize=7,
                               markerfacecolor=PRECISION_COLOR[k],
                               markeredgecolor=SURFACE, color="none", label=k)
                    for k in sorted({s["precision"] for s in pw})
                    if k in PRECISION_COLOR]
        ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2,
                  labelcolor=INK_MUTED, loc="upper left")
        p = outdir / "power_vs_throughput.png"
        fig.tight_layout(); fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)

    # --- 5. paired-ratio forest plot ---
    # Twelve overlapping confidence intervals in a table are hard to compare.
    # On one axis with a rule at 1.0, "which of these effects are real" is a
    # glance rather than a reading exercise.
    comparisons = _forest_data(rows, energy)
    if comparisons:
        fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(comparisons) + 1.7),
                               facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.axvline(1.0, color=INK_MUTED, linewidth=1.0, zorder=1)
        for i, c in enumerate(comparisons):
            y = len(comparisons) - i
            crosses = c["ci"] and c["ci"][0] <= 1.0 <= c["ci"][1]
            col = INK_MUTED if crosses else PRECISION_COLOR.get(c["color_key"], INK)
            if c["ci"]:
                # End caps, and a marker small enough not to swallow the
                # interval: on a well-powered dataset the CI can be narrower
                # than a dot, and an invisible interval reads as a missing one.
                ax.plot(c["ci"], [y, y], color=col, linewidth=1.4,
                        solid_capstyle="butt", zorder=2)
                for b in c["ci"]:
                    ax.plot([b, b], [y - 0.16, y + 0.16], color=col,
                            linewidth=1.4, zorder=2)
            ax.scatter([c["ratio"]], [y], s=22, color=col, zorder=3,
                       edgecolors=SURFACE, linewidths=1.0)
        ax.set_yticks(range(1, len(comparisons) + 1))
        ax.set_yticklabels([c["label"] for c in reversed(comparisons)],
                           fontsize=8, color=INK)
        ax.set_xscale("log")
        # Ratios live in a narrow band around 1; default log tick labels render
        # them as 10^0 and 2x10^0, which nobody reads as "1.0" and "2.0".
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if v in (0.8, 1.5, 2.0, 3.0) else ""))
        ax.set_xlabel(("joules/token ratio " if energy else "s/token ratio ")
                      + "(log scale; <1 favours the first config)",
                      color=INK_MUTED, fontsize=9)
        ax.set_title("Paired per-prompt ratios, bootstrap 95% CI.\n"
                     "Grey intervals cross 1.0 — the data does not resolve a "
                     "direction. Some intervals are narrower than their marker.",
                     color=INK, fontsize=10, loc="left")
        ax.grid(True, axis="x", color=GRID, linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        p = outdir / "paired_ratios.png"
        fig.tight_layout(); fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)

    return written


def _forest_data(rows: list[dict], energy: bool) -> list[dict]:
    """The same comparisons paired_section() tabulates, shaped for plotting."""
    cost_key = ((lambda r: (r.get("joules_per_new_token") or {}).get(ENERGY_DOMAIN))
                if energy else
                (lambda r: 1.0 / r["decode_tok_per_s"] if r["decode_tok_per_s"] else None))
    by_cfg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[(r["precision"], r["placement"])].append(r)
    if not by_cfg:
        return []
    precisions = sorted({p for p, _ in by_cfg})
    placements = sorted({pl for _, pl in by_cfg})
    ref_prec = "bf16" if "bf16" in precisions else precisions[0]
    ref_place = "P-1t" if "P-1t" in placements else placements[0]

    out = []
    if (ref_prec, ref_place) in by_cfg:
        base = by_cfg[(ref_prec, ref_place)]
        for pl in placements:
            if pl == ref_place or (ref_prec, pl) not in by_cfg:
                continue
            res = paired_ratio(by_cfg[(ref_prec, pl)], base, cost_key)
            if res:
                out.append({**res, "label": f"{pl} vs {ref_place}  ({ref_prec})",
                            "color_key": ref_prec})
        for p in precisions:
            if p == ref_prec or (p, ref_place) not in by_cfg:
                continue
            res = paired_ratio(by_cfg[(p, ref_place)], base, cost_key)
            if res:
                out.append({**res, "label": f"{p} vs {ref_prec}  ({ref_place})",
                            "color_key": p})
    return out


def main() -> int:
    global ENERGY_DOMAIN          # must precede any use of the name in scope
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(HERE / "results.jsonl"))
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    ap.add_argument("--report", default=str(HERE / "RESULTS.md"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN,
                    help=f"RAPL domain for the cost axis (default: {DEFAULT_DOMAIN})")
    a = ap.parse_args()
    ENERGY_DOMAIN = a.domain

    path = Path(a.results)
    if not path.exists():
        raise SystemExit(f"{path} not found - run runner.py first")

    meta, rows = load(path)
    if not rows:
        raise SystemExit("no result rows found")

    summary = summarise(rows)
    loss = quality_loss(summary)
    energy = any(s["energy_measured"] for s in summary)

    pts = [(s["config"],
            s["joules_per_token_med"] if energy else s["s_per_token_med"],
            loss.get(s["config"], 0.0))
           for s in summary
           if (s["joules_per_token_med"] if energy else s["s_per_token_med"])
           is not None]
    frontier = set(pareto(pts))

    machine = meta.get("machine", {})
    header = [
        "# Results",
        "",
        f"Model: `{meta.get('model', '?')}`  ·  "
        f"{meta.get('repeats', '?')} repeats per configuration  ·  "
        f"{meta.get('max_new_tokens', '?')} new tokens per generation",
        "",
        f"Machine: {machine.get('processor') or 'unknown'} — "
        + ", ".join(f"{n} {v['physical']}c/{v['logical']}t @ {v['max_ghz']}GHz"
                    for n, v in machine.get("islands", {}).items()),
        "",
    ]
    if energy:
        mode = baseline_mode_of(rows)
        if mode == "run":
            ref = (meta.get("idle_watts_reference")
                   or meta.get("idle_watts") or {}).get(ENERGY_DOMAIN)
            desc = ("net of a single run-level idle baseline"
                    + (f" of {ref:.2f} W" if ref else "")
                    + ", measured before any load")
        elif mode == "per-config":
            desc = ("net of a per-configuration idle baseline, measured after a "
                    "settle period and verified stable across two windows")
        else:
            desc = (f"net of a **{mode}** baseline — configurations were not "
                    "baselined the same way and are not directly comparable")
        header += [f"Energy from RAPL `{ENERGY_DOMAIN}`, {desc}.", ""]
        bl = baseline_report(summary)
        if bl:
            header += [bl, ""]
        neg_rep = sum(s.get("n_negative_reported", 0) for s in summary)
        neg_oth = sum(s.get("n_negative_other", 0) for s in summary)
        total_n = sum(s.get("n", 0) for s in summary)
        if neg_rep:
            header += [
                f"> **{neg_rep} of {total_n} measurements produced negative net "
                f"energy in `{ENERGY_DOMAIN}`, the domain every figure here is "
                "computed from.** The baseline exceeded the measurement, so the "
                "baseline is wrong for those runs and the affected numbers "
                "should not be trusted. Recorded unclamped rather than floored "
                "at zero so the error stays visible.",
                "",
            ]
        if neg_oth:
            header += [
                f"Net energy went slightly negative in a domain other than "
                f"`{ENERGY_DOMAIN}` for {neg_oth} of {total_n} measurements. "
                "This is expected rather than alarming: the `uncore` baseline is "
                "around 0.2 W and dominated by noise, so subtracting it can "
                "cross zero. No figure in this report is computed from those "
                f"domains, and `{ENERGY_DOMAIN}` itself did not go negative.",
                "",
            ]
    else:
        header += [
            "> **Energy was not measured** — RAPL counters were unreadable for "
            "this run, so cost is reported as seconds per token. See README for "
            "how to grant access; the same run then yields joules.",
            "",
        ]

    body = [
        "## Per-configuration summary",
        "",
        fmt_table(summary, loss, frontier, energy),
        "",
    ]
    ps = paired_section(rows, energy)
    if ps:
        body += [ps]
    ds = determinism_section(rows)
    if ds:
        body += [ds]
    body += [
        "## Non-dominated configurations",
        "",
    ]
    if frontier:
        for c in sorted(frontier):
            body.append(f"- `{c}`")
    else:
        body.append("_none identified_")
    body += [
        "",
        "Everything absent from that list is dominated: some other configuration "
        "is at least as cheap *and* at least as accurate, so it should never be "
        "chosen.",
        "",
    ]

    report = "\n".join(header + body)
    Path(a.report).write_text(report + "\n")

    print(fmt_table(summary, loss, frontier, energy))
    print()
    print("frontier:", ", ".join(sorted(frontier)) or "none")
    bl = baseline_report(summary)
    if bl:
        print(bl.replace("**", "").replace("`", ""))
    neg_rep = sum(s.get("n_negative_reported", 0) for s in summary)
    neg_oth = sum(s.get("n_negative_other", 0) for s in summary)
    if neg_rep:
        print(f"WARNING: {neg_rep} measurement(s) had negative net energy in "
              f"{ENERGY_DOMAIN} (baseline exceeded measurement)")
    if neg_oth:
        print(f"note: {neg_oth} measurement(s) went negative only in a domain "
              f"other than {ENERGY_DOMAIN}; expected for the noise-dominated "
              "uncore baseline, and no reported figure uses it")
    if not energy:
        print("note: energy unmeasured; cost axis is seconds/token")

    if not a.no_plots:
        figs = make_plots(summary, loss, frontier, rows, energy, Path(a.outdir))
        for f in figs:
            print(f"wrote {f}")
    print(f"wrote {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
