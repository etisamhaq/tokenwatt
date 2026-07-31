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
            "energy_measured": bool(jpt_vals),
            "param_bytes": first["param_bytes"],
            "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rs),
            "kv_bytes_final": first["kv_bytes_final"],
            "perplexity": first.get("perplexity"),
            "needle_pass": first.get("needle_pass"),
            "needle_total": first.get("needle_total"),
            "idle_w": ((first.get("idle_watts_config") or {})
                       .get(ENERGY_DOMAIN)),
            "n_negative_net": sum(1 for r in rs
                                  if r.get("net_negative_domains")),
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
    cost_hdr = "J/token" if energy else "s/token"
    lines = [
        f"| config | tok/s (IQR) | {cost_hdr} | perplexity | Δquality | needle | on frontier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        cost = (s["joules_per_token_med"] if energy else s["s_per_token_med"])
        cost_s = f"{cost:.4f}" if cost is not None else "-"
        ppl = f"{s['perplexity']:.3f}" if s.get("perplexity") is not None else "-"
        dq = f"{loss[s['config']] * 100:+.1f}%" if s["config"] in loss else "-"
        nd = (f"{s['needle_pass']}/{s['needle_total']}"
              if s.get("needle_total") else "-")
        lines.append(
            f"| `{s['config']}` | {s['decode_tok_per_s_med']:.1f} "
            f"(±{s['decode_tok_per_s_iqr']:.1f}) | {cost_s} | {ppl} | {dq} | {nd} "
            f"| {'yes' if s['config'] in frontier else ''} |"
        )
    return "\n".join(lines)


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
        for s, x, q in pts:
            if s["config"] in frontier:
                ax.annotate(f"{s['precision']} / {s['placement']}",
                            (x, q * 100), fontsize=8, color=INK,
                            xytext=(9, 4), textcoords="offset points",
                            zorder=5)

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

    return written


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
        neg = sum(s.get("n_negative_net", 0) for s in summary)
        if neg:
            header += [
                f"> **{neg} measurement(s) produced negative net energy.** The "
                "baseline exceeded the measurement, which means the baseline is "
                "wrong for those runs. Recorded unclamped rather than floored at "
                "zero so the error is visible.",
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
    neg = sum(s.get("n_negative_net", 0) for s in summary)
    if neg:
        print(f"WARNING: {neg} measurement(s) had negative net energy "
              "(baseline exceeded measurement)")
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
