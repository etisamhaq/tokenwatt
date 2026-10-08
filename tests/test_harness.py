#!/usr/bin/env python3
"""Unit tests for the measurement harness.

These cover the logic that would silently corrupt results if wrong -- counter
wraparound, tier classification, Pareto domination, needle placement -- and
deliberately avoid loading a model, so the suite runs in under a second and can
be run on every change.

Run:  python3 -m pytest tests/ -q      (or: python3 tests/test_harness.py)
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze                      # noqa: E402
import backends                     # noqa: E402
import energy                       # noqa: E402
import quality                      # noqa: E402
import topology                     # noqa: E402


# --------------------------------------------------------------- energy
class _FakeDomain(energy.RaplDomain):
    """RaplDomain without touching sysfs."""
    def __init__(self, max_uj):
        self.name = "fake"
        self.max_uj = max_uj


def test_energy_delta_normal():
    d = _FakeDomain(1000)
    assert d.delta_uj(100, 400) == 300


def test_energy_delta_wraparound():
    """The counter rolls at max_uj; a naive subtraction would go negative."""
    d = _FakeDomain(1000)
    assert d.delta_uj(900, 100) == 200      # 100 to the top, then 100 more


def test_energy_delta_exact_zero():
    d = _FakeDomain(1000)
    assert d.delta_uj(500, 500) == 0


def test_check_readable_returns_reason():
    ok, msg = energy.check_readable()
    assert isinstance(ok, bool)
    assert isinstance(msg, str) and msg          # never an empty explanation
    if not ok:
        assert "chmod" in msg or "unsupported" in msg or "no intel-rapl" in msg


# ------------------------------------------------------------- topology
def test_tiers_are_ordered_fastest_first():
    tiers = topology.merge_p_tiers(topology.discover_tiers())
    freqs = [t.max_khz for t in tiers if t.max_khz]
    assert freqs == sorted(freqs, reverse=True)


def test_turbo_bins_merge_into_one_island():
    """4800 and 4500 MHz P-cores are the same core type, not two islands."""
    a = topology.Tier("P", 4_800_000, [0, 1], {0, 1})
    b = topology.Tier("E", 4_500_000, [2, 3], {2, 3})   # within 10% -> merge
    c = topology.Tier("LP-E", 2_500_000, [4], {4})      # far apart -> separate
    merged = topology.merge_p_tiers([a, b, c])
    assert len(merged) == 2
    assert merged[0].cpus == [0, 1, 2, 3]
    assert merged[1].cpus == [4]


def test_distinct_tiers_are_not_merged():
    a = topology.Tier("P", 4_000_000, [0], {0})
    b = topology.Tier("E", 2_000_000, [1], {1})
    assert len(topology.merge_p_tiers([a, b])) == 2


def test_one_thread_per_core_dedupes_smt_siblings():
    t = topology.Tier("P", 4_000_000, [0, 1, 2, 3], {0, 1})
    picked = t.one_thread_per_core()
    assert len(picked) == len({topology._physical_core(c) for c in picked})


def test_islands_cover_every_online_cpu():
    """A CPU silently missing from the island map would mean an untested
    placement and a wrong machine description in the report."""
    covered = set()
    for t in topology.get_islands().values():
        covered |= set(t.cpus)
    assert covered == set(topology.online_cpus())


def test_affinity_restores_previous_mask():
    before = os.sched_getaffinity(0)
    target = sorted(before)[:1]
    with topology.Affinity(target):
        assert os.sched_getaffinity(0) == set(target)
    assert os.sched_getaffinity(0) == before


def test_affinity_with_none_is_a_noop():
    before = os.sched_getaffinity(0)
    with topology.Affinity(None):
        assert os.sched_getaffinity(0) == before
    assert os.sched_getaffinity(0) == before


# ------------------------------------------------------------- backends
class _T:
    """Minimal tensor stand-in for KV cache accounting."""
    def __init__(self, n, size):
        self._n, self._s = n, size

    def numel(self):
        return self._n

    def element_size(self):
        return self._s


class _Cache:
    def __init__(self, k, v):
        self.key_cache, self.value_cache = k, v


def test_kv_bytes_cache_object():
    c = _Cache([_T(100, 4), _T(100, 4)], [_T(100, 4), _T(100, 4)])
    assert backends._kv_cache_bytes(c) == 1600


def test_kv_bytes_legacy_tuple_layout():
    past = ((_T(50, 2), _T(50, 2)), (_T(50, 2), _T(50, 2)))
    assert backends._kv_cache_bytes(past) == 400


def test_kv_bytes_none_and_garbage():
    assert backends._kv_cache_bytes(None) == 0
    assert backends._kv_cache_bytes(42) == 0        # not iterable -> 0, no crash


def test_genresult_rates():
    r = backends.GenResult("x", n_prompt_tokens=10, n_new_tokens=20,
                           prefill_s=0.5, decode_s=2.0)
    assert r.total_s == 2.5
    assert r.decode_tok_per_s == 10.0
    assert r.prefill_tok_per_s == 20.0


def test_genresult_zero_time_does_not_divide_by_zero():
    r = backends.GenResult("x", 1, 1, 0.0, 0.0)
    assert r.decode_tok_per_s == 0.0
    assert r.prefill_tok_per_s == 0.0


def test_unknown_precision_rejected_early():
    try:
        backends.make_backend("cpu", "m", "int4-magic")
    except ValueError:
        return
    raise AssertionError("bad precision should raise before any model load")


def test_unimplemented_engines_fail_loudly():
    for eng in ("igpu", "npu"):
        b = backends.make_backend(eng, "m", "fp32")
        try:
            b.load()
        except NotImplementedError:
            continue
        raise AssertionError(f"{eng} must not pretend to work")


# -------------------------------------------------------------- quality
def test_needle_depth_places_fact_proportionally():
    filler = "F."
    early = quality._build_needle(filler, 10, 0.0)
    late = quality._build_needle(filler, 10, 1.0)
    mid = quality._build_needle(filler, 10, 0.5)
    assert early.index(quality.NEEDLE_FACT) < mid.index(quality.NEEDLE_FACT)
    assert mid.index(quality.NEEDLE_FACT) < late.index(quality.NEEDLE_FACT)


def test_needle_keeps_all_filler():
    n = quality._build_needle("F.", 10, 0.3)
    assert n.count("F.") == 10


class _TokWithTemplate:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        return "<|user|>" + msgs[0]["content"] + "<|assistant|>"


class _TokNoTemplate:
    chat_template = None


class _B:
    def __init__(self, tok):
        self.tok = tok


def test_as_chat_wraps_when_template_present():
    out = quality._as_chat(_B(_TokWithTemplate()), "hello")
    assert out == "<|user|>hello<|assistant|>"


def test_as_chat_returns_none_without_template():
    """Signals the caller to use the plain-continuation fallback."""
    assert quality._as_chat(_B(_TokNoTemplate()), "hello") is None
    assert quality._as_chat(_B(None), "hello") is None


def test_as_chat_survives_a_broken_template():
    class Boom:
        chat_template = "x"

        def apply_chat_template(self, *a, **k):
            raise RuntimeError("bad template")
    assert quality._as_chat(_B(Boom()), "hello") is None


def test_corpus_present_and_nonTrivial():
    text = quality.load_corpus()
    assert len(text) > 2000, "corpus too short for a stable perplexity signal"


# -------------------------------------------------------------- analyze
def test_pareto_drops_dominated_points():
    pts = [("cheap_good", 1.0, 0.0),
           ("dear_bad", 2.0, 0.5),        # dominated on both axes
           ("cheap_bad", 1.0, 0.9),       # dominated on quality at equal cost
           ("dear_good", 3.0, -0.1)]      # best quality, so not dominated
    front = set(analyze.pareto(pts))
    assert "cheap_good" in front
    assert "dear_good" in front
    assert "dear_bad" not in front
    assert "cheap_bad" not in front


def test_pareto_single_point_is_on_frontier():
    assert analyze.pareto([("only", 1.0, 1.0)]) == ["only"]


def test_iqr_basic_and_degenerate():
    assert analyze.iqr([1, 2, 3, 4, 5]) > 0
    assert analyze.iqr([7]) == 0.0
    assert analyze.iqr([]) == 0.0


def test_summarise_medians_over_repeats():
    rows = [
        {"config": "c", "engine": "cpu", "precision": "fp32", "placement": "all",
         "threads": 4, "decode_tok_per_s": 10.0, "prefill_s": 0.1,
         "joules_per_new_token": {"package-0": 1.0}, "param_bytes": 1,
         "peak_rss_bytes": 10, "kv_bytes_final": 5, "perplexity": 2.0},
        {"config": "c", "engine": "cpu", "precision": "fp32", "placement": "all",
         "threads": 4, "decode_tok_per_s": 20.0, "prefill_s": 0.3,
         "joules_per_new_token": {"package-0": 3.0}, "param_bytes": 1,
         "peak_rss_bytes": 20, "kv_bytes_final": 5, "perplexity": 2.0},
    ]
    s = analyze.summarise(rows)[0]
    assert s["n"] == 2
    assert s["decode_tok_per_s_med"] == 15.0
    assert s["joules_per_token_med"] == 2.0
    assert s["peak_rss_bytes"] == 20            # peak, not median
    assert s["energy_measured"] is True


def test_summarise_marks_energy_absent():
    rows = [{"config": "c", "engine": "cpu", "precision": "fp32",
             "placement": "all", "threads": 4, "decode_tok_per_s": 10.0,
             "prefill_s": 0.1, "joules_per_new_token": None, "param_bytes": 1,
             "peak_rss_bytes": 10, "kv_bytes_final": 5, "perplexity": None}]
    s = analyze.summarise(rows)[0]
    assert s["energy_measured"] is False
    assert s["joules_per_token_med"] is None


def test_baseline_mode_defaults_to_run_for_old_rows():
    """Rows written before the field existed used the single run-level baseline;
    the report must not claim a method the data did not use."""
    assert analyze.baseline_mode_of([{"config": "c"}]) == "run"


def test_baseline_mode_read_from_rows():
    assert analyze.baseline_mode_of(
        [{"baseline_mode": "per-config"}] * 3) == "per-config"


def test_baseline_mode_flags_mixed_data():
    out = analyze.baseline_mode_of([{"baseline_mode": "run"},
                                    {"baseline_mode": "per-config"}])
    assert out.startswith("mixed(")


def test_baseline_report_none_below_two_samples():
    assert analyze.baseline_report([]) is None
    assert analyze.baseline_report([{"idle_w": 10.0}]) is None


def test_baseline_report_computes_spread():
    out = analyze.baseline_report([{"idle_w": 9.0}, {"idle_w": 11.0},
                                   {"idle_w": 10.0}])
    assert out is not None
    assert "9.00-11.00 W" in out
    assert "20%" in out              # 2 W spread over a 10 W median


def test_baseline_report_flags_wide_drift():
    """Above 25% spread the absolute joule figures are not trustworthy and the
    report has to say so rather than print a tidy number."""
    out = analyze.baseline_report([{"idle_w": 5.0}, {"idle_w": 15.0}])
    assert "Wide drift" in out


def test_baseline_report_ignores_missing_values():
    out = analyze.baseline_report([{"idle_w": 10.0}, {"idle_w": None},
                                   {"idle_w": 12.0}])
    assert out is not None and "2 configurations" in out


def test_summarise_counts_negative_net_energy():
    """Negative net energy means a bad baseline; it must be surfaced, not
    silently averaged away -- and attributed to the right domain, because only
    a negative in the reported domain invalidates a figure."""
    base = {"config": "c", "engine": "cpu", "precision": "fp32",
            "placement": "all", "threads": 4, "decode_tok_per_s": 10.0,
            "prefill_s": 0.1, "joules_per_new_token": {"package-0": 1.0},
            "param_bytes": 1, "peak_rss_bytes": 10, "kv_bytes_final": 5,
            "perplexity": 2.0, "idle_watts_config": {"package-0": 9.5}}
    rows = [dict(base),
            dict(base, net_negative_domains=["package-0"]),
            dict(base, net_negative_domains=["uncore"])]
    s = analyze.summarise(rows)[0]
    assert s["n_negative_reported"] == 1
    assert s["n_negative_other"] == 1
    assert s["idle_w"] == 9.5


def test_quality_loss_is_relative_to_best():
    summary = [{"config": "a", "perplexity": 10.0},
               {"config": "b", "perplexity": 12.0}]
    loss = analyze.quality_loss(summary)
    assert loss["a"] == 0.0
    assert abs(loss["b"] - 0.2) < 1e-9


def test_quality_loss_empty_when_no_perplexity():
    assert analyze.quality_loss([{"config": "a", "perplexity": None}]) == {}



# ------------------------------------------------- bootstrap / paired stats
def _rows(precision, placement, prompt_vals, repeats=3):
    """prompt_vals: {prompt_index: joules_per_token}. Repeats are exact copies,
    so any spread the tests see comes from prompts, not from noise."""
    out = []
    for pi, v in prompt_vals.items():
        for r in range(repeats):
            out.append({
                "config": f"cpu/{precision}/{placement}/t1",
                "precision": precision, "placement": placement,
                "prompt_index": pi, "repeat": r,
                "decode_tok_per_s": 1.0 / v,
                "joules_per_new_token": {analyze.ENERGY_DOMAIN: v},
                "output_sha": f"{precision}-{pi}",
            })
    return out


def test_bootstrap_ci_brackets_the_median():
    lo, hi = analyze.bootstrap_ci([1.0, 1.1, 0.9, 1.05, 0.95])
    assert lo <= 1.0 <= hi


def test_bootstrap_ci_none_below_three_samples():
    assert analyze.bootstrap_ci([1.0, 2.0]) is None


def test_bootstrap_ci_on_constant_data_is_a_point():
    assert analyze.bootstrap_ci([2.0] * 6) == (2.0, 2.0)


def test_by_prompt_collapses_repeats_not_prompts():
    rows = _rows("bf16", "P-1t", {0: 1.0, 1: 3.0}, repeats=4)
    assert analyze.by_prompt(rows, lambda r: r["joules_per_new_token"]["package-0"]) \
        == {0: 1.0, 1: 3.0}


def test_paired_ratio_cancels_prompt_variance():
    """Prompts differ 4x; the A/B ratio is exactly 2x inside every prompt. The
    unpaired spread is huge, the paired answer is exact -- which is the whole
    reason for pairing."""
    a = _rows("x", "P", {0: 2.0, 1: 8.0})
    b = _rows("y", "P", {0: 1.0, 1: 4.0})
    res = analyze.paired_ratio(a, b,
                               lambda r: r["joules_per_new_token"]["package-0"])
    assert abs(res["ratio"] - 2.0) < 1e-9
    assert res["n_prompts"] == 2
    assert res["consistent"] is True


def test_paired_ratio_flags_inconsistent_sign():
    a = _rows("x", "P", {0: 2.0, 1: 1.0})
    b = _rows("y", "P", {0: 1.0, 1: 2.0})
    res = analyze.paired_ratio(a, b,
                               lambda r: r["joules_per_new_token"]["package-0"])
    assert res["consistent"] is False


def test_paired_ratio_uses_only_shared_prompts():
    a = _rows("x", "P", {0: 2.0, 5: 99.0})
    b = _rows("y", "P", {0: 1.0})
    res = analyze.paired_ratio(a, b,
                               lambda r: r["joules_per_new_token"]["package-0"])
    assert res["n_prompts"] == 1 and abs(res["ratio"] - 2.0) < 1e-9


def test_paired_ratio_none_without_overlap():
    a = _rows("x", "P", {0: 2.0})
    b = _rows("y", "P", {1: 1.0})
    assert analyze.paired_ratio(
        a, b, lambda r: r["joules_per_new_token"]["package-0"]) is None


# ------------------------------------------------------------ determinism
def test_determinism_silent_when_placements_agree():
    rows = _rows("bf16", "P-1t", {0: 1.0}) + _rows("bf16", "E", {0: 2.0})
    # _rows keys the sha on (precision, prompt), so both placements match.
    assert analyze.determinism_report(rows) == []


def test_determinism_flags_placement_divergence():
    rows = _rows("int8", "P-1t", {0: 1.0}) + _rows("int8", "LP-E", {0: 2.0})
    for r in rows:
        if r["placement"] == "LP-E":
            r["output_sha"] = "diverged"
    rep = analyze.determinism_report(rows)
    assert len(rep) == 1
    assert rep[0]["n_distinct"] == 2
    assert rep[0]["precision"] == "int8"
    assert rep[0]["unstable_within_placement"] == []


def test_determinism_flags_instability_within_one_placement():
    rows = _rows("bf16", "P-1t", {0: 1.0}, repeats=2)
    rows[0]["output_sha"] = "a"
    rows[1]["output_sha"] = "b"
    rep = analyze.determinism_report(rows)
    assert rep and rep[0]["unstable_within_placement"] == ["P-1t"]


def test_determinism_ignores_rows_without_sha():
    rows = _rows("bf16", "P-1t", {0: 1.0})
    for r in rows:
        r.pop("output_sha")
    assert analyze.determinism_report(rows) == []


# --------------------------------------------------- runner trial structure
def test_output_sha_is_stable_and_distinguishing():
    import runner
    assert runner.output_sha("abc") == runner.output_sha("abc")
    assert runner.output_sha("abc") != runner.output_sha("abd")


def test_trials_are_a_full_prompt_by_repeat_cross():
    """The bug this replaces: prompts were cycled by repeat index, so 5 repeats
    over 4 prompts measured prompt 0 twice and called the spread noise."""
    import runner
    prompts = ["p0", "p1", "p2"]
    repeats = 4
    trials = [(pi, rep, prompt)
              for pi, prompt in enumerate(prompts)
              for rep in range(repeats)]
    assert len(trials) == len(prompts) * repeats
    for pi in range(len(prompts)):
        assert sorted(r for p, r, _ in trials if p == pi) == list(range(repeats))
    # every prompt measured the same number of times
    counts = {pi: sum(1 for p, _, _ in trials if p == pi) for pi in range(3)}
    assert set(counts.values()) == {repeats}


def test_label_placement_separates_coincident_points():
    # matplotlib is an optional dependency: analyze.py guards its import and
    # skips plotting when it is absent, and requirements.txt says so. A test
    # that imports it unconditionally fails the whole suite in a configuration
    # the project explicitly supports. unittest.SkipTest is used because pytest
    # understands it natively and _run_all() handles it below.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise unittest.SkipTest("matplotlib not installed; plotting is optional")
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    analyze.place_labels(ax, [(0.5, 0.5, "alpha"), (0.5, 0.5, "beta"),
                              (0.5, 0.5, "gamma")])
    offsets = {tuple(a.get_position()) for a in ax.texts}
    assert len(ax.texts) == 3
    assert len(offsets) == 3, "coincident labels must not share an offset"
    plt.close(fig)



# ------------------------------------------------------------ warmup settling
class _RampBackend:
    """Throughput climbs toward a plateau, like a real frequency/cache ramp."""
    def __init__(self, rates):
        self.rates = list(rates); self.calls = 0
    def generate(self, prompt, max_new_tokens):
        r = self.rates[min(self.calls, len(self.rates) - 1)]
        self.calls += 1
        return backends.GenResult(text="x", n_prompt_tokens=1, n_new_tokens=10,
                                  prefill_s=0.1, decode_s=10.0 / r)


def test_warmup_stops_once_throughput_plateaus():
    import runner
    b = _RampBackend([9.0, 12.0, 13.5, 13.6, 13.6])
    rounds, settled = runner.warm_until_settled(b, "p", 256)
    assert settled is True
    assert rounds == 4          # 13.5 vs 13.6 is within 5%
    assert b.calls == 4


def test_warmup_reports_failure_when_it_never_settles():
    import runner
    b = _RampBackend([1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    rounds, settled = runner.warm_until_settled(b, "p", 256)
    assert settled is False and rounds == 6


def test_warmup_always_runs_at_least_two_rounds():
    """One warmup was the old behaviour and is what this replaces -- a backend
    that looks stable immediately must still be warmed twice."""
    import runner
    b = _RampBackend([10.0, 10.0, 10.0])
    rounds, settled = runner.warm_until_settled(b, "p", 256)
    assert rounds == 2 and settled is True


def test_warmup_uses_full_measurement_length():
    """An 8-token warmup does not populate the KV cache a 256-token run uses."""
    import runner
    seen = []
    class B(_RampBackend):
        def generate(self, prompt, max_new_tokens):
            seen.append(max_new_tokens)
            return super().generate(prompt, max_new_tokens)
    runner.warm_until_settled(B([10.0, 10.0]), "p", 256)
    assert seen == [256, 256]



def test_negative_net_is_counted_per_domain():
    """A clean dataset once reported "237 of 240 measurements produced negative
    net energy" because uncore noise was pooled with the reported domain."""
    rows = [{"config": "c", "engine": "cpu", "precision": "bf16",
             "placement": "P-1t", "threads": 1, "prompt_index": 0, "repeat": i,
             "decode_tok_per_s": 10.0, "prefill_s": 0.1, "param_bytes": 1,
             "peak_rss_bytes": 1, "kv_bytes_final": 1, "window_s": 1.0,
             "joules_net": {analyze.ENERGY_DOMAIN: 5.0, "uncore": -0.3},
             "joules_per_new_token": {analyze.ENERGY_DOMAIN: 0.5},
             "net_negative_domains": ["uncore"]} for i in range(3)]
    sm = analyze.summarise(rows)[0]
    assert sm["n_negative_reported"] == 0, "uncore noise must not be charged to the reported domain"
    assert sm["n_negative_other"] == 3


def test_negative_net_in_reported_domain_is_flagged():
    rows = [{"config": "c", "engine": "cpu", "precision": "bf16",
             "placement": "P-1t", "threads": 1, "prompt_index": 0, "repeat": i,
             "decode_tok_per_s": 10.0, "prefill_s": 0.1, "param_bytes": 1,
             "peak_rss_bytes": 1, "kv_bytes_final": 1, "window_s": 1.0,
             "joules_net": {analyze.ENERGY_DOMAIN: -2.0},
             "joules_per_new_token": {analyze.ENERGY_DOMAIN: -0.2},
             "net_negative_domains": [analyze.ENERGY_DOMAIN]} for i in range(3)]
    sm = analyze.summarise(rows)[0]
    assert sm["n_negative_reported"] == 3
    assert sm["n_negative_other"] == 0


# ----------------------------------------------------------------- main

# --- LlamaCppBackend ------------------------------------------------------
# Section V-D runs on this backend, and none of these need llama-cpp-python or
# the 1.1 GB of weights -- they cover the paths a reader without either hits.

def test_llamacpp_rejects_unknown_precision():
    for bad in ("bf16", "gguf-q5_k_m", ""):
        try:
            backends.LlamaCppBackend("/nonexistent.gguf", bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not be an accepted GGUF precision")


def test_llamacpp_accepts_the_two_published_schemes():
    # Q8_0 is the one the granularity argument rests on: 8-bit like PyTorch's
    # per-tensor dynamic int8, so the comparison holds bit width constant.
    assert set(backends.GGUF_PRECISIONS) == {"gguf-q4_k_m", "gguf-q8_0"}
    for prec in backends.GGUF_PRECISIONS:
        b = backends.LlamaCppBackend("/nonexistent.gguf", prec)
        assert b.name == f"llamacpp/{prec}"


def test_llamacpp_load_failure_says_what_to_do():
    # Either the package or the weights will be missing for most readers. Both
    # paths must name models/README.md rather than surfacing an ImportError or
    # a raw llama.cpp abort.
    b = backends.LlamaCppBackend("/nonexistent/weights.gguf", "gguf-q8_0")
    try:
        b.load()
    except (RuntimeError, FileNotFoundError) as e:
        assert "models/README.md" in str(e), f"unhelpful message: {e}"
    else:
        raise AssertionError("load() of absent weights should have raised")


def test_llamacpp_param_bytes_is_on_disk_size():
    import tempfile
    b = backends.LlamaCppBackend("/nonexistent/weights.gguf", "gguf-q8_0")
    assert b.param_bytes == 0, "a missing file must degrade to 0, not raise"
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
        f.write(b"\0" * 4096)
        f.flush()
        assert backends.LlamaCppBackend(f.name, "gguf-q8_0").param_bytes == 4096


def _run_all() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed, skipped = [], []
    for name, fn in fns:
        try:
            fn()
            print(f"  pass  {name}")
        except unittest.SkipTest as e:
            skipped.append((name, e))
            print(f"  skip  {name}: {e}")
        except Exception as e:                      # noqa: BLE001
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    tail = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n{len(fns) - len(failed) - len(skipped)}/{len(fns)} passed{tail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
