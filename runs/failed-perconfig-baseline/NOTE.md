# Failed experiment: per-configuration idle baselines

An attempt to fix the baseline problem described in `FINDINGS.md` section 6. It
made the data worse, and this directory is the record of how.

## What was tried

Instead of one idle baseline measured before the run, measure it per
configuration, immediately after model load and warmup, under the same affinity
mask — so that thermal state and pinning would match the generations it is
subtracted from.

## What happened

Gross power was stable. The baseline was not.

| bf16 config | gross | baseline | net |
| --- | --- | --- | --- |
| P-1t | 15.45 W | 7.12 W | 8.33 W |
| P-smt | 14.02 W | 13.18 W | 0.85 W |
| E | 15.57 W | 12.57 W | 3.00 W |
| LP-E | 15.19 W | 5.96 W | 9.23 W |

Gross power varied 10% across those four. The baseline varied **2.2x**. Net
energy therefore varied **11x**, essentially all of it artifact. Across all
configurations the baseline spanned 5.52–13.18 W, a 103% spread of its own median.

## Why

Measuring immediately after a warmup generation captures the CPU *decaying out
of turbo*, not idle. How far it has decayed depends on timing that was never
controlled, so each configuration got a baseline sampled at an arbitrary point on
a decay curve.

The lesson is not "per-configuration baselines are wrong" — they are the more
principled choice. It is that a baseline must be **settled and verified stable**
before it is trusted, and an unsettled baseline is worse than a slightly stale one.

## What changed as a result

`runner.py` now takes `--baseline {run,per-config}`, defaulting to `run`. The
`per-config` path waits a settle period, then takes two consecutive windows and
requires them to agree within 10% before accepting either, recording
`baseline_stable` per row so an unconverged baseline is visible in the data.

## Also worth noting

This run is truncated: configuration 12 of 12 (`int8-dynamic/LP-E`) never
completed, so only 11 configurations are present. Another reason not to use it.

The clean dataset is in `../qwen2.5-0.5b/`, measured with a single run-level
baseline taken on a quiet machine.
