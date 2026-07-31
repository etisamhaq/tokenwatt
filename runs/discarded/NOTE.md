# Discarded run — kept deliberately

This run of the Qwen2.5-0.5B-Instruct matrix was thrown away. It is here because
what a spoiled measurement looks like is worth being able to point at.

Two independent problems:

1. **Half-measured energy.** RAPL read permissions were granted partway through
   the run. The runner probed for access on every repeat, so configurations 1–7
   recorded no energy and 8–12 did — and with no idle baseline, those were gross
   joules. `analyze.py` then computed a Pareto frontier over only the
   configurations that happened to have numbers, which is how `int8-dynamic`
   appeared "non-dominated" at +926% perplexity.

2. **Contended machine.** Other work (a test suite, several torch-importing
   processes) ran concurrently during the early configurations, inflating their
   timings relative to the later ones.

Fixes now in place: `runner.py` decides energy availability once at startup and
holds it for the whole run, and aborts outright if RAPL is readable but the idle
baseline fails rather than recording gross-only energy. The lesson that a
measurement harness needs to fail loudly rather than produce a plausible-looking
mixed dataset is the reason this directory exists.

The clean replacement is in `../qwen2.5-0.5b/`.
