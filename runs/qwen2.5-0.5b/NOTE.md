# Superseded by `runs/qwen2.5-0.5b-v3/`

The original dataset. Two defects, both discovered on 2026-10-08:

**1. Measurement length.** 24 generated tokens gives a window of about 2.8 s.
The idle baseline (10.45 W) exceeded the net signal, and throughput was still on
the frequency and cache ramp when the trials ran. Three conclusions drawn from
this file did not survive remeasurement at 256 tokens: `fp32/E` as the cheapest
fp32 configuration, SMT helping fp32 while hurting bf16, and int8 energy being
unchanged against bf16.

**2. Repeats were not repeats.** `runner.py` selected the prompt as
`PROMPTS[rep % len(PROMPTS)]`, so "5 repeats" over 4 prompts measured prompt 0
twice and each of the others once. The spread across those rows mixed
prompt-to-prompt variation into what was reported as measurement noise, and no
paired analysis was possible.

**Still valid.** The headline direction. The LP-E penalty measured 2.31x here
and 2.306x [2.301, 2.314] in v3 — the finding was right, the interval was not
computable.
