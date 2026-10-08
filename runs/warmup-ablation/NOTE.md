# Discarded: inadequate warmup (kept as evidence)

5 of 12 configurations, Qwen2.5-0.5B, 256 new tokens, 4 prompts x 5 repeats.
Stopped deliberately once the flaw below was identified.

## The flaw

The harness warmed each configuration with a single 8-token generation. That
was adequate when measurements were 24 tokens long. At 256 tokens it is not:
decode throughput was still climbing when the measured trials began.

Within `cpu/fp32/P-1t/t6`, in trial order:

```
p0 r0   9.95 tok/s        p2 r0  12.90 tok/s
p0 r4  12.88 tok/s        p3 r0  14.14 tok/s
p1 r0   9.02 tok/s        p3 r4  13.61 tok/s
p1 r3  13.01 tok/s
```

Median of the first five trials 11.51 tok/s, of the last five 13.87 tok/s:
a **+20.5%** drift inside one configuration, at fixed prompt set and placement.

## Why it is not self-cancelling

The ramp is not the same size everywhere — +20.5% for `fp32/P-1t`, +20.4% for
`fp32/E`, +7.7% for `fp32/P-smt`, -3.7% for `fp32/LP-E`, +1.8% for
`bf16/P-1t`. Configurations measured against each other therefore sit at
different points on their own ramps, and the comparison inherits the
difference. It landed squarely on the precision comparison: this dataset
reports bf16/P-1t at 9.8 tok/s against fp32/P-1t at 12.5 tok/s, reversing the
v1 finding that bf16 dominates fp32. That reversal is not trustworthy and is
the reason this run was stopped.

## What replaced it

`warm_until_settled()` in `runner.py`: generate at the full measurement length
until two consecutive warmups agree within 5%, minimum two rounds, maximum six.
Each row records `warmup_rounds` and `warmup_settled`, so a configuration that
never converged is visible in the data rather than silently treated as warm.

## What this run is still good for

It is the evidence that warmup length matters, and it is kept for that reason.
Gross package power is also worth noting: 13.88-14.18 W across every trial of
every configuration here, against throughputs spanning 9.0-14.1 tok/s, with
RAPL PL1 at 40 W. The invariance is a property of the workload, not a power cap.
