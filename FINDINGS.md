# Findings

Interpretation of [`runs/qwen2.5-0.5b-v3/RESULTS.md`](runs/qwen2.5-0.5b-v3/RESULTS.md),
which is generated. This file is written by hand and is where claims live.

> **Rewritten 2026-10-08 against the v3 dataset.** The previous version rested
> on 24-token generations with a single 8-token warmup. Three of its claims did
> not survive remeasurement at 256 tokens with a settled warmup, and one
> reversed outright. The superseded run and the reason it was discarded are in
> [`runs/warmup-ablation/`](runs/warmup-ablation/NOTE.md).

**Setup.** Qwen2.5-0.5B-Instruct, 12 configurations (3 precisions x 4 core
placements), 4 fixed prompts x 5 true repeats = 20 measurements per
configuration, 240 rows. 256 new tokens per generation. Energy from RAPL
`package-0`, net of an 11.23 W idle baseline measured on a quiet machine before
any load. Each configuration warmed at full measurement length until two
consecutive warmups agreed within 5% (2-5 rounds were needed; one was never
enough). Intel Core Ultra 7 155H: 6 P-cores (SMT), 8 E-cores, 2 LP-E cores.

| config | tok/s | J/token | gross W | net W | perplexity | needle |
| --- | --- | --- | --- | --- | --- | --- |
| **bf16 / P-1t** | 16.1 | **0.1758** | 14.00 | 2.77 | 12.427 | 3/3 |
| bf16 / P-smt | 15.1 | 0.1878 | 14.01 | 2.77 | 12.427 | 3/3 |
| bf16 / E | 14.4 | 0.1987 | 14.01 | 2.78 | 12.427 | 3/3 |
| bf16 / LP-E | 7.0 | 0.4053 | 14.00 | 2.77 | 12.427 | 3/3 |
| fp32 / P-1t | 12.0 | 0.2313 | 14.00 | 2.77 | 12.449 | 3/3 |
| fp32 / E | 11.8 | 0.2357 | 14.00 | 2.77 | 12.449 | 3/3 |
| fp32 / P-smt | 11.1 | 0.2515 | 14.00 | 2.77 | 12.449 | 3/3 |
| fp32 / LP-E | 7.0 | 0.4019 | 14.00 | 2.77 | 12.449 | 3/3 |
| int8-dynamic / P-1t | 27.7 | 0.1281 | 14.78 | 3.55 | 109.953 | 0/3 |
| int8-dynamic / P-smt | 22.6 | 0.1470 | 14.59 | 3.36 | 109.522 | 0/3 |
| int8-dynamic / E | 22.5 | 0.1998 | 15.70 | 4.47 | 110.608 | 0/3 |
| int8-dynamic / LP-E | 11.2 | 0.3209 | 14.82 | 3.59 | 106.391 | 0/3 |

## 1. Energy per token is duration, and the identity is exact

Paired per-prompt ratios against the performance cores, at fixed quality (bf16,
byte-identical output in all four placements):

```
P-1t     0.1758 J/tok   16.1 tok/s    baseline
P-smt    0.1878 J/tok   15.1 tok/s    1.073x  [1.066, 1.081]
E        0.1987 J/tok   14.4 tok/s    1.148x  [1.117, 1.189]
LP-E     0.4053 J/tok    7.0 tok/s    2.306x  [2.301, 2.314]
```

The low-power island costs **2.306x** the energy per token of the performance
cores. That is the headline; the mechanism is the finding.

**Gross package power is 14.00 W for every fp32 and bf16 configuration**, across
islands spanning 7.0 to 16.1 tok/s. Not approximately: the medians are 14.00,
14.00, 14.01, 14.00, 14.00, 14.00, 14.00, 14.01. This is not a power cap --
RAPL's long-term limit on this part is 40 W.

With power constant, energy per token reduces to duration, and that prediction
is testable. Each configuration's measured energy ratio against the inverse of
its measured throughput ratio:

| | energy ratio | 1 / throughput | difference |
| --- | --- | --- | --- |
| fp32 / P-smt | 1.086 | 1.088 | -0.16% |
| fp32 / E | 1.021 | 1.017 | +0.42% |
| fp32 / LP-E | 1.737 | 1.727 | +0.62% |
| bf16 / P-smt | 1.073 | 1.069 | +0.42% |
| bf16 / E | 1.148 | 1.136 | +1.10% |
| bf16 / LP-E | 2.306 | 2.296 | +0.43% |

Agreement within 1.1% everywhere. The low-power island does not run at low power
on this workload -- it runs at the same power for 2.3x as long. **Race to idle,
measured.**

The reason is that decode is memory-bandwidth-bound. A 0.5 B-parameter model in
fp32 streams roughly 2 GB of weights per generated token, so the package power
budget goes to the memory subsystem and static draw rather than to core
execution. Which island issues the loads barely matters.

## 2. The identity breaks exactly where the mechanism says it should

int8-dynamic is the one precision that does not hold 14.00 W: it spans
13.96-15.92 W, median 14.81 W. It is also the one precision where the identity
fails:

```
int8 / P-smt    energy 1.142   1/tput 1.220    -6.39%
int8 / E        energy 1.558   1/tput 1.222   +27.51%
int8 / LP-E     energy 2.509   1/tput 2.447    +2.54%
```

Against a worst case of 1.10% for fp32 and bf16. Quantization moves work off the
memory system and back into the cores, power stops being flat, and duration
stops being the whole story. The exception is the best evidence that section 1's
explanation is the right one rather than a coincidence of this platform.

`int8 / E` is the outlier worth naming: it draws 15.70 W, the highest of any
configuration measured, and lands 27.5% worse than its duration predicts.

## 3. bf16 beats fp32 -- but only where the hardware supports it

On P-1t, fp32 costs **1.319x [1.309, 1.330]** the energy of bf16 and runs at
0.745x the throughput, with perplexity marginally worse (12.449 against 12.427).
No configuration on the performance cores makes fp32 the right choice.

The advantage is not uniform across islands:

| island | fp32/bf16 energy | fp32/bf16 tok/s |
| --- | --- | --- |
| P-1t | 1.319 [1.309, 1.330] | 0.745 |
| P-smt | 1.335 [1.303, 1.367] | 0.732 |
| E | 1.173 [1.132, 1.205] | 0.833 |
| **LP-E** | **0.994 [0.989, 1.000]** | **0.991** |

**On the low-power island the bf16 advantage disappears entirely.** fp32 and
bf16 reach the same 7.0 tok/s and the same ~0.40 J/token there. The likely
explanation is that the LP-E cores lack the bf16 instruction support the P-cores
have, so bf16 executes at fp32 rates; confirming it needs instruction-level
profiling this harness does not do.

The practical consequence is that **precision and placement cannot be chosen
independently.** A scheduler migrating a bf16 workload to the low-power island
does not merely slow it by the island's ratio -- it also forfeits the entire
benefit of the precision it was running.

## 4. Per-tensor int8 does save energy, and is still the wrong choice

This reverses the previous version of this file, and the reversal is
instructive.

int8-dynamic on P-1t costs **0.730x [0.726, 0.733]** the energy per token of
bf16 -- a real 27% saving, with the interval nowhere near 1.0. The 24-token
dataset reported 0.929x with an interval spanning unity and concluded that
quantization "bought latency, not energy." At 256 tokens with a settled warmup
that is wrong: the saving is there and it is unambiguous.

What has not changed is that the saving is unusable:

- perplexity **12.427 -> 109.953**, a 785% increase
- needle retrieval **3/3 -> 0/3**, gone entirely

The honest statement is a cost/quality trade rather than an absence of benefit.
That is a stronger claim than the one it replaces, and a more useful one: the
energy is real and you still should not take it.

The point stands about **granularity, not bit width**. PyTorch dynamic
quantization uses one scale per tensor. GGUF K-quants and GPTQ use per-block or
per-channel scales and hold quality at 4 bits, far below 8. The conclusion is
not "8-bit is too aggressive", it is "per-tensor scaling is the wrong scheme",
and testing K-quants is the obvious next experiment.

**And it does not shrink the KV cache.** Dynamic quantization touches only
Linear weights, so the cache stays at the unquantized dtype. For long-context
work, where the cache rather than the weights dominates footprint, per-tensor
int8 offers nothing on the axis that matters while bf16 offers a factor of two.

## 5. Placement changes the arithmetic, but not the output

Greedy decode produced **byte-identical generated text on all four islands**,
for every precision and prompt (`output_sha` identical throughout). Placement
affects cost, not tokens.

The arithmetic underneath is not identical. int8 perplexity varies by placement
across a 4% range -- 106.391 on LP-E against 110.608 on E -- while fp32 and bf16
perplexity is constant to three decimals. Different islands dispatch int8
kernels differently, the numerics differ measurably, and greedy argmax happens
to be robust to the difference.

Two consequences. Any claim of bit-exact reproducibility across islands is false
for quantized models. And a sampling-based decoder has no argmax margin to
absorb the difference.

## 6. The Pareto frontier is necessary, not sufficient

The frontier contains `bf16/P-1t`, `int8-dynamic/P-1t` and
`int8-dynamic/P-smt`. Two of the three score 0/3 on needle retrieval and carry a
785% perplexity penalty. They are non-dominated because they are genuinely
cheaper, and genuinely unusable.

This is a limitation of the method rather than a result: a frontier tells you
what is not dominated, not what is worth choosing. The actionable output is
**bf16 on the performance cores, one thread per core**, and the frontier must be
read alongside the quality column rather than instead of it.

## 7. SMT costs throughput on every precision

```
fp32   P-1t 12.0 -> P-smt 11.1    -7.5%
bf16   P-1t 16.1 -> P-smt 15.1    -6.2%
int8   P-1t 27.7 -> P-smt 22.6   -18.4%
```

The previous version reported an inversion -- SMT helping fp32 by 38% while
hurting bf16 by 20% -- and called it unexplained. There is no inversion. SMT
costs throughput on all three precisions, which is what a memory-bound workload
predicts: the sibling thread adds cache contention and contributes no useful
parallelism, because the bottleneck is off-core. The penalty is largest for
int8, the precision that does the most arithmetic per byte and so contends
hardest.

## 10. Granularity, not bit width -- tested at matched bit width

Section 4 argued that per-tensor scaling rather than 8-bit arithmetic was the
problem. Comparing per-tensor int8 against a per-block 4-bit scheme would vary
two things at once, so we added `Q8_0`: per-block, at **the same eight bits**.
Bit width is then fixed and only granularity differs.

| scheme | engine | bits | granularity | perplexity | vs bf16 |
| --- | --- | --- | --- | --- | --- |
| bf16 | PyTorch | 16 | -- | 12.427 | -- |
| `Q8_0` | llama.cpp | 8 | per-block | 13.013 | **+4.7%** |
| int8-dynamic | PyTorch | 8 | per-tensor | 109.953 | **+785%** |
| `Q4_K_M` | llama.cpp | 4 | per-block | 13.346 | +7.4% |

At eight bits, per-block costs 4.7% and per-tensor costs 785%: a factor of
**167** in quality damage from granularity alone, bit width held constant.
Halving to four bits with per-block scales costs 7.4% -- still two orders of
magnitude less than the 8-bit per-tensor scheme. Bit width is not the operative
variable; granularity is.

## 11. The flat-power result reproduces on a second engine

llama.cpp is a different runtime with its own kernels, written in C++ rather
than dispatched through PyTorch. It shows the same invariance:

| scheme | P-1t | LP-E | energy ratio | 1/throughput | agreement |
| --- | --- | --- | --- | --- | --- |
| `Q4_K_M` | 31.7 tok/s @ 14.01 W | 19.5 @ 14.01 W | 1.660x | 1.678 | -1.07% |
| `Q8_0` | 30.1 tok/s @ 14.00 W | 16.4 @ 13.97 W | 1.876x | 1.884 | -0.43% |

Gross package power is 13.99-14.01 W across both schemes and both islands,
spanning 16.4 to 31.7 tok/s -- against 14.00 W measured under PyTorch. Four
independent observations of the same figure now, on two engines and two models.
The duration identity holds to within 1.1% here too.

The island penalty is smaller under llama.cpp (1.66-1.88x against 2.306x for
bf16/PyTorch), which is expected: the ratio depends on how completely an engine
exploits the performance cores. The direction and mechanism are unchanged; the
magnitude is engine-dependent and should be reported as such.

> **Cross-engine energy numbers are not controlled.** llama.cpp is simply a
> faster engine, so comparing its J/token against PyTorch's confounds
> quantization with implementation. The within-engine comparisons above and the
> perplexity table in section 10 are unaffected -- every perplexity is scored
> over the same corpus with the same tokenizer.

## What changed from the 24-token dataset, and why

| claim | 24-token result | 256-token result | verdict |
| --- | --- | --- | --- |
| LP-E costs more energy | 2.31x | 2.306x [2.301, 2.314] | holds, now with a 0.6%-wide interval |
| bf16 beats fp32 | +86% throughput | +34% throughput | holds, magnitude was inflated |
| fp32/E cheapest fp32 config | yes | no -- P-1t is | artifact |
| SMT helps fp32, hurts bf16 | yes | no -- hurts both | artifact |
| int8 energy unchanged | 0.929x, unresolvable | 0.730x [0.726, 0.733] | **reversed** |
| placement changes perplexity | noted, unexplained | real, int8 only, output unaffected | refined |

The single cause of every artifact was measurement length. At 24 tokens the
window is about 2.8 s, the idle baseline exceeds the net signal, and throughput
is still on the frequency and cache ramp. At 256 tokens the window is 15-40 s
and the measurement is of steady-state decode.

## 8. The island penalty grows as models shrink

The obvious objection to a sub-1B study is that it describes toy models. The
objection runs the wrong way: the penalty for choosing the wrong island is
*larger* on the smaller model, and small models are what on-device deployment
actually runs.

Repeating the full matrix on SmolLM2-135M (240 rows, same protocol):

| model | bf16 LP-E vs P-1t | identity error |
| --- | --- | --- |
| Qwen2.5-0.5B | 2.306x [2.301, 2.314] | +0.43% |
| SmolLM2-135M | **3.712x [3.159, 4.342]** | +0.68% |

The penalty is 61% larger on the 135M model, and the duration identity from
section 1 holds on both. The mechanism is visible in the throughput: the
performance cores exploit the smaller model (16.1 -> 34.7 tok/s, a 2.2x gain)
while the low-power island barely moves (7.0 -> 8.3, only 1.2x). Less weight
traffic means the workload is relatively less memory-bound, so core capability
starts to matter -- and that is exactly what the low-power island does not have.

Package power stayed flat on this model too: fp32 median 14.00 W, bf16 14.00 W,
int8 14.26 W. Third independent confirmation, on a different model, with a
different idle baseline (9.66 W).

**The same mechanism erases the quantization benefit.** int8 against bf16 on
SmolLM2 costs **1.053x [1.007, 1.101]** the energy -- marginally *more*
expensive, with the interval excluding 1.0 on the wrong side, against 0.730x on
Qwen. Quantization saves energy only to the extent the model is memory-bound.
At 0.5 B it saves 27%; at 135 M it costs 5%, and perplexity still rises from
13.4 to 40.8.

> **The needle probe carries no signal at this model size.** SmolLM2-135M is a
> base model rather than instruction-tuned, and scores 0/3 at bf16 and 1/3 at
> int8. That is the probe failing, not int8 outperforming bf16, and the column
> should not be read as a quality comparison for this model.

## 9. The result survives a 4x longer context

Generating 1024 tokens instead of 256 quadruples the KV cache and slows both
islands, but barely moves the ratio:

| | P-1t | LP-E | KV cache | LP-E vs P-1t energy |
| --- | --- | --- | --- | --- |
| 256 tokens | 16.1 tok/s @ 14.00 W | 7.0 @ 14.00 W | 3,204 KiB | 2.306x [2.301, 2.314] |
| 1024 tokens | 14.0 tok/s @ 14.00 W | 6.2 @ 14.00 W | 12,384 KiB | 2.251x (see caveat) |

Both islands lose about 12% throughput as attention over a longer context grows
more expensive, and package power is still exactly 14.00 W on both at four
times the context length.

> **Caveat: this comparison has no confidence interval.** The long-context run
> used 2 prompts rather than 4, which leaves only two paired ratios -- below the
> three that `bootstrap_ci` requires before it will report an interval. The
> guard behaved correctly; the experiment design did not. The point estimate is
> sound and the conclusion is stable, but a robustness check without an interval
> invites exactly the question it exists to answer, and this should be re-run at
> 4 prompts before the result is relied on.

## Confounds controlled

Idle baseline measured once on a quiet machine, subtracted, reported, and
recorded per row; warmup at full measurement length until throughput settles
within 5%, with `warmup_settled` recorded so an unconverged configuration is
visible; 20 measurements per configuration as a 4x5 prompt-by-repeat cross;
paired per-prompt ratios with bootstrap 95% intervals, and no directional claim
where an interval spans 1.0; identical prompts, seed and corpus throughout;
affinity restored between configurations; energy availability fixed once per
run; net energy recorded unclamped.

## Not covered

- **One machine, one SoC.** Nothing here generalises beyond Meteor Lake. The
  result is explicitly a property of a platform whose package power is flat
  across islands for this workload; a platform where it is not would behave
  differently, and one has been reported on ARM mobile.
- **Small models only.** 0.5 B and 135 M. Section 8 shows the ordering holds
  across that range and that the penalty grows as the model shrinks, but 7 B --
  where memory pressure is far higher and power may cease to be flat -- is
  untested and remains the most likely place these conclusions break.
- **iGPU and NPU unmeasured.** Both present and idle. Section 1 makes them more
  interesting, not less: if energy is set by duration at fixed power, an engine
  that is both faster and genuinely lower-power should win by more than either
  factor alone.
- **The bf16/LP-E explanation is a hypothesis.** Instruction-level profiling
  would confirm or refute the missing-bf16-support account in section 3.
- **Perplexity and one needle probe** are a thin quality measure. They caught
  the int8 collapse decisively; a real evaluation suite would be better.
- **No hardware performance counters**, so the SMT and LP-E mechanisms in
  sections 7 and 3 remain inference from throughput rather than direct evidence.
- **The machine was not perfectly quiet.** Some trials show excursions
  consistent with background load during the run. That widens intervals rather
  than biasing them, so the reported effects are if anything understated.
