# Findings

Interpretation of [`runs/qwen2.5-0.5b/RESULTS.md`](runs/qwen2.5-0.5b/RESULTS.md),
which is generated. This file is written by hand and is where claims live.

**Setup.** Qwen2.5-0.5B-Instruct, 12 configurations (3 precisions × 4 core
placements), 5 repeats each, 24 new tokens per generation. Energy from RAPL
`package-0`, net of a 10.45 W idle baseline measured on a quiet machine before any
load. Intel Core Ultra 7 155H: 6 P-cores (SMT), 8 E-cores, 2 LP-E cores.

| config | tok/s | J/token | net W | perplexity | needle |
| --- | --- | --- | --- | --- | --- |
| **bf16 / P-1t** | 17.7 | **0.2692** | 4.78 | 11.091 | 3/3 |
| bf16 / P-smt | 14.8 | 0.2990 | 4.41 | 11.091 | 3/3 |
| bf16 / E | 14.4 | 0.3534 | 5.10 | 11.091 | 3/3 |
| bf16 / LP-E | 7.4 | 0.6210 | 4.58 | 11.091 | 3/3 |
| fp32 / P-1t | 9.5 | 0.3755 | 3.55 | 11.178 | 3/3 |
| fp32 / P-smt | 13.1 | 0.3031 | 3.97 | 11.178 | 3/3 |
| fp32 / E | 12.6 | 0.2891 | 3.64 | 11.178 | 3/3 |
| fp32 / LP-E | 7.3 | 0.5817 | 4.26 | 11.178 | 3/3 |
| int8-dynamic / P-1t | 36.3 | 0.2659 | 9.65 | 113.763 | 0/3 |
| int8-dynamic / P-smt | 27.9 | 0.2906 | 8.09 | 113.763 | 0/3 |
| int8-dynamic / E | 22.1 | 0.3404 | 7.52 | 113.763 | 0/3 |
| int8-dynamic / LP-E | 13.3 | 0.6408 | 8.53 | 118.205 | 0/3 |

The `net W` column is what makes the rest of this legible, so its structure is
worth stating up front:

- **Within a precision, net power barely moves across core islands.** fp32 spans
  3.55–4.26 W, bf16 spans 4.41–5.10 W, int8 spans 7.52–9.65 W — 15–25% each,
  against a 5× spread in throughput.
- **Between precisions, net power scales with work rate.** int8 draws roughly 2×
  what bf16 does, and does roughly 2× the work per second.

## 1. The efficiency cores are not energy-efficient here

Ranked by joules per token at fixed quality (bf16, byte-identical output in all
four placements):

```
P-1t    0.2692 J/tok    17.7 tok/s    4.78 W      baseline
P-smt   0.2990 J/tok    14.8 tok/s    4.41 W      +11%
E       0.3534 J/tok    14.4 tok/s    5.10 W      +31%
LP-E    0.6210 J/tok     7.4 tok/s    4.58 W      +131%
```

The low-power island costs **2.31× the energy per token** of the performance
cores. The mechanism is now visible rather than inferred: net power on LP-E
(4.58 W) and P-1t (4.78 W) differ by **4%**, while throughput differs by **2.39×**.
The energy ratio is, to within measurement error, the inverse throughput ratio.

So the low-power island does not run at low power on this workload — it runs at
the same power and takes 2.4× as long. **Race-to-idle, measured.** The practical
advice for local LLM inference on this class of SoC is the opposite of the
intuitive one: use the performance cores and finish sooner.

Two things follow that are worth separating:

- The **ordering is robust**, being driven by a 2.4× duration difference against a
  4% power difference. No plausible baseline error inverts that.
- The framing "efficiency cores save energy" fails not because the saving is small
  but because there is **no power saving to trade** in the first place.

A throughput-only study would have reached the right ranking here by accident. On
the smaller SmolLM2-135M it would have reached the wrong one: E-cores beat P-cores
on tokens per second there, which read as an argument for the efficiency island.

## 2. bf16 strictly dominates fp32

On P-1t, bf16 against fp32:

- **28% less energy** per token (0.2692 vs 0.3755)
- **86% more throughput** (17.7 vs 9.5 tok/s)
- **Marginally better perplexity** (11.091 vs 11.178)

Better on every axis measured; no configuration in this study makes fp32 the right
choice. bf16 does draw more power (4.78 vs 3.55 W) because it is doing more work
per second, but nowhere near enough to offset the speedup.

The perplexity difference is within noise. The claim is only that bf16 is
certainly not worse, at half the weight traffic.

## 3. int8-dynamic buys nothing and costs everything

This is the result that changed most once net power was in view.

The quantized kernels are genuinely fast: **2.05× the throughput of bf16** on P-1t
(36.3 vs 17.7 tok/s). But they draw **2.02× the power** (9.65 vs 4.78 W). The two
cancel almost exactly, leaving joules per token essentially unchanged — 0.2659
against 0.2692, a 1.2% difference.

So reading that 1.2% as "a trivial saving" was wrong in an interesting way. It is
neither noise nor a saving: **it is what happens when you double the work rate and
double the power draw.** Quantization here bought latency, not energy.

The quality cost is total:

- perplexity **11.091 → 113.763**, a 10× increase
- needle retrieval **3/3 → 0/3**, gone entirely

This is a point about **granularity, not bit width**. PyTorch dynamic quantization
uses one scale per tensor. GGUF K-quants and GPTQ use per-block or per-channel
scales and are widely reported to hold quality at 4 bits — far below 8. The
conclusion is not "8-bit is too aggressive", it is "per-tensor scaling is the wrong
scheme", and testing K-quants is the obvious next experiment.

The damage also grows with model size: the same scheme cost +228% perplexity on
SmolLM2-135M and +926% here.

## 4. Being on the Pareto frontier is not the same as being sensible

The frontier contains `bf16/P-1t` and `int8-dynamic/P-1t`, the latter only because
it is 1.2% cheaper in energy — technically non-dominated, practically absurd, since
it pays for that with a 10× perplexity increase.

Worth stating because it is a limitation of the method rather than a result: a
frontier tells you what is *not dominated*, not what is *worth choosing*. The
actionable output is **bf16 on the performance cores, one thread per core**, and
the frontier has to be read alongside the quality column rather than instead of it.

## 5. SMT helps fp32 and hurts bf16

An inversion I cannot fully explain:

- **fp32**: P-smt 13.1 tok/s beats P-1t 9.5 — SMT is 38% *faster*
- **bf16**: P-1t 17.7 tok/s beats P-smt 14.8 — SMT is 20% *slower*

A plausible reading is that fp32 leaves enough per-core stall time for a second
hyperthread to fill, while bf16 already saturates the per-core memory path so the
SMT sibling only adds cache contention. That is a hypothesis, not a measurement.
Testing it needs hardware performance counters — cache misses, memory stall
cycles — which this harness does not read.

## 6. The idle baseline is the weakest part of the measurement

Net energy is gross minus `idle_watts × duration`, and the baseline is the least
trustworthy term in that expression.

**What was tried and failed.** The obvious improvement is to measure the baseline
per configuration rather than once per run. Implemented naively — immediately after
model load and warmup — it made the data substantially worse. Gross power across
four bf16 placements varied 10%; the per-config baselines varied **2.2×**
(5.96–13.18 W), so net energy varied **11×**, almost all of it artifact. The record
is in [`runs/failed-perconfig-baseline/`](runs/failed-perconfig-baseline/NOTE.md).

The cause: a baseline sampled right after a burst of work captures the CPU decaying
out of turbo, not idle, and how far it has decayed depends on timing nobody
controlled.

**What is implemented now.** `runner.py --baseline {run,per-config}`, defaulting to
`run`. The `per-config` path waits a settle period, then requires two consecutive
measurement windows to agree within 10% before accepting either, and records
`baseline_stable` per row so an unconverged baseline is visible rather than
silently folded into a result. The numbers in this document use `run`.

**What this means for the results above.** Absolute joules-per-token figures carry
a systematic baseline error that is not fully quantified. Every *relative* claim in
sections 1–3 is safe, because each rests on a ratio far larger than any plausible
baseline error: 2.31× for the island ordering, 1.86× for bf16 against fp32, 10× for
the quality collapse. The one comparison that is *not* safe on its own is the 1.2%
int8-versus-bf16 energy gap — which is why section 3 leans on the power and
throughput measurements rather than on that difference.

Net energy is recorded **unclamped**. A negative value means the baseline exceeded
the measurement, and flooring it at zero would hide exactly the error worth seeing.
In the clean run `package-0` never went negative; the `uncore` domain did, which is
expected — its baseline is around 0.2 W and dominated by noise.

## Confounds controlled

Idle baseline subtracted; one warmup generation per configuration discarded for
thermal soak; 5 repeats with IQR reported; identical prompts and seed throughout;
affinity restored between configurations; energy availability fixed once per run so
a permission change mid-run cannot produce a half-measured dataset.

## Not covered

- **One machine, one SoC.** Nothing here generalises beyond Meteor Lake.
- **iGPU and NPU unmeasured.** Both are present and idle. Section 1 makes them more
  interesting, not less: if energy per token is set by duration at roughly constant
  power, an engine that is both faster *and* genuinely lower-power should win by
  more than either factor alone. Nothing in this data predicts by how much.
- **Small models only.** 0.5B and 135M. Whether the island ordering survives at 7B,
  where memory pressure is far higher and power may stop being flat across
  placements, is untested — and is the most likely place these conclusions break.
- **Perplexity and one needle probe** are a thin quality measure. They caught the
  int8 collapse decisively; a real evaluation suite would be better.
- **24 tokens per generation** is short. KV cache growth is recorded but no
  conclusion here depends on it — the one instrument that has not yet earned its
  place.
- **No hardware performance counters**, so section 5 stays a hypothesis.
