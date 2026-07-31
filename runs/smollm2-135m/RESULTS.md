# Results

Model: `HuggingFaceTB/SmolLM2-135M`  ·  3 repeats per configuration  ·  24 new tokens per generation

Machine: x86_64 — P 6c/12t @ 4.8GHz, E 8c/8t @ 3.8GHz, LP-E 2c/2t @ 2.5GHz

> **Energy was not measured** — RAPL counters were unreadable for this run, so cost is reported as seconds per token. See README for how to grant access; the same run then yields joules.

## Per-configuration summary

| config | tok/s (IQR) | s/token | perplexity | Δquality | needle | on frontier |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu/bf16/E/t8` | 23.6 (±5.5) | 0.0424 | 11.748 | +0.4% | 0/3 | yes |
| `cpu/bf16/LP-E/t2` | 8.0 (±0.8) | 0.1245 | 11.748 | +0.4% | 0/3 |  |
| `cpu/bf16/P-1t/t6` | 22.4 (±0.7) | 0.0447 | 11.748 | +0.4% | 0/3 |  |
| `cpu/fp32/E/t8` | 16.7 (±1.4) | 0.0600 | 11.706 | +0.0% | 0/3 |  |
| `cpu/fp32/LP-E/t2` | 9.8 (±0.2) | 0.1022 | 11.706 | +0.0% | 0/3 |  |
| `cpu/fp32/P-1t/t6` | 20.2 (±0.5) | 0.0495 | 11.706 | +0.0% | 0/3 | yes |
| `cpu/int8-dynamic/E/t8` | 15.6 (±1.4) | 0.0640 | 38.470 | +228.6% | 1/3 |  |
| `cpu/int8-dynamic/LP-E/t2` | 11.0 (±0.1) | 0.0912 | 38.529 | +229.1% | 0/3 |  |
| `cpu/int8-dynamic/P-1t/t6` | 15.4 (±3.5) | 0.0648 | 37.715 | +222.2% | 0/3 |  |

## Non-dominated configurations

- `cpu/bf16/E/t8`
- `cpu/fp32/P-1t/t6`

Everything absent from that list is dominated: some other configuration is at least as cheap *and* at least as accurate, so it should never be chosen.

