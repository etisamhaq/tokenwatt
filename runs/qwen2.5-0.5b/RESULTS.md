# Results

Model: `Qwen/Qwen2.5-0.5B-Instruct`  ·  5 repeats per configuration  ·  24 new tokens per generation

Machine: x86_64 — P 6c/12t @ 4.8GHz, E 8c/8t @ 3.8GHz, LP-E 2c/2t @ 2.5GHz

Energy from RAPL `package-0`, net of a per-configuration idle baseline measured after load and warmup under the same affinity mask.

## Per-configuration summary

| config | tok/s (IQR) | J/token | perplexity | Δquality | needle | on frontier |
| --- | --- | --- | --- | --- | --- | --- |
| `cpu/bf16/E/t8` | 14.4 (±1.1) | 0.3534 | 11.091 | +0.0% | 3/3 |  |
| `cpu/bf16/LP-E/t2` | 7.4 (±0.1) | 0.6210 | 11.091 | +0.0% | 3/3 |  |
| `cpu/bf16/P-1t/t6` | 17.7 (±0.3) | 0.2692 | 11.091 | +0.0% | 3/3 | yes |
| `cpu/bf16/P-smt/t12` | 14.8 (±0.2) | 0.2990 | 11.091 | +0.0% | 3/3 |  |
| `cpu/fp32/E/t8` | 12.6 (±0.3) | 0.2891 | 11.178 | +0.8% | 3/3 |  |
| `cpu/fp32/LP-E/t2` | 7.3 (±0.2) | 0.5817 | 11.178 | +0.8% | 3/3 |  |
| `cpu/fp32/P-1t/t6` | 9.5 (±0.3) | 0.3755 | 11.178 | +0.8% | 3/3 |  |
| `cpu/fp32/P-smt/t12` | 13.1 (±0.0) | 0.3031 | 11.178 | +0.8% | 3/3 |  |
| `cpu/int8-dynamic/E/t8` | 22.1 (±7.2) | 0.3404 | 113.763 | +925.7% | 0/3 |  |
| `cpu/int8-dynamic/LP-E/t2` | 13.3 (±0.2) | 0.6408 | 118.205 | +965.7% | 0/3 |  |
| `cpu/int8-dynamic/P-1t/t6` | 36.3 (±2.7) | 0.2659 | 113.763 | +925.7% | 0/3 | yes |
| `cpu/int8-dynamic/P-smt/t12` | 27.9 (±2.5) | 0.2906 | 113.763 | +925.7% | 0/3 |  |

## Non-dominated configurations

- `cpu/bf16/P-1t/t6`
- `cpu/int8-dynamic/P-1t/t6`

Everything absent from that list is dominated: some other configuration is at least as cheap *and* at least as accurate, so it should never be chosen.

