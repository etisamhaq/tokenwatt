# Results

Model: `Qwen/Qwen2.5-0.5B-Instruct`  ·  5 repeats per configuration  ·  256 new tokens per generation

Machine: x86_64 — P 6c/12t @ 4.8GHz, E 8c/8t @ 3.8GHz, LP-E 2c/2t @ 2.5GHz

Energy from RAPL `package-0`, net of a single run-level idle baseline of 11.23 W, measured before any load.

Idle baseline for `package-0` across 12 configurations: 11.23-11.23 W, median 11.23 W (spread 0% of median).

> **237 measurement(s) produced negative net energy.** The baseline exceeded the measurement, which means the baseline is wrong for those runs. Recorded unclamped rather than floored at zero so the error is visible.

## Per-configuration summary

| config | n | tok/s [95% CI] | J/token [95% CI] | perplexity | Δquality | needle | on frontier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu/bf16/E/t8` | 20 | 14.4 [13.9, 14.5] | 0.1987 [0.1966, 0.2045] | 12.427 | +0.0% | 3/3 |  |
| `cpu/bf16/LP-E/t2` | 20 | 7.0 [7.0, 7.0] | 0.4053 [0.4041, 0.4069] | 12.427 | +0.0% | 3/3 |  |
| `cpu/bf16/P-1t/t6` | 20 | 16.1 [16.1, 16.2] | 0.1758 [0.1753, 0.1763] | 12.427 | +0.0% | 3/3 | yes |
| `cpu/bf16/P-smt/t12` | 20 | 15.1 [14.7, 15.2] | 0.1878 [0.1861, 0.1903] | 12.427 | +0.0% | 3/3 |  |
| `cpu/fp32/E/t8` | 20 | 11.8 [11.7, 11.9] | 0.2357 [0.2338, 0.2381] | 12.449 | +0.2% | 3/3 |  |
| `cpu/fp32/LP-E/t2` | 20 | 7.0 [7.0, 7.0] | 0.4019 [0.4014, 0.4030] | 12.449 | +0.2% | 3/3 |  |
| `cpu/fp32/P-1t/t6` | 20 | 12.0 [11.7, 12.3] | 0.2313 [0.2290, 0.2365] | 12.449 | +0.2% | 3/3 |  |
| `cpu/fp32/P-smt/t12` | 20 | 11.1 [10.8, 11.3] | 0.2515 [0.2448, 0.2586] | 12.449 | +0.2% | 3/3 |  |
| `cpu/int8-dynamic/E/t8` | 20 | 22.5 [22.4, 22.6] | 0.1998 [0.1966, 0.2022] | 110.608 | +790.1% | 0/3 |  |
| `cpu/int8-dynamic/LP-E/t2` | 20 | 11.2 [11.2, 11.3] | 0.3209 [0.3200, 0.3224] | 106.391 | +756.1% | 0/3 |  |
| `cpu/int8-dynamic/P-1t/t6` | 20 | 27.7 [27.4, 28.0] | 0.1281 [0.1271, 0.1292] | 109.953 | +784.8% | 0/3 | yes |
| `cpu/int8-dynamic/P-smt/t12` | 20 | 22.6 [21.8, 23.7] | 0.1470 [0.1442, 0.1502] | 109.522 | +781.3% | 0/3 | yes |

## Paired comparisons

Ratios computed within each of the 4 prompts and then aggregated (geometric mean, bootstrap 95% CI over prompts). Pairing removes prompt-to-prompt variation, which is otherwise large enough to swamp these effects. A ratio below 1.000 favours the first config; an interval spanning 1.000 means the data does not resolve the direction.

### Core placement (precision fixed at `bf16`)

| comparison | J/token ratio | tok/s ratio |
| --- | --- | --- |
| `E` vs `P-1t` | 1.148x [1.117, 1.189] | 0.881x [0.860, 0.901] |
| `LP-E` vs `P-1t` | 2.306x [2.301, 2.314] | 0.435x [0.434, 0.437] |
| `P-smt` vs `P-1t` | 1.073x [1.066, 1.081] | 0.936x [0.926, 0.943] |

### Precision (placement fixed at `P-1t`)

| comparison | J/token ratio | tok/s ratio |
| --- | --- | --- |
| `fp32` vs `bf16` | 1.319x [1.309, 1.330] | 0.745x [0.743, 0.747] |
| `int8-dynamic` vs `bf16` | 0.730x [0.726, 0.733] | 1.705x [1.643, 1.746] |

## Output determinism across placements

Greedy decode produced byte-identical output on every core island, for every precision and prompt. Placement affected cost only, not the tokens generated.

## Non-dominated configurations

- `cpu/bf16/P-1t/t6`
- `cpu/int8-dynamic/P-1t/t6`
- `cpu/int8-dynamic/P-smt/t12`

Everything absent from that list is dominated: some other configuration is at least as cheap *and* at least as accurate, so it should never be chosen.

