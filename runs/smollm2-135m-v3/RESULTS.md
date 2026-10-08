# Results

Model: `HuggingFaceTB/SmolLM2-135M`  ·  5 repeats per configuration  ·  256 new tokens per generation

Machine: x86_64 — P 6c/12t @ 4.8GHz, E 8c/8t @ 3.8GHz, LP-E 2c/2t @ 2.5GHz

Energy from RAPL `package-0`, net of a single run-level idle baseline of 9.66 W, measured before any load.

Idle baseline for `package-0` across 12 configurations: 9.66-9.66 W, median 9.66 W (spread 0% of median).

> **207 measurement(s) produced negative net energy.** The baseline exceeded the measurement, which means the baseline is wrong for those runs. Recorded unclamped rather than floored at zero so the error is visible.

## Per-configuration summary

| config | n | tok/s [95% CI] | J/token [95% CI] | perplexity | Δquality | needle | on frontier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu/bf16/E/t8` | 20 | 26.9 [26.0, 27.3] | 0.1658 [0.1610, 0.1695] | 13.353 | +0.2% | 0/3 |  |
| `cpu/bf16/LP-E/t2` | 20 | 8.3 [8.1, 10.6] | 0.5289 [0.4169, 0.5454] | 13.353 | +0.2% | 0/3 |  |
| `cpu/bf16/P-1t/t6` | 20 | 34.7 [34.3, 35.8] | 0.1280 [0.1226, 0.1297] | 13.353 | +0.2% | 0/3 | yes |
| `cpu/bf16/P-smt/t12` | 20 | 26.7 [24.7, 27.4] | 0.1671 [0.1598, 0.1800] | 13.353 | +0.2% | 0/3 |  |
| `cpu/fp32/E/t8` | 20 | 24.2 [24.1, 24.3] | 0.1807 [0.1802, 0.1817] | 13.321 | +0.0% | 0/3 |  |
| `cpu/fp32/LP-E/t2` | 20 | 12.6 [12.5, 12.6] | 0.3481 [0.3476, 0.3495] | 13.321 | +0.0% | 0/3 |  |
| `cpu/fp32/P-1t/t6` | 20 | 30.0 [28.2, 30.9] | 0.1459 [0.1406, 0.1550] | 13.321 | +0.0% | 0/3 | yes |
| `cpu/fp32/P-smt/t12` | 20 | 25.3 [24.5, 25.9] | 0.1734 [0.1700, 0.1776] | 13.321 | +0.0% | 0/3 |  |
| `cpu/int8-dynamic/E/t8` | 20 | 27.9 [27.8, 28.0] | 0.1742 [0.1731, 0.1751] | 40.792 | +206.2% | 1/3 |  |
| `cpu/int8-dynamic/LP-E/t2` | 20 | 14.9 [14.9, 15.0] | 0.3081 [0.3036, 0.3097] | 41.338 | +210.3% | 0/3 |  |
| `cpu/int8-dynamic/P-1t/t6` | 20 | 36.3 [35.7, 36.4] | 0.1301 [0.1295, 0.1327] | 40.792 | +206.2% | 1/3 |  |
| `cpu/int8-dynamic/P-smt/t12` | 20 | 28.5 [28.0, 29.1] | 0.1594 [0.1550, 0.1609] | 40.125 | +201.2% | 0/3 |  |

## Paired comparisons

Ratios computed within each of the 4 prompts and then aggregated (geometric mean, bootstrap 95% CI over prompts). Pairing removes prompt-to-prompt variation, which is otherwise large enough to swamp these effects. A ratio below 1.000 favours the first config; an interval spanning 1.000 means the data does not resolve the direction.

### Core placement (precision fixed at `bf16`)

| comparison | J/token ratio | tok/s ratio |
| --- | --- | --- |
| `E` vs `P-1t` | 1.311x [1.271, 1.354] | 0.769x [0.750, 0.789] |
| `LP-E` vs `P-1t` | 3.712x [3.159, 4.342] | 0.271x [0.235, 0.313] |
| `P-smt` vs `P-1t` | 1.359x [1.232, 1.477] | 0.750x [0.699, 0.821] |

### Precision (placement fixed at `P-1t`)

| comparison | J/token ratio | tok/s ratio |
| --- | --- | --- |
| `fp32` vs `bf16` | 1.178x [1.096, 1.266] | 0.848x [0.808, 0.890] |
| `int8-dynamic` vs `bf16` | 1.053x [1.007, 1.101] | 1.009x [0.963, 1.058] ⚠ sign varies by prompt |

## Output determinism across placements

Greedy decode produced byte-identical output on every core island, for every precision and prompt. Placement affected cost only, not the tokens generated.

## Non-dominated configurations

- `cpu/bf16/P-1t/t6`
- `cpu/fp32/P-1t/t6`

Everything absent from that list is dominated: some other configuration is at least as cheap *and* at least as accurate, so it should never be chosen.

