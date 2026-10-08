# Results

Model: `Qwen/Qwen2.5-0.5B-Instruct`  ·  3 repeats per configuration  ·  1024 new tokens per generation

Machine: x86_64 — P 6c/12t @ 4.8GHz, E 8c/8t @ 3.8GHz, LP-E 2c/2t @ 2.5GHz

Energy from RAPL `package-0`, net of a single run-level idle baseline of 5.29 W, measured before any load.

Idle baseline for `package-0` across 2 configurations: 5.29-5.29 W, median 5.29 W (spread 0% of median).

Net energy went slightly negative in a domain other than `package-0` for 12 of 12 measurements. This is expected rather than alarming: the `uncore` baseline is around 0.2 W and dominated by noise, so subtracting it can cross zero. No figure in this report is computed from those domains, and `package-0` itself did not go negative.

## Per-configuration summary

| config | n | tok/s [95% CI] | J/token [95% CI] | perplexity | Δquality | needle | on frontier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu/bf16/LP-E/t2` | 6 | 6.2 [6.2, 6.3] | 1.4064 [1.3961, 1.4200] | 12.427 | +0.0% | 3/3 |  |
| `cpu/bf16/P-1t/t6` | 6 | 14.0 [13.9, 14.0] | 0.6244 [0.6224, 0.6282] | 12.427 | +0.0% | 3/3 | yes |

## Paired comparisons

Ratios computed within each of the 2 prompts and then aggregated (geometric mean, bootstrap 95% CI over prompts). Pairing removes prompt-to-prompt variation, which is otherwise large enough to swamp these effects. A ratio below 1.000 favours the first config; an interval spanning 1.000 means the data does not resolve the direction.

### Core placement (precision fixed at `bf16`)

| comparison | J/token ratio | tok/s ratio |
| --- | --- | --- |
| `LP-E` vs `P-1t` | 2.251x | 0.444x |

## Output determinism across placements

Greedy decode produced byte-identical output on every core island, for every precision and prompt. Placement affected cost only, not the tokens generated.

## Non-dominated configurations

- `cpu/bf16/P-1t/t6`

Everything absent from that list is dominated: some other configuration is at least as cheap *and* at least as accurate, so it should never be chosen.

