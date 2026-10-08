# Long-context check — valid, but underpowered

1024 generated tokens, bf16, `P-1t` against `LP-E`, 2 prompts x 3 repeats.

**Result.** The island penalty is stable against context length: 2.251x here
against 2.306x at 256 tokens, with package power still exactly 14.00 W on both
islands while the KV cache grows from 3,204 to 12,384 KiB.

**Limitation.** Run with 2 prompts rather than 4, which leaves only two paired
ratios — below the three `bootstrap_ci` requires before it will report an
interval. The guard behaved correctly; the experiment design did not. The point
estimate is sound, but a robustness check without an interval invites exactly
the question it exists to answer. Re-run at `--prompts 4` before relying on it.
