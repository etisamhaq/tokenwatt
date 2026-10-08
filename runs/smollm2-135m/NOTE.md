# Superseded by `runs/smollm2-135m-v3/`

Measured before RAPL read access was available, so every energy field is null
and cost is reported in seconds per token.

It also carries both defects described in `runs/qwen2.5-0.5b/NOTE.md`: 24-token
generations and prompts cycled by repeat index.

One claim from this file was repeated in the original FINDINGS.md and is false:
that E-cores beat P-cores on tokens per second for this model. Remeasured at 256
tokens with a settled warmup, `P-1t` reaches 30.0 tok/s against `E` at 24.2.
