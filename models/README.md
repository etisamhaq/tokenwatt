# GGUF weights

Not committed: 1.1 GB combined, over GitHub's 100 MB per-file limit, and
reproducible by download. The `llamacpp` engine in `backends.py` expects them
here.

```bash
cd models
BASE=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main
curl -LO $BASE/qwen2.5-0.5b-instruct-q4_k_m.gguf   # 469 MB, per-block 4-bit
curl -LO $BASE/qwen2.5-0.5b-instruct-q8_0.gguf     # 644 MB, per-block 8-bit
```

`Q8_0` is the one that matters for the granularity argument: it is 8-bit, the
same width as PyTorch's per-tensor dynamic int8, so comparing the two isolates
granularity with bit width held constant.

Reproduce with:

```bash
make sweep-kquant      # energy, both schemes, P-1t vs LP-E
make kquant-perplexity # quality, measured separately (see note below)
```

Perplexity is measured in a separate pass because `logits_all=True` is required
to read per-position logits and costs roughly half the decode throughput
(68.6 -> 32.7 tok/s measured), which would corrupt any energy figure taken
under it.
