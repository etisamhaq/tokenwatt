.PHONY: help test check sweep sweep-big sweep-longctx sweep-kquant kquant-perplexity reproduce analyze clean rapl idle
.DEFAULT_GOAL := help

PY      ?= python3
MODEL   ?= HuggingFaceTB/SmolLM2-135M
BIG     ?= Qwen/Qwen2.5-0.5B-Instruct
REPEATS ?= 5
PROMPTS ?= 4
TOKENS  ?= 256
PLACES  ?= P-1t,P-smt,E,LP-E
PRECS   ?= fp32,bf16,int8-dynamic

# Output directories are versioned. The unsuffixed runs/qwen2.5-0.5b and
# runs/smollm2-135m hold SUPERSEDED datasets kept as documented evidence, and
# nothing here writes to them -- an earlier version of this Makefile did, which
# would have destroyed the record while failing to reproduce the paper.

help:
	@echo "make test        run the unit suite (fast, no model needed)"
	@echo "make check       show CPU islands and whether RAPL is readable"
	@echo "make rapl        print the command to grant RAPL access (needs your sudo)"
	@echo "make idle        measure idle power baseline"
	@echo "make sweep       full sweep, small model      (MODEL=$(MODEL))"
	@echo "make sweep-big   full sweep, larger model     (BIG=$(BIG))"
	@echo "make sweep-longctx  1024-token run, bf16, best vs worst island"
	@echo "make sweep-kquant     GGUF K-quants via llama.cpp (needs models/, see models/README.md)"
	@echo "make kquant-perplexity quality for the GGUF schemes, measured separately"
	@echo "make reproduce   all published runs (~5 h)"
	@echo "make analyze     regenerate RESULTS.md and figures/ from results.jsonl"
	@echo "make clean       remove generated results, figures and caches"

test:
	$(PY) tests/test_harness.py

check:
	@$(PY) topology.py
	@echo
	@$(PY) energy.py --selftest || true

rapl:
	@echo "RAPL counters are root-only (CVE-2020-8694 mitigation). Run:"
	@echo
	@echo "  sudo chmod -R a+r /sys/devices/virtual/powercap/"
	@echo
	@echo "NOTE: /sys/class/powercap are symlinks; chmod -R will not follow"
	@echo "them, so chmod'ing that path silently does nothing."
	@echo
	@echo "or install the permanent udev rule described in README.md."

idle:
	$(PY) energy.py --idle 30

# Each target reproduces one published dataset with the exact flags used.
# Leave the machine idle while these run: the idle baseline is ~10 W against a
# net signal of ~3 W, so background load lands directly in the result.

sweep:
	$(PY) runner.py --model $(MODEL) --precisions $(PRECS) \
	  --placements $(PLACES) --prompts $(PROMPTS) --repeats $(REPEATS) \
	  --max-new-tokens $(TOKENS) --idle-seconds 20 \
	  --out runs/smollm2-135m-v3/results.jsonl
	$(PY) analyze.py --results runs/smollm2-135m-v3/results.jsonl \
	  --report runs/smollm2-135m-v3/RESULTS.md --outdir runs/smollm2-135m-v3/figures

sweep-big:
	$(PY) runner.py --model $(BIG) --precisions $(PRECS) \
	  --placements $(PLACES) --prompts $(PROMPTS) --repeats $(REPEATS) \
	  --max-new-tokens $(TOKENS) --idle-seconds 20 \
	  --out runs/qwen2.5-0.5b-v3/results.jsonl
	$(PY) analyze.py --results runs/qwen2.5-0.5b-v3/results.jsonl \
	  --report runs/qwen2.5-0.5b-v3/RESULTS.md --outdir runs/qwen2.5-0.5b-v3/figures

# Published with --prompts 2, which leaves too few paired ratios for a
# bootstrap interval; see runs/longctx-1024/NOTE.md. Use --prompts 4 to fix it.
sweep-longctx:
	$(PY) runner.py --model $(BIG) --precisions bf16 \
	  --placements P-1t,LP-E --prompts 2 --repeats 3 \
	  --max-new-tokens 1024 --idle-seconds 20 \
	  --out runs/longctx-1024/results.jsonl
	$(PY) analyze.py --results runs/longctx-1024/results.jsonl \
	  --report runs/longctx-1024/RESULTS.md --outdir runs/longctx-1024/figures

# GGUF K-quants through llama.cpp. Q8_0 is the one that matters for the
# granularity argument: 8-bit like PyTorch's per-tensor dynamic int8, so
# comparing them holds bit width constant. Needs `pip install llama-cpp-python`
# and the weights in models/ -- see models/README.md.
sweep-kquant:
	$(PY) runner.py --engines llamacpp --precisions gguf-q4_k_m \
	  --placements P-1t,LP-E --prompts $(PROMPTS) --repeats 3 \
	  --max-new-tokens $(TOKENS) --idle-seconds 20 --no-quality \
	  --out runs/kquant/results.jsonl
	$(PY) runner.py --engines llamacpp --precisions gguf-q8_0 \
	  --placements P-1t,LP-E --prompts $(PROMPTS) --repeats 3 \
	  --max-new-tokens $(TOKENS) --idle-seconds 20 --no-quality \
	  --out runs/kquant-q8/results.jsonl

# Separate from the energy sweep on purpose: perplexity needs logits_all=True,
# which costs about half the decode throughput and would corrupt any energy
# figure measured under it.
kquant-perplexity:
	$(PY) kquant_perplexity.py

reproduce: sweep-big sweep sweep-longctx sweep-kquant kquant-perplexity

analyze:
	$(PY) analyze.py --results runs/qwen2.5-0.5b-v3/results.jsonl \
	  --report runs/qwen2.5-0.5b-v3/RESULTS.md --outdir runs/qwen2.5-0.5b-v3/figures

clean:
	rm -rf __pycache__ tests/__pycache__ *.log
