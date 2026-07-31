.PHONY: help test check sweep sweep-big analyze clean rapl idle
.DEFAULT_GOAL := help

PY      ?= python3
MODEL   ?= HuggingFaceTB/SmolLM2-135M
BIG     ?= Qwen/Qwen2.5-0.5B-Instruct
REPEATS ?= 5
PLACES  ?= P-1t,P-smt,E,LP-E
PRECS   ?= fp32,bf16,int8-dynamic

help:
	@echo "make test        run the unit suite (fast, no model needed)"
	@echo "make check       show CPU islands and whether RAPL is readable"
	@echo "make rapl        print the command to grant RAPL access (needs your sudo)"
	@echo "make idle        measure idle power baseline"
	@echo "make sweep       full sweep, small model      (MODEL=$(MODEL))"
	@echo "make sweep-big   full sweep, larger model     (BIG=$(BIG))"
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

sweep:
	$(PY) runner.py --model $(MODEL) --precisions $(PRECS) \
	  --placements $(PLACES) --repeats $(REPEATS) --out runs/smollm2-135m/results.jsonl
	$(PY) analyze.py --results runs/smollm2-135m/results.jsonl \
	  --report runs/smollm2-135m/RESULTS.md --outdir runs/smollm2-135m/figures

sweep-big:
	$(PY) runner.py --model $(BIG) --precisions $(PRECS) \
	  --placements $(PLACES) --repeats $(REPEATS) \
	  --corpus-chars 2000 --needle-repeats 12 \
	  --out runs/qwen2.5-0.5b/results.jsonl
	$(PY) analyze.py --results runs/qwen2.5-0.5b/results.jsonl \
	  --report runs/qwen2.5-0.5b/RESULTS.md --outdir runs/qwen2.5-0.5b/figures

analyze:
	$(PY) analyze.py --results runs/qwen2.5-0.5b/results.jsonl \
	  --report runs/qwen2.5-0.5b/RESULTS.md --outdir runs/qwen2.5-0.5b/figures

clean:
	rm -rf __pycache__ tests/__pycache__ *.log
