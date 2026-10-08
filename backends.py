#!/usr/bin/env python3
"""Inference backends under measurement.

Only CPU (via PyTorch) is implemented here. The iGPU and NPU paths need separate
toolchains (oneAPI/SYCL and OpenVINO), so they are declared as stubs with a
clear error rather than pretended into existence -- the Backend interface is
designed so adding them does not change the runner.

Why a hand-written decode loop instead of model.generate()
----------------------------------------------------------
generate() gives one wall-clock number for a whole request, which conflates two
phases with completely different cost structure: prefill is compute-bound over
the whole prompt, decode is memory-bound and produces one token at a time. For
an energy study those must be separated, because the joules-per-token of prefill
and decode differ by a large factor.

The manual loop also exposes the KV cache, which lets us measure its growth in
bytes as context extends -- one of the questions the project exists to answer.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

PRECISIONS = ("fp32", "bf16", "int8-dynamic")

# GGUF K-quants, served through llama.cpp. These exist to test the claim that
# quantization GRANULARITY rather than bit width is what costs quality:
# PyTorch dynamic int8 assigns one scale per tensor, K-quants assign scales per
# block. Mapped to the GGUF files that provide them.
GGUF_PRECISIONS = {
    "gguf-q4_k_m": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "gguf-q8_0":   "qwen2.5-0.5b-instruct-q8_0.gguf",
}


HERE = str(Path(__file__).resolve().parent)


@dataclass
class GenResult:
    """One generation, with phases separated."""
    text: str
    n_prompt_tokens: int
    n_new_tokens: int
    prefill_s: float
    decode_s: float
    decode_step_s: list[float] = field(default_factory=list)
    kv_bytes_final: int = 0
    kv_bytes_by_step: list[int] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s

    @property
    def decode_tok_per_s(self) -> float:
        return self.n_new_tokens / self.decode_s if self.decode_s > 0 else 0.0

    @property
    def prefill_tok_per_s(self) -> float:
        return self.n_prompt_tokens / self.prefill_s if self.prefill_s > 0 else 0.0


class Backend(ABC):
    name: str = "abstract"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> GenResult: ...

    @abstractmethod
    def logprobs(self, text: str, stride: int) -> tuple[float, int]:
        """Sum of token log-likelihoods and token count, for perplexity."""

    def unload(self) -> None:
        pass

    @property
    def param_bytes(self) -> int:
        return 0


def _kv_cache_bytes(past) -> int:
    """Bytes held in the KV cache.

    transformers has moved between a tuple-of-tuples-of-tensors layout and Cache
    objects, and both appear across 4.x versions, so handle either."""
    if past is None:
        return 0
    total = 0
    # Cache objects (DynamicCache and friends)
    for attr in ("key_cache", "value_cache"):
        seq = getattr(past, attr, None)
        if seq is not None:
            for t in seq:
                if t is not None and hasattr(t, "numel"):
                    total += t.numel() * t.element_size()
    if total:
        return total
    # Legacy tuple layout
    try:
        for layer in past:
            for t in layer:
                if t is not None and hasattr(t, "numel"):
                    total += t.numel() * t.element_size()
    except TypeError:
        return 0
    return total


class TransformersCPU(Backend):
    """PyTorch CPU inference at a chosen numeric precision.

    fp32          reference, no quantization
    bf16          half the weight traffic; Meteor Lake has AVX support for it
    int8-dynamic  torch dynamic quantization of Linear layers, weights to int8
                  with activations quantized per batch at runtime. This is the
                  quantization CPU inference actually has available without a
                  separate toolchain, and it is the interesting energy case
                  because it cuts memory traffic rather than arithmetic.
    """

    def __init__(self, model_id: str, precision: str = "fp32",
                 threads: int | None = None, seed: int = 0):
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        self.model_id = model_id
        self.precision = precision
        self.threads = threads
        self.seed = seed
        self.name = f"cpu/{precision}"
        self.model = None
        self.tok = None
        self._torch = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        torch.manual_seed(self.seed)
        if self.threads:
            torch.set_num_threads(self.threads)

        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        # low_cpu_mem_usage / device_map deliberately not used: they require
        # accelerate, and for the small models this study targets the loading
        # path is not what we are measuring.
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype)
        model.eval()

        if self.precision == "int8-dynamic":
            # Linear layers only: that is where the weights are, and dynamic
            # quantization avoids needing a calibration dataset.
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)

        self.model = model

    @property
    def param_bytes(self) -> int:
        if self.model is None:
            return 0
        total = 0
        for p in self.model.parameters():
            total += p.numel() * p.element_size()
        # Quantized Linear weights live outside .parameters(), so account for
        # them explicitly or int8 looks implausibly small.
        for m in self.model.modules():
            packed = getattr(m, "_packed_params", None)
            if packed is not None:
                try:
                    w = packed._weight_bias()[0]
                    total += w.numel() * w.element_size()
                except Exception:
                    pass
        return total

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult:
        torch = self._torch
        ids = self.tok(prompt, return_tensors="pt").input_ids
        n_prompt = int(ids.shape[1])

        with torch.inference_mode():
            # --- prefill: one forward pass over the whole prompt ---
            t0 = time.perf_counter()
            out = self.model(ids, use_cache=True)
            prefill_s = time.perf_counter() - t0

            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            produced = [int(next_id)]
            step_times: list[float] = []
            kv_by_step: list[int] = [_kv_cache_bytes(past)]

            # --- decode: greedy, one token per iteration ---
            t_decode0 = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                s = time.perf_counter()
                out = self.model(next_id, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                step_times.append(time.perf_counter() - s)
                produced.append(int(next_id))
                kv_by_step.append(_kv_cache_bytes(past))
                if self.tok.eos_token_id is not None and \
                        int(next_id) == self.tok.eos_token_id:
                    break
            decode_s = time.perf_counter() - t_decode0

        return GenResult(
            text=self.tok.decode(produced, skip_special_tokens=True),
            n_prompt_tokens=n_prompt,
            n_new_tokens=len(produced),
            prefill_s=prefill_s,
            decode_s=decode_s,
            decode_step_s=step_times,
            kv_bytes_final=kv_by_step[-1] if kv_by_step else 0,
            kv_bytes_by_step=kv_by_step,
        )

    def logprobs(self, text: str, stride: int = 512) -> tuple[float, int]:
        """Total log-likelihood over `text` using non-overlapping windows.

        Windowing is necessary because the context length is finite; using
        disjoint windows (rather than a sliding window with overlap) keeps the
        token count unambiguous, at the cost of a slightly pessimistic estimate
        at window boundaries. Same treatment for every configuration, so the
        comparison between them stays fair."""
        torch = self._torch
        ids = self.tok(text, return_tensors="pt").input_ids
        n = int(ids.shape[1])
        total_ll = 0.0
        counted = 0
        with torch.inference_mode():
            for start in range(0, n, stride):
                window = ids[:, start:start + stride]
                if window.shape[1] < 2:
                    continue
                logits = self.model(window).logits.float()
                # predict token t+1 from position t
                lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
                tgt = window[:, 1:]
                gathered = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                total_ll += float(gathered.sum())
                counted += int(tgt.numel())
        return total_ll, counted

    def unload(self) -> None:
        self.model = None
        self.tok = None


class _NotImplementedBackend(Backend):
    def __init__(self, name: str, how: str):
        self.name = name
        self.how = how

    def load(self) -> None:
        raise NotImplementedError(
            f"{self.name} backend is not implemented yet. {self.how}")

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult:
        raise NotImplementedError(self.name)

    def logprobs(self, text: str, stride: int) -> tuple[float, int]:
        raise NotImplementedError(self.name)


def igpu_backend() -> Backend:
    return _NotImplementedBackend(
        "igpu", "Needs llama.cpp built with the SYCL backend against oneAPI, "
                "or OpenVINO with device='GPU'.")


def npu_backend() -> Backend:
    return _NotImplementedBackend(
        "npu", "Needs OpenVINO GenAI with device='NPU'. The intel_vpu driver "
               "and /dev/accel/accel0 are present on this machine.")


class LlamaCppBackend(Backend):
    """GGUF inference through llama.cpp, for the K-quant comparison.

    A separate engine rather than another precision of TransformersCPU, because
    the whole point is that the quantization is done by a different toolchain
    with a different scale granularity. Timing is kept deliberately comparable:
    prefill is one eval() over the prompt, decode is a manual loop of
    sample()+eval() so the two phases are measured the same way the PyTorch
    backend measures them.

    Greedy decode (temp=0) so output is deterministic and comparable with the
    PyTorch path."""

    def __init__(self, model_path: str, precision: str,
                 threads: int | None = None, seed: int = 0):
        if precision not in GGUF_PRECISIONS:
            raise ValueError(f"precision must be one of {tuple(GGUF_PRECISIONS)}")
        self.model_path = model_path
        self.precision = precision
        self.threads = threads
        self.seed = seed
        self.llm = None

    @property
    def name(self) -> str:
        return f"llamacpp/{self.precision}"

    def load(self, logits_all: bool = False) -> None:
        """Load the model. `logits_all` retains per-position logits, which
        perplexity needs and which costs roughly half the decode throughput
        (measured: 68.6 -> 32.7 tok/s). Timed generation therefore always runs
        with it off, and logprobs() reloads around its own measurement."""
        try:
            from llama_cpp import Llama
        except ImportError as e:  # optional dependency; see requirements.txt
            raise RuntimeError(
                "the llamacpp engine needs llama-cpp-python: "
                "pip install llama-cpp-python (see models/README.md)") from e
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GGUF weights not found at {self.model_path}. They are not "
                "committed (1.1 GB, over GitHub's per-file limit); "
                "models/README.md has the download commands.")
        self._logits_all = logits_all
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_threads=self.threads or os.cpu_count(),
            n_batch=512,
            seed=self.seed,
            logits_all=logits_all,
            verbose=False,
        )

    @property
    def param_bytes(self) -> int:
        """On-disk size of the quantized weights. Not directly comparable with
        the PyTorch backend's in-memory parameter count, and labelled as such."""
        try:
            return os.path.getsize(self.model_path)
        except OSError:
            return 0

    def generate(self, prompt: str, max_new_tokens: int) -> GenResult:
        llm = self.llm
        llm.reset()
        ids = llm.tokenize(prompt.encode("utf-8"))
        n_prompt = len(ids)

        t0 = time.perf_counter()
        llm.eval(ids)                      # prefill: whole prompt, one pass
        prefill_s = time.perf_counter() - t0

        out, step_s = [], []
        eos = llm.token_eos()
        t_dec = time.perf_counter()
        for _ in range(max_new_tokens):
            t = time.perf_counter()
            tok = llm.sample(temp=0.0)     # greedy, for determinism
            if tok == eos:
                break
            llm.eval([tok])
            step_s.append(time.perf_counter() - t)
            out.append(tok)
        decode_s = time.perf_counter() - t_dec

        text = llm.detokenize(out).decode("utf-8", errors="replace")
        # llama.cpp manages its KV cache internally and does not expose a
        # per-step byte count. Recorded as empty rather than estimated, so the
        # analysis cannot mistake a model for a measurement.
        return GenResult(text=text, n_prompt_tokens=n_prompt,
                         n_new_tokens=len(out), prefill_s=prefill_s,
                         decode_s=decode_s, decode_step_s=step_s,
                         kv_bytes_final=0, kv_bytes_by_step=[])

    def logprobs(self, text: str, stride: int = 512) -> tuple[float, int]:
        """Total log-likelihood over `text`, windowed like the PyTorch path.

        Computed from the raw logits rather than through create_completion's
        echo mode, which returns values near-uniform over the vocabulary in
        this llama-cpp-python release (measured perplexity 5e5 against 11.4
        from the logits, on identical text) -- it is not usable for this.

        Reloads the model with logits_all on and restores it afterwards, so a
        timed generation never runs under the configuration that halves decode
        throughput."""
        import numpy as np

        was_logits_all = getattr(self, "_logits_all", False)
        if not was_logits_all:
            self.unload()
            self.load(logits_all=True)
        try:
            llm = self.llm
            ids = llm.tokenize(text.encode("utf-8"))
            window = min(stride, llm.n_ctx() - 8)
            total, n = 0.0, 0
            for start in range(0, len(ids), window):
                chunk = ids[start:start + window]
                if len(chunk) < 2:
                    continue
                llm.reset()
                llm.eval(chunk)
                sc = np.asarray(llm.scores[:len(chunk)], dtype=np.float64)
                # log-softmax per position, scored against the token that
                # actually followed. Max-subtracted for numerical stability.
                rows = sc[:-1]
                rows = rows - rows.max(axis=1, keepdims=True)
                lse = np.log(np.exp(rows).sum(axis=1))
                picked = rows[np.arange(len(rows)), np.asarray(chunk[1:])]
                total += float((picked - lse).sum())
                n += len(rows)
            return total, n
        finally:
            if not was_logits_all:
                self.unload()
                self.load(logits_all=False)

    def unload(self) -> None:
        self.llm = None


def make_backend(engine: str, model_id: str, precision: str,
                 threads: int | None = None, seed: int = 0) -> Backend:
    if engine == "cpu":
        return TransformersCPU(model_id, precision, threads, seed)
    if engine == "llamacpp":
        fname = GGUF_PRECISIONS.get(precision)
        if fname is None:
            raise ValueError(f"precision must be one of {tuple(GGUF_PRECISIONS)}")
        path = os.path.join(HERE, "models", fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GGUF not found at {path}. Download it from the "
                f"Qwen2.5-0.5B-Instruct-GGUF repository before running the "
                f"llamacpp engine.")
        return LlamaCppBackend(path, precision, threads, seed)
    if engine == "igpu":
        return igpu_backend()
    if engine == "npu":
        return npu_backend()
    raise ValueError(f"unknown engine {engine!r}")
