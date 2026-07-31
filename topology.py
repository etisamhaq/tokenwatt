#!/usr/bin/env python3
"""CPU topology discovery and affinity control for heterogeneous SoCs.

Modern Intel client parts have several core types with different
power/performance characteristics. On the Core Ultra 7 155H (Meteor Lake) there
are three islands:

    P-cores    6 physical, SMT (2 threads each), 4.5-4.8 GHz
    E-cores    8 physical, no SMT, 3.8 GHz
    LP-E cores 2 physical, no SMT, 2.5 GHz, on the low-power SoC tile

Which island you run inference on is a free variable that nobody characterises,
and it plausibly matters more for joules-per-token than it does for latency.
This module classifies the islands by max frequency and provides affinity
pinning so a run can be confined to one of them.

Classification is by cpuinfo_max_freq rather than by a hardcoded CPU model,
so it degrades sensibly on other machines: a homogeneous CPU yields one tier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CPU_ROOT = Path("/sys/devices/system/cpu")


@dataclass
class Tier:
    """One core type, e.g. all P-cores."""
    name: str
    max_khz: int
    cpus: list[int] = field(default_factory=list)      # logical CPU ids
    physical_cores: set[int] = field(default_factory=set)

    @property
    def n_logical(self) -> int:
        return len(self.cpus)

    @property
    def n_physical(self) -> int:
        return len(self.physical_cores)

    @property
    def smt(self) -> bool:
        return self.n_logical > self.n_physical

    def one_thread_per_core(self) -> list[int]:
        """First logical CPU of each physical core.

        For memory-bound work like LLM decode, SMT siblings contend for the same
        cache and execution ports, so one-thread-per-core is often both faster
        and more energy-efficient than using every hyperthread. Making that
        testable is the point of exposing this separately."""
        seen: set[int] = set()
        out = []
        for cpu in sorted(self.cpus):
            core = _physical_core(cpu)
            if core not in seen:
                seen.add(core)
                out.append(cpu)
        return out


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _physical_core(cpu: int) -> int:
    """Physical core id for a logical CPU; falls back to the cpu id itself."""
    v = _read_int(CPU_ROOT / f"cpu{cpu}/topology/core_id")
    return v if v is not None else cpu


def online_cpus() -> list[int]:
    cpus = []
    for p in CPU_ROOT.glob("cpu[0-9]*"):
        try:
            cpu = int(p.name[3:])
        except ValueError:
            continue
        online = _read_int(p / "online")
        if online == 0:
            continue          # explicitly offlined
        cpus.append(cpu)
    return sorted(cpus)


# Names applied to frequency tiers from fastest to slowest. Beyond three tiers
# we fall back to generic labels rather than inventing vendor terminology.
_TIER_NAMES = ["P", "E", "LP-E"]


def discover_tiers() -> list[Tier]:
    """Group online CPUs into frequency tiers, fastest first.

    CPUs whose max frequency is unreadable are collected into a single 'unknown'
    tier so they are never silently dropped from the machine description."""
    by_freq: dict[int, list[int]] = {}
    unknown: list[int] = []
    for cpu in online_cpus():
        khz = _read_int(CPU_ROOT / f"cpu{cpu}/cpufreq/cpuinfo_max_freq")
        if khz is None:
            unknown.append(cpu)
        else:
            by_freq.setdefault(khz, []).append(cpu)

    tiers: list[Tier] = []
    for i, khz in enumerate(sorted(by_freq, reverse=True)):
        name = _TIER_NAMES[i] if i < len(_TIER_NAMES) else f"T{i}"
        cpus = sorted(by_freq[khz])
        tiers.append(Tier(name=name, max_khz=khz, cpus=cpus,
                          physical_cores={_physical_core(c) for c in cpus}))
    if unknown:
        tiers.append(Tier(name="unknown", max_khz=0, cpus=sorted(unknown),
                          physical_cores={_physical_core(c) for c in unknown}))
    return tiers


def merge_p_tiers(tiers: list[Tier]) -> list[Tier]:
    """Merge tiers that differ only by turbo bin into one logical island.

    Meteor Lake reports two P-core frequencies (4800 and 4500 MHz) because only
    some cores carry the higher turbo bin. They are the same core type and
    treating them as separate islands would be misleading, so tiers within 10%
    of each other are merged."""
    if not tiers:
        return tiers
    merged: list[Tier] = []
    for t in tiers:
        if merged and t.max_khz and merged[-1].max_khz:
            if abs(merged[-1].max_khz - t.max_khz) / merged[-1].max_khz <= 0.10:
                prev = merged[-1]
                prev.cpus = sorted(prev.cpus + t.cpus)
                prev.physical_cores |= t.physical_cores
                prev.max_khz = max(prev.max_khz, t.max_khz)
                continue
        merged.append(Tier(t.name, t.max_khz, list(t.cpus), set(t.physical_cores)))
    # Re-label after merging so names stay ordered fastest-first.
    for i, t in enumerate(merged):
        if t.name != "unknown":
            t.name = _TIER_NAMES[i] if i < len(_TIER_NAMES) else f"T{i}"
    return merged


def get_islands() -> dict[str, Tier]:
    """Canonical island map for this machine, keyed by tier name."""
    return {t.name: t for t in merge_p_tiers(discover_tiers())}


class Affinity:
    """Temporarily pin the current process to a set of logical CPUs.

    Restores the previous mask on exit, so a runner can sweep placements within
    one process without leaking state between configurations.
    """

    def __init__(self, cpus: list[int] | None):
        self.cpus = cpus
        self._saved: set[int] | None = None

    def __enter__(self) -> "Affinity":
        if self.cpus:
            self._saved = os.sched_getaffinity(0)
            os.sched_setaffinity(0, set(self.cpus))
        return self

    def __exit__(self, *_) -> bool:
        if self._saved is not None:
            os.sched_setaffinity(0, self._saved)
        return False


def describe() -> str:
    lines = []
    for t in get_islands().values():
        smt = f", SMT x{t.n_logical // max(t.n_physical, 1)}" if t.smt else ""
        lines.append(
            f"{t.name:6s} {t.n_physical:2d} cores / {t.n_logical:2d} threads  "
            f"{t.max_khz / 1_000_000:.1f} GHz{smt}\n"
            f"       cpus: {t.cpus}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    isl = get_islands()
    for name, t in isl.items():
        print(f"{name}: one-thread-per-core -> {t.one_thread_per_core()}")
