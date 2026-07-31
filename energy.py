#!/usr/bin/env python3
"""RAPL energy sampling for inference benchmarking.

Reads Intel RAPL counters from /sys/class/powercap to get ground-truth energy in
joules for a measured workload, rather than estimating from power draw.

Three things make this less trivial than it looks:

  * The counters are monotonically increasing microjoule totals that WRAP at a
    domain-specific maximum, so a naive end-minus-start goes negative.
  * They are root-readable only by default (mitigation for CVE-2020-8694, a
    side-channel that recovered AES keys from power traces). See README.
  * On integrated parts the package domain includes the iGPU, so CPU and iGPU
    energy cannot be cleanly separated within a single run.

Usage:
    from energy import EnergyMeter
    with EnergyMeter() as m:
        run_inference(...)
    print(m.joules)          # {'package-0': 412.7, 'core': 301.2, ...}
    print(m.watts)           # mean power per domain over the window

CLI:
    python3 energy.py --selftest      # check readability, show a live sample
    python3 energy.py --idle 30       # measure idle baseline for N seconds
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

POWERCAP = Path("/sys/class/powercap")


class RaplDomain:
    """One RAPL measurement domain (a package, or a subdomain like core/uncore)."""

    def __init__(self, path: Path):
        self.path = path
        self.name = (path / "name").read_text().strip()
        # Parent package prefixes subdomain names so 'core' from two sockets
        # doesn't collide.
        parent = path.parent
        if parent.name.startswith("intel-rapl:"):
            self.name = f"{(parent / 'name').read_text().strip()}/{self.name}"
        self._energy_file = path / "energy_uj"
        self.max_uj = int((path / "max_energy_range_uj").read_text().strip())

    def read_uj(self) -> int:
        return int(self._energy_file.read_text().strip())

    def delta_uj(self, start: int, end: int) -> int:
        """End minus start, correcting for a single counter wrap.

        A wrap is indistinguishable from a very long run that wrapped twice, so
        callers should keep windows well under the wrap period (typically tens of
        seconds at full load for the package domain)."""
        return end - start if end >= start else (self.max_uj - start) + end


def discover_domains() -> list[RaplDomain]:
    """All readable RAPL domains. Skips intel-rapl-mmio, which duplicates the
    same physical counters through a different interface."""
    domains = []
    for path in sorted(POWERCAP.glob("intel-rapl:*")):
        if "mmio" in path.name:
            continue
        for candidate in [path, *sorted(path.glob("intel-rapl:*:*"))]:
            if not (candidate / "energy_uj").exists():
                continue
            try:
                domains.append(RaplDomain(candidate))
            except PermissionError:
                pass  # reported by check_readable(), not swallowed silently
    return domains


def check_readable() -> tuple[bool, str]:
    if not POWERCAP.exists():
        return False, "no /sys/class/powercap - RAPL unsupported on this CPU"
    probes = list(POWERCAP.glob("intel-rapl:*/energy_uj"))
    if not probes:
        return False, "powercap present but no intel-rapl domains found"
    try:
        probes[0].read_text()
    except PermissionError:
        # /sys/class/powercap/* are symlinks into /sys/devices, and chmod -R
        # does NOT follow symlinks -- chmod'ing the /sys/class path silently
        # changes nothing. Report the resolved path so the fix actually works.
        real = probes[0].resolve().parent.parent
        return False, (
            "RAPL counters are root-only (CVE-2020-8694 mitigation).\n"
            f"  sudo chmod -R a+r {real}\n"
            "Note: chmod'ing /sys/class/powercap has no effect -- those are\n"
            "symlinks and chmod -R does not follow them. See README.md."
        )
    return True, "ok"


class EnergyMeter:
    """Context manager measuring energy over the enclosed block.

    Also records wall time so mean power can be derived. Energy alone is
    ambiguous: a slow efficient run and a fast wasteful one can consume the same
    joules, and for inference the joules-per-token is the number that matters.
    """

    def __init__(self, domains: list[RaplDomain] | None = None):
        ok, msg = check_readable()
        if not ok:
            raise PermissionError(msg)
        self.domains = domains if domains is not None else discover_domains()
        if not self.domains:
            raise RuntimeError("no readable RAPL domains")
        self.joules: dict[str, float] = {}
        self.seconds: float = 0.0

    def __enter__(self) -> "EnergyMeter":
        self._start = {d.name: d.read_uj() for d in self.domains}
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> bool:
        self.seconds = time.perf_counter() - self._t0
        for d in self.domains:
            delta = d.delta_uj(self._start[d.name], d.read_uj())
            self.joules[d.name] = delta / 1_000_000
        return False  # never suppress exceptions

    @property
    def watts(self) -> dict[str, float]:
        if self.seconds <= 0:
            return {}
        return {k: v / self.seconds for k, v in self.joules.items()}

    def per_token(self, tokens: int) -> dict[str, float]:
        """Joules per token — the headline metric for this study."""
        if tokens <= 0:
            return {}
        return {k: v / tokens for k, v in self.joules.items()}


def cmd_selftest() -> int:
    ok, msg = check_readable()
    print(f"readable: {ok}")
    if not ok:
        print(msg)
        return 1
    domains = discover_domains()
    print(f"domains : {len(domains)}")
    for d in domains:
        print(f"  {d.name:20s} wrap at {d.max_uj / 1_000_000:.1f} J")

    print("\nsampling 3s of whatever this machine is currently doing...")
    with EnergyMeter(domains) as m:
        time.sleep(3)
    for name, j in sorted(m.joules.items()):
        print(f"  {name:20s} {j:8.2f} J   {m.watts[name]:6.2f} W")
    return 0


def cmd_idle(seconds: int) -> int:
    """Idle baseline. Subtracting this is what separates the energy cost of
    inference from the energy cost of the machine merely being switched on."""
    print(f"measuring idle baseline for {seconds}s - close other applications")
    with EnergyMeter() as m:
        time.sleep(seconds)
    print(f"\nover {m.seconds:.1f}s:")
    for name, j in sorted(m.joules.items()):
        print(f"  {name:20s} {j:8.2f} J   {m.watts[name]:6.2f} W idle")
    print("\nRecord these. Report inference energy both gross and net of idle.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="verify RAPL access and show a 3s sample")
    ap.add_argument("--idle", type=int, metavar="SECONDS",
                    help="measure idle baseline power")
    args = ap.parse_args()

    if args.selftest:
        return cmd_selftest()
    if args.idle:
        return cmd_idle(args.idle)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
