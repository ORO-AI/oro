"""Minimum-spec host check for the validator.

Called once at startup so an operator running on a too-small host gets a
loud WARNING in the log before the eval loop begins. The reference shape
for the production fleet is a `c7g.2xlarge` (8 vCPU / 16 GiB RAM); the
warn thresholds sit a step below so a mild under-spec doesn't spam, but
anything meaningfully small (like the 2 vCPU / 3.8 GiB box that surfaced
this issue on 2026-07-28) trips.

Under-spec hosts silently cascade into low miner scores: search-server
thrashes when the index cannot stay resident, agents blow past the 300s
per-problem cap while inference clients sit behind CPU contention, and
every affected miner gets a near-zero score they cannot debug. Catching
it at boot is much cheaper than untangling it after the fact.

The check reads what the *process* sees (cgroup-aware CPU affinity,
`/proc/meminfo`), so a container with tight cgroup limits also trips
the warning — which is what we want.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from bittensor.utils.btlogging import logging


# Reference production shape: c7g.2xlarge = 8 vCPU / 16 GiB. Warn a step
# below so hosts within a reasonable margin don't get pinged; anything
# clearly small (say a c7g.large = 2 vCPU / 4 GiB) trips both thresholds.
MIN_CPUS = 6
MIN_RAM_GIB = 12.0


@dataclass(frozen=True)
class HostSpecs:
    cpus: int
    ram_gib: float

    def deficits(self) -> list[str]:
        out: list[str] = []
        if self.cpus and self.cpus < MIN_CPUS:
            out.append(f"cpus={self.cpus} (min {MIN_CPUS})")
        if self.ram_gib and self.ram_gib < MIN_RAM_GIB:
            out.append(f"ram_gib={self.ram_gib:.1f} (min {MIN_RAM_GIB:.0f})")
        return out


def read_host_specs() -> HostSpecs:
    """Return what the running process can see for CPU + RAM."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        cpus = os.cpu_count() or 0

    ram_gib = 0.0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # MemTotal is reported in KiB.
                    ram_gib = int(line.split()[1]) / (1024 * 1024)
                    break
    except OSError:
        pass

    return HostSpecs(cpus=cpus, ram_gib=ram_gib)


def check_host_min_specs() -> HostSpecs:
    """Log INFO if specs meet the minimum, WARNING (with reason) if not."""
    specs = read_host_specs()
    deficits = specs.deficits()
    if deficits:
        logging.warning(
            "HOST UNDER MIN SPEC: %s. Reference is 8 vCPU / 16 GiB "
            "(c7g.2xlarge). Under-spec hosts cause search-server thrashing "
            "and inference stalls, and produce systematically low agent "
            "scores. Validator will continue starting.",
            ", ".join(deficits),
        )
    else:
        logging.info(
            "Host specs OK: cpus=%d ram_gib=%.1f (min %d / %.0f)",
            specs.cpus,
            specs.ram_gib,
            MIN_CPUS,
            MIN_RAM_GIB,
        )
    return specs
