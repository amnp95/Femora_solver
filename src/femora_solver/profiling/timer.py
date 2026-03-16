"""
SolverProfiler — phase-level timing instrumentation for HPC profiling.

Usage:
    profiler = SolverProfiler()

    with profiler.region("model_compilation"):
        plan = compiler.full_compile(model)

    print(profiler.summary())
    profiler.to_json("profile_results.json")

Phases tracked:
    model_compilation, field_layout_construction, partitioning,
    bulk_element_kernels, interface_coupling_kernels,
    constraint_enforcement, explicit_update, recorder_extraction,
    async_host_transfer, output_writing
"""

from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Canonical phase names (the 10 phases the user requested)
PHASE_NAMES = (
    "model_compilation",
    "field_layout_construction",
    "partitioning",
    "bulk_element_kernels",
    "interface_coupling_kernels",
    "constraint_enforcement",
    "explicit_update",
    "recorder_extraction",
    "async_host_transfer",
    "output_writing",
)


@dataclass
class _PhaseRecord:
    """Accumulated timings for a single phase."""
    durations_s: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.durations_s)

    @property
    def total_s(self) -> float:
        return sum(self.durations_s) if self.durations_s else 0.0

    @property
    def mean_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0

    @property
    def min_s(self) -> float:
        return min(self.durations_s) if self.durations_s else 0.0

    @property
    def max_s(self) -> float:
        return max(self.durations_s) if self.durations_s else 0.0


class SolverProfiler:
    """Phase-level profiler for Femora Solver HPC runs.

    Records wall-clock durations for named phases.  All timing uses
    ``time.perf_counter_ns()`` for high-resolution measurement.

    For GPU phases, callers are responsible for calling
    ``jax.block_until_ready()`` before and after the timed region so
    that the wall-clock captures actual device execution time.
    """

    def __init__(self) -> None:
        self._phases: Dict[str, _PhaseRecord] = {}
        self._wall_start_ns: Optional[int] = None
        self._wall_end_ns: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def region(self, name: str):
        """Context manager that times a named region.

        Example::

            with profiler.region("bulk_element_kernels"):
                f_int = block.compute_internal_forces(...)
        """
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            self.record(name, elapsed_ns / 1e9)

    def record(self, name: str, elapsed_s: float) -> None:
        """Manually record a duration (in seconds) for a phase."""
        if name not in self._phases:
            self._phases[name] = _PhaseRecord()
        self._phases[name].durations_s.append(elapsed_s)

    def mark_start(self) -> None:
        """Mark overall profiling start (for total wall-clock)."""
        self._wall_start_ns = time.perf_counter_ns()

    def mark_end(self) -> None:
        """Mark overall profiling end."""
        self._wall_end_ns = time.perf_counter_ns()

    @property
    def total_wall_s(self) -> float:
        """Total wall-clock time between mark_start and mark_end."""
        if self._wall_start_ns is None or self._wall_end_ns is None:
            return 0.0
        return (self._wall_end_ns - self._wall_start_ns) / 1e9

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._phases.clear()
        self._wall_start_ns = None
        self._wall_end_ns = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, title: str = "Femora Solver Profile") -> str:
        """Return a formatted summary table of all recorded phases."""
        total_accounted = sum(p.total_s for p in self._phases.values())
        wall = self.total_wall_s or total_accounted or 1.0  # avoid div-by-0

        lines = []
        lines.append("")
        lines.append(f"{'=' * 90}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 90}")
        lines.append(
            f"  {'Phase':<32s} {'Count':>7s} {'Total(s)':>10s} "
            f"{'Mean(s)':>10s} {'Min(s)':>10s} {'Max(s)':>10s} {'%Wall':>7s}"
        )
        lines.append(f"  {'-' * 86}")

        # Print canonical phases first (in order), then any extras
        printed = set()
        for name in PHASE_NAMES:
            if name in self._phases:
                p = self._phases[name]
                pct = (p.total_s / wall) * 100.0
                lines.append(
                    f"  {name:<32s} {p.count:>7d} {p.total_s:>10.4f} "
                    f"{p.mean_s:>10.6f} {p.min_s:>10.6f} {p.max_s:>10.6f} {pct:>6.1f}%"
                )
                printed.add(name)

        # Extra phases (user-defined or sub-phases)
        for name, p in sorted(self._phases.items()):
            if name in printed:
                continue
            pct = (p.total_s / wall) * 100.0
            lines.append(
                f"  {name:<32s} {p.count:>7d} {p.total_s:>10.4f} "
                f"{p.mean_s:>10.6f} {p.min_s:>10.6f} {p.max_s:>10.6f} {pct:>6.1f}%"
            )

        lines.append(f"  {'-' * 86}")
        lines.append(f"  {'Total accounted':<32s} {'':>7s} {total_accounted:>10.4f}")
        if self.total_wall_s > 0:
            lines.append(f"  {'Total wall-clock':<32s} {'':>7s} {self.total_wall_s:>10.4f}")
            unaccounted = self.total_wall_s - total_accounted
            lines.append(f"  {'Unaccounted (overhead)':<32s} {'':>7s} {unaccounted:>10.4f}")
        lines.append(f"{'=' * 90}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return all profiling data as a plain dictionary."""
        phases = {}
        for name, p in self._phases.items():
            phases[name] = {
                "count": p.count,
                "total_s": p.total_s,
                "mean_s": p.mean_s,
                "min_s": p.min_s,
                "max_s": p.max_s,
                "durations_s": p.durations_s,
            }

        meta = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "total_wall_s": self.total_wall_s,
        }
        # Try to add JAX info
        try:
            import jax
            meta["jax_version"] = jax.__version__
            meta["jax_backend"] = str(jax.default_backend())
            meta["jax_device_count"] = jax.device_count()
            meta["jax_devices"] = [str(d) for d in jax.devices()]
        except Exception:
            pass

        return {"meta": meta, "phases": phases}

    def to_json(self, path: str) -> None:
        """Export profiling results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self) -> str:
        n = len(self._phases)
        total = sum(p.total_s for p in self._phases.values())
        return f"SolverProfiler({n} phases, {total:.4f}s total)"
