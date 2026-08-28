"""Anonymized software and compute metadata."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _memory_gib() -> float | None:
    path = Path("/proc/meminfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return kib / 1024.0**2
    return None


def collect_system_info() -> dict[str, object]:
    packages = ["numpy", "scipy", "pandas", "matplotlib", "PyYAML", "threadpoolctl"]
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus_available": os.cpu_count(),
        "memory_gib_available": _memory_gib(),
        "accelerator": "none (CPU-only)",
        "package_versions": {name: version(name) for name in packages},
    }
