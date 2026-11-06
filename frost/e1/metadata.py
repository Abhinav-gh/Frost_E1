"""Reproducibility metadata helpers shared by experiment entry points."""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from typing import Any


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(root=None) -> dict[str, Any]:
    try:
        import torch
        cuda = torch.cuda.is_available()
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        gpu = torch.cuda.get_device_name() if cuda else None
    except ImportError:
        cuda = False
        torch_version = None
        cuda_version = None
        gpu = None
    provenance = {"commit": None, "dirty": None}
    if root is not None:
        try:
            provenance["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            provenance["dirty"] = bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda": cuda_version,
        "cuda_available": cuda,
        "gpu": gpu,
        "mace": package_version("mace-torch"),
        "ase": package_version("ase"),
        "provenance": provenance,
    }
