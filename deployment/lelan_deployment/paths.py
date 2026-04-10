"""Helpers for finding config and data files in source and install layouts."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


PACKAGE_NAME = "lelan_deployment"


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def source_deployment_dir() -> Optional[Path]:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path(__file__).resolve().parents[2] / "deployment",
    ]
    for candidate in candidates:
        if (candidate / "config").is_dir():
            return candidate
    return None


def package_share_dir() -> Optional[Path]:
    try:
        return Path(get_package_share_directory(PACKAGE_NAME))
    except PackageNotFoundError:
        return None


def deployment_roots() -> list[Path]:
    roots = []
    source_dir = source_deployment_dir()
    if source_dir is not None:
        roots.append(source_dir)
    share_dir = package_share_dir()
    if share_dir is not None:
        roots.append(share_dir)
    return _dedupe(roots)


def resolve_readonly_path(*relative_parts: str) -> Path:
    relative = Path(*relative_parts)
    for root in deployment_roots():
        candidate = root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve {'/'.join(relative.parts)} from deployment roots")


def resolve_from_deployment(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    for root in deployment_roots():
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    source_dir = source_deployment_dir()
    if source_dir is not None:
        return (source_dir / path).resolve()
    return path.resolve()


def resolve_writable_dir(*relative_parts: str) -> Path:
    relative = Path(*relative_parts)
    source_dir = source_deployment_dir()
    if source_dir is not None:
        path = source_dir / relative
        path.mkdir(parents=True, exist_ok=True)
        return path
    fallback_root = Path(os.environ.get("LELAN_RUNTIME_DIR", Path.cwd() / "deployment_runtime"))
    path = fallback_root / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_train_on_path() -> None:
    if importlib.util.find_spec("vint_train") is not None:
        return

    candidates = []
    env_root = os.environ.get("LELAN_TRAIN_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    source_dir = source_deployment_dir()
    if source_dir is not None:
        candidates.append(source_dir.parent / "train")

    candidates.append(Path.cwd() / "train")

    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            if importlib.util.find_spec("vint_train") is not None:
                return

    raise ImportError(
        "Could not import 'vint_train'. Install the training package with "
        "'pip install -e train/' or set LELAN_TRAIN_ROOT."
    )
