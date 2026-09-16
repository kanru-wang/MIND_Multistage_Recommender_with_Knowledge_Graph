from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm


@lru_cache(maxsize=64)
def _sha256_file_state(path_text: str, size: int, mtime_ns: int) -> str:
    """Hash an immutable file state, caching by resolved path, size, and mtime."""
    del mtime_ns  # It intentionally participates in the cache key.
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle, tqdm(
        total=size,
        desc=f"Fingerprint {path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=size < 8 * 1024 * 1024,
        dynamic_ncols=True,
    ) as progress:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return _sha256_file_state(
        str(path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
    )


def artifact_signature(
    path: Path,
    *,
    content_hash: bool = False,
    include_mtime: bool = False,
) -> dict[str, Any]:
    exists = path.is_file()
    signature: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }
    if include_mtime:
        signature["mtime_ns"] = path.stat().st_mtime_ns if exists else None
    signature["sha256"] = sha256_file(path) if exists and content_hash else None
    return signature
