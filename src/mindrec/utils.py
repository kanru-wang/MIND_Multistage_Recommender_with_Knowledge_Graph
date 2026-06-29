from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str | None = "cuda") -> torch.device:
    requested_str = str(requested or "cuda").strip().lower()
    if requested_str == "auto":
        requested_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build and confirm the NVIDIA driver is active, "
            "or set the config device to 'cpu' or 'auto'."
        )
    return device


def device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested_device": str(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_total_memory_gb": round(props.total_memory / (1024**3), 3),
            }
        )
    return info


def log_device(device: torch.device, label: str) -> None:
    info = device_info(device)
    if device.type == "cuda":
        print(
            f"{label} device: {device} "
            f"({info['cuda_device_name']}, {info['cuda_total_memory_gb']} GB)"
        )
    else:
        print(f"{label} device: {device}")


def set_seed(seed: int, seed_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if is_dataclass(o):
            return asdict(o)
        return str(o)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def position_bias_weights(k: int, mode: str = "log") -> np.ndarray:
    pos = np.arange(1, k + 1, dtype=np.float32)
    if mode == "log":
        return 1.0 / np.log2(pos + 1.0)
    if mode == "linear":
        return (k - pos + 1.0) / k
    raise ValueError(f"Unknown position bias mode: {mode}")


def validation_split_name(_cfg: dict[str, Any]) -> str:
    return "val"


def test_split_name(_cfg: dict[str, Any]) -> str:
    return "test"


def teacher_artifact_run_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("artifacts", {}).get("teacher_run_name", cfg["run_name"]))


def teacher_artifact_root(cfg: dict[str, Any]) -> Path:
    return Path("runs") / teacher_artifact_run_name(cfg) / "teacher"


def ranker_time_feature_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    time_cfg = dict(cfg.get("ranker", {}).get("time_features", {}))
    interactions = dict(time_cfg.get("interactions", {}))
    enabled = bool(time_cfg.get("enabled", False))
    return {
        "use_time_features": enabled,
        "time_embedding_dim": int(time_cfg.get("embedding_dim", 4)),
        "time_gate_init": float(time_cfg.get("gate_init", 0.0)),
        "time_gate_max_abs": float(time_cfg.get("gate_max_abs", 1.0)),
        "use_category_hour_interaction": enabled
        and bool(
            interactions.get(
                "category_hour",
                time_cfg.get("hour_enabled", True),
            )
        ),
        "use_subcategory_hour_interaction": enabled
        and bool(
            interactions.get("subcategory_hour", False)
        ),
        "use_category_weekday_interaction": enabled
        and bool(
            interactions.get(
                "category_weekday",
                time_cfg.get("weekday_enabled", False),
            )
        ),
        "use_subcategory_weekday_interaction": enabled
        and bool(
            interactions.get("subcategory_weekday", False)
        ),
    }


def pair_artifact_path(proc_root: Path, split_name: str) -> Path:
    return proc_root / f"{split_name}_pairs.parquet"


def impression_artifact_path(proc_root: Path, split_name: str) -> Path:
    return proc_root / f"{split_name}_impressions.parquet"


def behavior_artifact_path(proc_root: Path, split_name: str) -> Path:
    return proc_root / f"{split_name}_behaviors.parquet"
