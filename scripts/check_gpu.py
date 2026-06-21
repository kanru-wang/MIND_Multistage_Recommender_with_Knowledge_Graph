from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mindrec.config import load_config
from mindrec.utils import device_info, log_device, resolve_device


def _requested_device(cfg: dict, component: str) -> str:
    if component == "teacher":
        return str(cfg["teacher"].get("device", "cuda"))
    if component == "ranker":
        return str(cfg["ranker"].get("device", "cuda"))
    raise ValueError(f"Unknown component: {component}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that PyTorch can run model work on the configured device."
    )
    parser.add_argument("--config", default="configs/mind_small.yaml")
    parser.add_argument("--component", choices=["teacher", "ranker"], default="ranker")
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(_requested_device(cfg, args.component))
    log_device(device, "GPU check")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(args.size, args.size, device=device)
    w = torch.randn(args.size, args.size, device=device, requires_grad=True)
    y = (x @ w).relu().mean()
    y.backward()
    torch.cuda.synchronize(device) if device.type == "cuda" else None

    info = device_info(device)
    print(f"torch_version: {info['torch_version']}")
    print(f"cuda_available: {info['cuda_available']}")
    print(f"cuda_version: {info['cuda_version']}")
    print(f"result_device: {y.device}")
    print(f"loss: {float(y.detach().cpu()):.6f}")
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"cuda_peak_memory_mb: {allocated:.1f}")


if __name__ == "__main__":
    main()
