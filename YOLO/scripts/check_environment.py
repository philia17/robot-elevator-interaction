from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path("D:/yolo/project/Lift_Robot_Project")
DEFAULT_MODEL = Path("D:/yolo/trainingruns/pretrained/yolo11n-seg.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the Windows YOLO runtime environment.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def collect_environment(project_root: Path, model: Path) -> dict[str, object]:
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    total_memory_mib = None
    if cuda_available:
        total_memory_mib = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "project_root": str(project_root),
        "model": str(model),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_total_memory_mib": total_memory_mib,
        "ultralytics_version": ultralytics.__version__,
    }


def main() -> int:
    args = parse_args()
    info = collect_environment(args.project_root, args.model)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    if not info["cuda_available"]:
        print("CUDA is unavailable; stop and diagnose before training.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
