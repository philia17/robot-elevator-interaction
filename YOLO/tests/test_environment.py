from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_environment_reports_cuda_gpu() -> None:
    module = load_module()
    info = module.collect_environment(Path("D:/yolo/project/Lift_Robot_Project"), Path("D:/yolo/trainingruns/pretrained/yolo11n-seg.pt"))
    assert info["cuda_available"] is True
    assert "RTX 4070" in str(info["gpu_name"])
    assert info["torch_cuda"] == "12.8"
