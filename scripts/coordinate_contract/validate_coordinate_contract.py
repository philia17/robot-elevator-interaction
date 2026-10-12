#!/usr/bin/env python3
"""CLI for validating the frozen contract and optional button target YAML."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from coordinate_contract import ContractError, load_contract, validate_contract, validate_target


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "coordinate_contract_v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--require-execution-ready", action="store_true")
    args = parser.parse_args()
    try:
        contract = load_contract(args.contract)
        ready = validate_contract(contract)
        if args.require_execution_ready and not ready:
            raise ContractError("contract is valid but calibration is not execution-ready")
        if args.target:
            target = yaml.safe_load(args.target.read_text(encoding="utf-8"))
            validate_target(target, contract, now_ns=time.time_ns())
        print(
            f"coordinate_contract={contract['version']} valid=true "
            f"execution_ready={str(ready).lower()}"
        )
        return 0
    except (ContractError, OSError, yaml.YAMLError) as exc:
        print(f"coordinate contract validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
