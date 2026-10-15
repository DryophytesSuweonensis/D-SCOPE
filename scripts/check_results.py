#!/usr/bin/env python3
"""Validate newly generated outputs for the six bundled examples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_group(kind, output):
    labels = ("1", "2", "3") if kind == "synthetic" else ("A", "B", "C")
    prefix = "emitter" if kind == "synthetic" else "monitor"
    expected_count = 134003 if kind == "synthetic" else 50000
    method_sets = []
    passed = 0
    for label in labels:
        case = f"{prefix}_{label}"
        with np.load(ROOT / f"data/{kind}/{case}.npz", allow_pickle=False) as record:
            input_metadata = json.loads(str(record["metadata_json"].item()))
            fs_hz = float(record["fs_hz"])
            input_hash = hashlib.sha256(record["iq"].tobytes(order="C")).hexdigest()
        methods = {method for method in ("proposed", "baseline")
                   if (output / case / method / "parameters.json").is_file()}
        require(methods, f"Missing results: {kind}/{case}")
        method_sets.append(methods)
        for method in sorted(methods):
            folder = output / case / method
            params = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
            name = f"{kind}/{case}/{method}"
            require(params.get("success") is True, f"Execution failed: {name}")
            require(params.get("sample_count") == expected_count, f"Wrong sample count: {name}")
            require(params.get("fs_hz") == fs_hz, f"Wrong sampling rate: {name}")
            require(params.get("processing", {}).get("input_iq_sha256") == input_hash,
                    f"Results belong to different input data: {name}; rerun this case")
            for field in ("fc_hz", "bandwidth_hz", "modulation_hz", "phi_rad"):
                require(np.isfinite(float(params[field])), f"Nonfinite {field}: {name}")
            require(0 < params["bandwidth_hz"] <= fs_hz and
                    0 < params["modulation_hz"] < fs_hz / 2,
                    f"Invalid physical parameters: {name}")
            with np.load(folder / "demodulated.npz", allow_pickle=False) as record:
                iq = record["iq"]
                require(iq.dtype == np.complex64 and iq.shape == (expected_count,) and
                        np.all(np.isfinite(iq)), f"Invalid demodulated I/Q: {name}")
                require(float(record["fs_hz"]) == fs_hz, f"Wrong demodulated sampling rate: {name}")
            if kind == "synthetic":
                truth = input_metadata["dominant_truth"]
                require(abs(params["fc_hz"] - truth["fc_hz"]) <= 500000,
                        f"Center-frequency error exceeds 500 kHz: {name}")
                require(abs(params["bandwidth_hz"] - truth["bandwidth_hz"]) <= 500000,
                        f"Bandwidth error exceeds 500 kHz: {name}")
                relative_error = abs(params["modulation_hz"] - truth["modulation_hz"]) / truth["modulation_hz"]
                require(relative_error <= 0.001, f"Modulation-rate error exceeds 0.1%: {name}")
            else:
                before = float(params["target_band_fraction_before"])
                after = float(params["target_band_fraction_after"])
                require(0 <= before < after <= 1, f"Target concentration did not improve: {name}")
                require(float(params["bandwidth_compression_ratio"]) > 1,
                        f"Occupied bandwidth did not decrease: {name}")
            print(f"PASS {name}")
            passed += 1
    require(all(methods == method_sets[0] for methods in method_sets),
            f"Incomplete method coverage in {kind} results")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=ROOT / "outputs/synthetic")
    parser.add_argument("--measured", type=Path, default=ROOT / "outputs/measured")
    args = parser.parse_args()
    passed = check_group("synthetic", args.synthetic) + check_group("measured", args.measured)
    print(f"Result validation passed: {passed} case/method combinations")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Result validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
