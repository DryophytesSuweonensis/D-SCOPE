#!/usr/bin/env python3
"""Check the portable release files using only Python and NumPy.

Run from any directory: python /path/to/github_release/scripts/verify_release.py
This checks stored data and references; it does not run the neural estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FS_HZ = 67_001_500.0
IGNORED_DIRS = {
    "outputs", "__pycache__", ".git", ".cache", ".pytest_cache", ".venv", "venv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_path(value):
    require(isinstance(value, str) and bool(value), "A path is empty or is not text")
    path = PurePosixPath(value)
    require(not path.is_absolute() and not PureWindowsPath(value).drive,
            f"Absolute path is not portable: {value}")
    require("\\" not in value and ".." not in path.parts,
            f"Path must be relative and cannot traverse parents: {value}")
    return path


def excluded(path):
    return any(part in IGNORED_DIRS for part in path.parts) or path.suffix == ".pyc"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(root):
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        print("SHA256SUMS absent: content checks will run without manifest hashes")
        return
    indexed = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"Invalid SHA256SUMS line {number}")
        expected, filename = match.groups()
        path = relative_path(filename)
        if excluded(path):
            continue
        require(str(path) != "SHA256SUMS", "Manifest cannot hash itself")
        require(str(path) not in indexed, f"Duplicate manifest path: {path}")
        target = root / str(path)
        require(target.is_file() and not target.is_symlink(), f"Missing/linked manifest file: {path}")
        require(sha256(target) == expected.lower(), f"SHA-256 mismatch: {path}")
        indexed.add(str(path))
    actual = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
        and not excluded(path.relative_to(root))
    }
    require(indexed == actual,
            f"Manifest coverage differs: unlisted={sorted(actual-indexed)}, missing={sorted(indexed-actual)}")
    print(f"SHA-256 verified for {len(indexed)} release files (outputs and caches excluded)")


def inspect_json(value, label="metadata"):
    if isinstance(value, dict):
        for key, item in value.items():
            require(key not in {"device_name", "manufacturer", "brand", "model_name", "serial_number",
                                "source_path", "file_path", "capture_filename", "original_source",
                                "original_alpha_f_source", "original_sample_metadata"},
                    f"Identifying or unnecessary source metadata is not allowed: {label}.{key}")
            inspect_json(item, f"{label}.{key}")
            is_path = key == "alpha_f_image"
            if is_path and item is not None:
                relative_path(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect_json(item, f"{label}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"Nonfinite JSON number: {label}")
    elif isinstance(value, str):
        require(not value.startswith(("/home/", "/tmp/", "/Users/")),
                f"Machine-specific absolute path: {label}")


def png_dimensions(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    require(len(header) == 24 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[12:16] == b"IHDR", f"Invalid PNG header: {path.name}")
    return struct.unpack(">II", header[16:24])


def png_text_metadata(path):
    fields = {}
    with path.open("rb") as stream:
        require(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {path.name}")
        while True:
            header = stream.read(8)
            require(len(header) == 8, f"Truncated PNG: {path.name}")
            size, kind = struct.unpack(">I4s", header)
            payload = stream.read(size)
            require(len(payload) == size and len(stream.read(4)) == 4,
                    f"Truncated PNG chunk: {path.name}")
            if kind == b"tEXt":
                key, separator, value = payload.partition(b"\0")
                require(bool(separator), f"Invalid PNG text field: {path.name}")
                fields[key.decode("latin-1")] = value.decode("latin-1")
            if kind == b"IEND":
                return fields


def check_alpha_f_png(path, metadata):
    require(png_dimensions(path) == (3841, 3840), f"Unexpected alpha-f PNG dimensions: {path.name}")
    require(metadata.get("alpha_f_image_orientation") == "alpha_max_at_top",
            f"PNG metadata must place minimum cyclic frequency at the bottom: {path.name}")
    fields = png_text_metadata(path)
    require(fields.get("alpha_f_image_orientation") == "alpha_max_at_top",
            f"PNG must display cyclic frequency increasing upward: {path.name}")
    for key in ("alpha_min_hz", "alpha_max_hz"):
        require(float(fields[key]) == float(metadata[key]), f"Wrong PNG {key}: {path.name}")


def load_record(path, expected_samples):
    require(path.is_file(), f"Missing data record: {path}")
    with np.load(path, allow_pickle=False) as record:
        require(set(record.files) == {"iq", "fs_hz", "metadata_json"},
                f"Unexpected NPZ keys: {path.name}")
        for key in record.files:
            require(not record[key].dtype.hasobject, f"Object/pickle array: {path.name}:{key}")
        iq = record["iq"]
        require(iq.dtype == np.complex64 and iq.shape == (expected_samples,),
                f"I/Q must be {expected_samples} complex64 samples: {path.name}")
        require(bool(np.isfinite(iq).all()), f"Nonfinite I/Q samples: {path.name}")
        require(record["fs_hz"].shape == () and float(record["fs_hz"]) == FS_HZ,
                f"Wrong scalar sampling rate: {path.name}")
        require(record["metadata_json"].shape == () and record["metadata_json"].dtype.kind in "US",
                f"Metadata must be a scalar JSON string: {path.name}")
        metadata = json.loads(str(record["metadata_json"].item()))
    require(isinstance(metadata, dict), f"Metadata must be a JSON object: {path.name}")
    inspect_json(metadata)
    if "iq_sha256" in metadata:
        require(hashlib.sha256(iq.tobytes(order="C")).hexdigest() == metadata["iq_sha256"],
                f"Input I/Q fingerprint differs: {path.name}")
    if "fs_hz" in metadata:
        require(float(metadata["fs_hz"]) == FS_HZ, f"Wrong metadata sampling rate: {path.name}")
    for key in ("samples", "n_samples"):
        if key in metadata:
            require(int(metadata[key]) == expected_samples, f"Wrong metadata sample count: {path.name}")
    return metadata


def near(first, second, label):
    require(math.isclose(float(first), float(second), rel_tol=1e-9, abs_tol=1e-7),
            f"Values disagree: {label}")


def check_synthetic(metadata, count, results):
    require(metadata["kind"] == "synthetic", "Wrong synthetic kind")
    require(int(metadata["num_emitters"]) == count and int(metadata["num_signals"]) == count,
            "Wrong emitter count")
    require(metadata.get("map_style") == "revision", "Wrong synthetic alpha-f map convention")
    for key, expected in [("alpha_min_hz", 1000), ("alpha_max_hz", 128000), ("alpha_step_hz", 1000)]:
        require(float(metadata[key]) == expected, f"Wrong synthetic {key}")
    components = metadata["components"]
    require(len(components) == count, "Ground truth component count differs")
    dominant = [component for component in components if component.get("is_dominant") == 1]
    require(len(dominant) == 1, "Expected exactly one dominant truth component")
    dominant = dominant[0]
    truth = metadata["dominant_truth"]
    for key, component_key in [("fc_hz", "f_base"), ("bandwidth_hz", "fspan"), ("modulation_hz", "alpha")]:
        require(math.isfinite(float(truth[key])), f"Invalid truth {key}")
        near(truth[key], dominant[component_key], key)
    for component in components:
        require(float(component["amp"]) > 0 and float(component["alpha"]) > 0,
                "Invalid component amplitude/modulation rate")
        near(component["fspan"], 2 * float(component["f_range"]), "peak-to-peak bandwidth")
    require(results.get("success") is True, "Expected estimator execution did not succeed")
    require(results.get("processing", {}).get("input_iq_sha256") == metadata["iq_sha256"],
            "Expected synthetic result belongs to different input data")
    require(results.get("joint_parameter_success") is True, "Expected synthetic parameters are not jointly successful")
    for field, tolerance in [("fc_hz", 500_000.0), ("bandwidth_hz", 500_000.0), ("modulation_hz", None)]:
        error = float(results[field]) - float(truth[field])
        near(error, results[f"{field}_error"], f"{field} signed error")
        near(truth[field], results[f"{field}_truth"], f"{field} truth")
        if tolerance is not None:
            require(abs(error) <= tolerance, f"Expected result exceeds success threshold: {field}")
        else:
            relative = abs(error) / abs(float(truth[field]))
            near(relative, results["modulation_relative_error"], "modulation relative error")
            require(relative <= 0.001, "Expected modulation error exceeds 0.1%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="release directory to check")
    args = parser.parse_args()
    root = args.root.resolve()
    verify_hashes(root)
    model = root / "models/rtdetr_preprocessed.pt"
    require(model.is_file() and model.stat().st_size > 1_000_000,
            "Model is missing or unexpectedly small")
    require(zipfile.is_zipfile(model), "Model is not a supported checkpoint archive")
    with zipfile.ZipFile(model) as archive:
        require(any(name.endswith("/data.pkl") for name in archive.namelist()),
                "Checkpoint archive has no model structure")
        require(any("/data/" in name for name in archive.namelist()),
                "Checkpoint archive has no numeric weights")
    print(f"Verified model archive: {model.stat().st_size:,} bytes; integrity covered by SHA256SUMS")
    for count in (1, 2, 3):
        path = root / f"data/synthetic/emitter_{count}.npz"
        metadata = load_record(path, 134_003)
        for method in ("proposed", "baseline"):
            expected = root / f"expected_results/synthetic/emitter_{count}/{method}/parameters.json"
            results = json.loads(expected.read_text(encoding="utf-8"))
            inspect_json(results, f"expected_synthetic_{count}_{method}")
            check_synthetic(metadata, count, results)
        image = root / str(relative_path(metadata["alpha_f_image"]))
        check_alpha_f_png(image, metadata)
        print(f"Verified {path.relative_to(root)}: 134003 samples; expected estimates meet accuracy thresholds")
    for monitor in ("A", "B", "C"):
        path = root / f"data/measured/monitor_{monitor}.npz"
        metadata = load_record(path, 50_000)
        require(metadata.get("kind") == "measured", f"Wrong measured kind: {path.name}")
        require(metadata.get("has_ground_truth") is False, f"Measured data cannot claim ground truth: {path.name}")
        require(metadata.get("map_style") == "legacy", f"Wrong measured alpha-f map convention: {path.name}")
        require(metadata.get("prefiltered") is False and metadata.get("input_stage") == "raw_iq",
                f"Measured samples must be raw, before preprocessing: {path.name}")
        require(metadata.get("alpha_f_input_stage") == "front_end_filtered_then_phase_if_median",
                f"Missing cached-image preprocessing provenance: {path.name}")
        front_end = metadata.get("front_end_filter", {})
        require(front_end.get("kind") == "raised_cosine", f"Wrong measured front-end filter: {path.name}")
        for key, expected in [("bandwidth_hz", 30e6), ("beta", 0.1), ("center_hz", 0.0)]:
            require(float(front_end[key]) == expected, f"Wrong measured front-end {key}: {path.name}")
        with np.load(path, allow_pickle=False) as record:
            iq = record["iq"]
        for component in (iq.real, iq.imag):
            require(bool(np.all(component == np.round(component)))
                    and bool(np.all(component >= -32768)) and bool(np.all(component <= 32767)),
                    f"Raw measured I/Q must contain unprocessed signed 16-bit ADC values: {path.name}")
        if "raw_iq_sha256" in metadata:
            require(hashlib.sha256(iq.tobytes(order="C")).hexdigest() == metadata["raw_iq_sha256"],
                    f"Raw measured I/Q fingerprint differs: {path.name}")
        if "processing_iq_sha256" in metadata:
            require(bool(re.fullmatch(r"[0-9a-f]{64}", str(metadata["processing_iq_sha256"]))),
                    f"Invalid processing-I/Q fingerprint: {path.name}")
        for key, expected in [("alpha_min_hz", 21000), ("alpha_max_hz", 148000), ("alpha_step_hz", 1000)]:
            require(float(metadata[key]) == expected, f"Wrong measured {key}: {path.name}")
        for key in ("monitor_key", "monitor"):
            if key in metadata:
                require(str(metadata[key]) in {monitor, f"Monitor_{monitor}"}, f"Wrong monitor identity: {path.name}")
        for key in ("source_capture", "source_capture_id", "record_id"):
            if key in metadata:
                require(metadata[key] == monitor, f"Measured capture identity must use only A/B/C: {path.name}")
        image_name = metadata.get("alpha_f_image")
        require(image_name is not None, f"Missing measured alpha-f image metadata: {path.name}")
        image = root / str(relative_path(image_name))
        check_alpha_f_png(image, metadata)
        before_image = root / f"data/measured/monitor_{monitor}_alpha_f_before_median.png"
        after_image = root / f"data/measured/monitor_{monitor}_alpha_f_after_median.png"
        for pair_image in (before_image, after_image):
            check_alpha_f_png(pair_image, metadata)
        require(sha256(after_image) == sha256(image),
                f"After-median image differs from the detector cache: {after_image.name}")
        require(sha256(before_image) != sha256(after_image),
                f"Before/after median images are identical: {monitor}")
        for method in ("proposed", "baseline"):
            expected = root / f"expected_results/measured/monitor_{monitor}/{method}/parameters.json"
            result = json.loads(expected.read_text(encoding="utf-8"))
            inspect_json(result, f"expected_measured_{monitor}_{method}")
            require(result.get("success") is True and int(result["sample_count"]) == 50_000,
                    f"Invalid measured expected execution: {monitor}/{method}")
            require(result.get("processing", {}).get("input_iq_sha256") == metadata["raw_iq_sha256"],
                    f"Expected measured result belongs to different input data: {monitor}/{method}")
            require("joint_parameter_success" not in result, "Measured results cannot claim truth-based accuracy")
        print(f"Verified {path.relative_to(root)}: 50000 raw measured samples; preprocessing recorded separately")
    print("Release verification passed: model, six data records, cached images, median PNG pairs, truth, references, and portable paths")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
