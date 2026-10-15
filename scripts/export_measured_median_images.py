#!/usr/bin/env python3
"""Export measured alpha-f maps before and after only the IF median step.

Both branches start from the configured band-filtered I/Q, use phase-only
normalization and instantaneous frequency, and reconstruct 49,999 complex
samples. Only the after branch applies the IF median filter. Each alpha-f map
uses its own linear minimum/maximum normalization, matching the detector cache.
Raw NPZ records and the canonical after-median caches selected by metadata
are read only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssc_tfm.cli import load_record
from ssc_tfm.estimation import (
    EstimationConfig,
    instantaneous_frequency,
    make_alpha_f_image,
    median_preprocess,
    phase_only,
    prepare_input_iq,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_if_median(iq: np.ndarray, cfg: EstimationConfig) -> np.ndarray:
    """Match the after branch's sample count and phase origin, omitting median."""
    trace = instantaneous_frequency(phase_only(iq), cfg.fs_hz)
    phase = 2 * np.pi * np.cumsum(np.asarray(trace, dtype=float)) / cfg.fs_hz
    return np.exp(1j * phase).astype(np.complex64)


def comparison_sheet(rows: list[dict], output: Path) -> None:
    """Make an overview; full-resolution numerical maps remain separate PNGs."""
    panel = 640
    top = 100
    row_height = panel + 50
    canvas = Image.new("RGB", (panel * 2, top + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default(size=28)
    draw.text((24, 30), "Before IF median", fill="black", font=font)
    draw.text((panel + 24, 30), "After IF median (k=9)", fill="black", font=font)
    for index, row in enumerate(rows):
        y = top + index * row_height
        draw.text((24, y), f'Monitor {row["monitor"]}', fill="black", font=font)
        for column, key in enumerate(("before_path", "after_path")):
            with Image.open(row[key]) as source:
                thumb = source.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (column * panel, y + 45))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", "A", "B", "C"], default="all")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA index, for example 0")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads (default: 8)")
    parser.add_argument("--output", type=Path, default=ROOT / "data/measured")
    parser.add_argument("--comparison", type=Path, help="Optional before/after overview PNG")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    cases = list("ABC") if args.case == "all" else [args.case]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    mismatch = False
    for monitor in cases:
        record_path = ROOT / f"data/measured/monitor_{monitor}.npz"
        record_hash_before = sha256(record_path)
        raw, fs_hz, metadata = load_record(record_path)
        cache = (ROOT / str(metadata["alpha_f_image"])).resolve()
        cache_hash_before = sha256(cache)
        iq, processing = prepare_input_iq(raw, fs_hz, metadata)
        cfg = EstimationConfig(
            fs_hz=fs_hz,
            device=args.device,
            alpha_min_hz=float(metadata.get("alpha_min_hz", 21e3)),
            alpha_max_hz=float(metadata.get("alpha_max_hz", 148e3)),
            alpha_step_hz=float(metadata.get("alpha_step_hz", 1e3)),
            map_style=metadata.get("map_style", "legacy"),
            max_tau=int(metadata.get("max_tau", 1920)),
            alpha_upscale=int(metadata.get("alpha_upscale", 30)),
            median_kernel=int(metadata.get("median_kernel", 9)),
        )
        print(f"Monitor {monitor}: computing maps before and after IF median", flush=True)
        before_signal = without_if_median(iq, cfg)
        after_signal = median_preprocess(iq, cfg)
        if len(before_signal) != len(after_signal) or len(before_signal) != len(raw) - 1:
            raise RuntimeError("The comparison branches must have equal N-1 sample counts.")
        before = make_alpha_f_image(before_signal, cfg)
        after = make_alpha_f_image(after_signal, cfg)
        before_path = output / f"monitor_{monitor}_alpha_f_before_median.png"
        after_path = output / f"monitor_{monitor}_alpha_f_after_median.png"
        if before_path.resolve() == cache:
            raise RuntimeError("The before-median output cannot replace the canonical detector cache.")
        same_after_path = after_path.resolve() == cache
        Image.fromarray(before).save(before_path)
        with Image.open(cache) as source:
            cached = np.asarray(source.convert("L"), dtype=np.uint8)
        same_shape = after.shape == cached.shape
        same_pixels = same_shape and np.array_equal(after, cached)
        difference = np.abs(after.astype(np.int16) - cached.astype(np.int16)) if same_shape else None
        if same_pixels:
            # Regeneration has been checked. Preserve the cache's exact PNG
            # encoding as well as its pixels in the explicit after image.
            if not same_after_path:
                shutil.copyfile(cache, after_path)
        else:
            if not same_after_path:
                Image.fromarray(after).save(after_path)
            mismatch = True
        record_unchanged = sha256(record_path) == record_hash_before
        cache_unchanged = sha256(cache) == cache_hash_before
        if not record_unchanged or not cache_unchanged:
            raise RuntimeError("A source record or canonical detector cache was modified.")
        row = {
            "monitor": monitor,
            "raw_sample_count": len(raw),
            "before_sample_count": len(before_signal),
            "after_sample_count": len(after_signal),
            "width": int(after.shape[1]),
            "height": int(after.shape[0]),
            "before_path": str(before_path),
            "after_path": str(after_path),
            "normalization": "Each map independently uses linear minimum/maximum normalization to uint8 0..255.",
            "before_stage": "front_end_filtered_iq -> phase_only -> instantaneous_frequency -> phase_reconstruction",
            "after_stage": "front_end_filtered_iq -> phase_only -> instantaneous_frequency -> median_filter -> phase_reconstruction",
            "config": asdict(cfg),
            "processing_matches_reference": processing["processing_matches_reference"],
            "after_cache_pixels_identical": bool(same_pixels),
            "after_cache_png_bytes_identical": sha256(after_path) == cache_hash_before,
            "after_cache_differing_pixels": int(np.count_nonzero(difference)) if difference is not None else None,
            "after_cache_max_abs_pixel_difference": int(difference.max()) if difference is not None else None,
            "before_after_differing_pixels": int(np.count_nonzero(before != after)),
            "before_after_mean_abs_pixel_difference": float(np.mean(np.abs(before.astype(np.int16) - after.astype(np.int16)))),
            "raw_npz_unchanged": record_unchanged,
            "existing_alias_unchanged": cache_unchanged,
        }
        rows.append(row)
        print(f"  after/cache identical pixels={same_pixels}, identical PNG bytes={row['after_cache_png_bytes_identical']}; raw and canonical cache unchanged", flush=True)
    if args.comparison:
        comparison_sheet(rows, args.comparison.resolve())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        public_rows = [{**row,
                        "before_path": Path(row["before_path"]).name,
                        "after_path": Path(row["after_path"]).name}
                       for row in rows]
        args.report.write_text(json.dumps({"cases": public_rows}, indent=2) + "\n", encoding="utf-8")
    if mismatch:
        print("Regenerated after pixels differ from at least one canonical cache; see the report. Canonical caches and raw records were preserved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
