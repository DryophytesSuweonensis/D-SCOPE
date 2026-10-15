"""Command-line runners and compact reproducible result exports."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .estimation import EstimationConfig, demodulate, estimate, load_detector, prepare_input_iq

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def load_record(path: Path) -> tuple[np.ndarray, float, dict]:
    with np.load(path, allow_pickle=False) as archive:
        original = np.asarray(archive["iq"])
        if not np.iscomplexobj(original) or original.ndim != 1:
            raise ValueError(f"Expected a one-dimensional complex I/Q array in {path}")
        iq = np.asarray(original, dtype=np.complex64)
        fs_hz = float(archive["fs_hz"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if iq.size < 4096 or not np.all(np.isfinite(iq)):
        raise ValueError(f"Invalid complex I/Q record: {path}")
    if fs_hz <= 0 or not np.isfinite(fs_hz):
        raise ValueError(f"Invalid sampling rate in {path}")
    return iq, fs_hz, metadata


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def spectrum(iq: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(iq, dtype=np.complex128)
    nfft = max(262144, len(x))
    transformed = np.fft.fftshift(np.fft.fft(x * np.hanning(len(x)), n=nfft))
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs_hz)), np.abs(transformed) ** 2


def _pnr(freqs: np.ndarray, power: np.ndarray, fc: float, bandwidth: float) -> float:
    p = power[np.abs(freqs - fc) <= max(2e6, 3 * bandwidth)].copy()
    if not p.size:
        p = power.copy()
    peak = int(np.argmax(p))
    keep = np.ones(len(p), dtype=bool)
    keep[max(0, peak - 7):min(len(p), peak + 8)] = False
    noise = p[keep] if np.count_nonzero(keep) >= 4 else p[p < p.max()]
    if not noise.size:
        noise = p
    floor = np.percentile(noise + 1e-12, 50)
    return float(10 * np.log10((p[peak] + 1e-12) / (floor + 1e-12)))


def _occupied_bandwidth(freqs: np.ndarray, power: np.ndarray, fc: float, bandwidth: float) -> float:
    """Peak-centered greedy interval containing 90% of local spectral power."""
    indices = np.flatnonzero(np.abs(freqs - fc) <= max(2e6, 3 * bandwidth))
    if not indices.size:
        return float("nan")
    p = np.maximum(power[indices], 0)
    target = 0.90 * (p.sum() + 1e-12)
    left = right = int(np.argmax(p))
    accumulated = float(p[left])
    while accumulated < target and (left > 0 or right < len(p) - 1):
        pl = p[left - 1] if left > 0 else -1
        pr = p[right + 1] if right < len(p) - 1 else -1
        if pr >= pl:
            right += 1
            accumulated += float(p[right])
        else:
            left -= 1
            accumulated += float(p[left])
    return float(freqs[indices[right]] - freqs[indices[left]])


def _spectrum_sample_indices(before: np.ndarray, after: np.ndarray, limit: int = 8192) -> np.ndarray:
    """Keep both curves' minima and maxima in each frequency block."""
    width = max(1, int(np.ceil(len(before) / max(1, limit // 4))))
    selected = [0, len(before) - 1]
    for start in range(0, len(before), width):
        end = min(len(before), start + width)
        for curve in (before, after):
            selected.extend([start + int(np.argmin(curve[start:end])),
                             start + int(np.argmax(curve[start:end]))])
    return np.unique(selected)


def save_results(out: Path, iq: np.ndarray, cfg: EstimationConfig, metadata: dict,
                 result: dict, image: np.ndarray | None, plots: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    y = demodulate(iq, result["bandwidth_hz"], result["modulation_hz"], result["phi_rad"], cfg)
    freqs, before = spectrum(iq, cfg.fs_hz)
    _, after = spectrum(y, cfg.fs_hz)
    result["pnr_before_db"] = _pnr(freqs, before, result["fc_hz"], result["bandwidth_hz"])
    result["pnr_after_db"] = _pnr(freqs, after, result["fc_hz"], result["bandwidth_hz"])
    result["pnr_gain_db"] = result["pnr_after_db"] - result["pnr_before_db"]
    result["occupied_bandwidth_before_hz"] = _occupied_bandwidth(freqs, before, result["fc_hz"], result["bandwidth_hz"])
    result["occupied_bandwidth_after_hz"] = _occupied_bandwidth(freqs, after, result["fc_hz"], result["bandwidth_hz"])
    result["bandwidth_compression_ratio"] = result["occupied_bandwidth_before_hz"] / max(result["occupied_bandwidth_after_hz"], cfg.fs_hz / len(freqs))
    target = np.abs(freqs - result["fc_hz"]) <= 250e3
    result["target_band_half_width_hz"] = 250e3
    result["target_band_fraction_before"] = float(before[target].sum() / (before.sum() + 1e-12))
    result["target_band_fraction_after"] = float(after[target].sum() / (after.sum() + 1e-12))
    truth = metadata.get("dominant_truth", {})
    for field in ("fc_hz", "bandwidth_hz", "modulation_hz"):
        if field in truth:
            result[f"{field}_truth"] = float(truth[field])
            result[f"{field}_error"] = result[field] - float(truth[field])
    if all(field in truth for field in ("fc_hz", "bandwidth_hz", "modulation_hz")):
        result["modulation_relative_error"] = abs(result["modulation_hz_error"]) / max(abs(float(truth["modulation_hz"])), 1e-12)
        result["joint_parameter_success"] = bool(abs(result["fc_hz_error"]) <= 0.5e6 and
                                                 abs(result["bandwidth_hz_error"]) <= 0.5e6 and
                                                 result["modulation_relative_error"] <= 0.001)
    write_json(out / "parameters.json", {**result, "sample_count": len(iq), "fs_hz": cfg.fs_hz,
                                         "config": asdict(cfg), "source_metadata": metadata,
                                         "metric_definitions": {
                                             "spectrum": "Hann window, FFT length max(262144, record samples)",
                                             "target_band_fraction": "power within final fc +/-250 kHz / full spectrum power; same interval before and after",
                                             "occupied_bandwidth": "peak-centered greedy interval containing 90% of power within final fc +/-max(2 MHz, 3*estimated bandwidth)",
                                             "bandwidth_compression_ratio": "occupied bandwidth before / after (minimum denominator: one FFT bin)",
                                             "pnr": "diagnostic peak / median power in the occupied-bandwidth search interval, excluding +/-7 bins; includes filtered-out bands and is not a calibrated SNR"
                                         }})
    np.savez_compressed(out / "demodulated.npz", iq=y, fs_hz=cfg.fs_hz,
                        metadata_json=json.dumps({"method": result["method"], "fc_hz": result["fc_hz"],
                                                  "bandwidth_hz": result["bandwidth_hz"],
                                                  "modulation_hz": result["modulation_hz"],
                                                  "phi_rad": result["phi_rad"]}))
    np.savez_compressed(out / "spectra.npz", frequency_hz=freqs, before_power=before, after_power=after)
    before_db = 10 * np.log10(before / (before.max() + 1e-12) + 1e-12)
    after_db = 10 * np.log10(after / (after.max() + 1e-12) + 1e-12)
    select = _spectrum_sample_indices(before, after)
    np.savetxt(out / "spectra.csv", np.column_stack((freqs[select], before_db[select], after_db[select])),
               delimiter=",", header="frequency_hz,before_normalized_db,after_normalized_db", comments="")
    if image is not None:
        Image.fromarray(image).save(out / "alpha_f.png")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
        ax.plot(freqs[select] / 1e6, before_db[select], color="0.65", linewidth=0.7, label="Before demodulation")
        ax.plot(freqs[select] / 1e6, after_db[select], color="black", linewidth=0.7, label="After demodulation")
        ax.set(xlabel="Baseband frequency (MHz)", ylabel="Normalized power (dB)", ylim=(-90, 3),
               title=f'{out.parent.name}: {result["method"]}')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.savefig(out / "spectrum.png", dpi=180)
        fig.savefig(out / "spectrum.pdf")
        plt.close(fig)
    return {k: v for k, v in result.items() if not isinstance(v, (dict, list))}


def run_cli(kind: str) -> int:
    parser = argparse.ArgumentParser(description=f"Estimate and demodulate {kind} triangular-FM examples.")
    choices = ["all", "1", "2", "3"] if kind == "synthetic" else ["all", "A", "B", "C"]
    parser.add_argument("--case", choices=choices, default="all")
    parser.add_argument("--method", choices=["proposed", "baseline", "both"], default="proposed")
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, ... (default: cpu)")
    parser.add_argument("--output", type=Path, default=RELEASE_ROOT / "outputs" / kind)
    parser.add_argument("--input", type=Path, help="Process one custom NPZ (overrides --case).")
    parser.add_argument("--weights", type=Path, default=RELEASE_ROOT / "models" / "rtdetr_preprocessed.pt")
    parser.add_argument("--recompute-image", action="store_true", help="Compute alpha-f from I/Q even when a bundled image is available.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for PyTorch (default: 8).")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    methods = ["proposed", "baseline"] if args.method == "both" else [args.method]
    model = None
    if "proposed" in methods:
        import torch
        torch.set_num_threads(args.threads)
        if args.device != "cpu" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable. Use --device cpu.")
        model = load_detector(args.weights)
    selected = choices[1:] if args.case == "all" else [args.case]
    prefix = "emitter" if kind == "synthetic" else "monitor"
    paths = [args.input.resolve()] if args.input else [RELEASE_ROOT / "data" / kind / f"{prefix}_{case}.npz" for case in selected]
    rows = []
    failures = 0
    for path in paths:
        raw_iq, fs_hz, metadata = load_record(path)
        if kind == "measured" and not args.input and len(raw_iq) != 50000:
            raise ValueError(f"Bundled measured record must contain exactly 50,000 samples: {path}")
        iq, processing = prepare_input_iq(raw_iq, fs_hz, metadata)
        cfg = EstimationConfig(fs_hz=fs_hz, device=args.device,
                               alpha_min_hz=float(metadata.get("alpha_min_hz", 1e3 if kind == "synthetic" else 21e3)),
                               alpha_max_hz=float(metadata.get("alpha_max_hz", 128e3 if kind == "synthetic" else 148e3)),
                               alpha_step_hz=float(metadata.get("alpha_step_hz", 1e3)),
                               map_style=metadata.get("map_style", "revision" if kind == "synthetic" else "legacy"))
        image = None
        image_source = "computed_from_iq"
        image_name = metadata.get("alpha_f_image")
        if image_name and not args.recompute_image and processing["processing_matches_reference"] is not False:
            image_path = RELEASE_ROOT / image_name
            if not image_path.is_file():
                image_path = path.parent / image_name
            if image_path.is_file():
                image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
                image_source = "bundled_alpha_f_image"
        for method in methods:
            out = args.output.resolve() / path.stem / method
            print(f"[{path.stem}] {method}: {len(iq):,} samples, {cfg.map_style} map", flush=True)
            try:
                result, used_image = estimate(iq, cfg, method=method, model=model, image=image)
                result["input_file"] = path.name
                result["detector_image_source"] = image_source if method == "proposed" else "unused"
                result["iq_input_stage"] = processing["input_stage"]
                result["raw_input_sample_count"] = len(raw_iq)
                result["processing_sample_count"] = len(iq)
                result["processing"] = {**processing,
                                         "detector_preprocessing": processing["detector_preprocessing"] if method == "proposed" else "unused"}
                summary = save_results(out, iq, cfg, metadata, result, used_image, plots=not args.no_plots)
                rows.append({"case": path.stem, **summary})
                if used_image is not None:
                    image = used_image
                print(f'  fc={result["fc_hz"] / 1e6:.6f} MHz, B={result["bandwidth_hz"] / 1e6:.6f} MHz, '
                      f'fm={result["modulation_hz"] / 1e3:.6f} kHz, '
                      f'90% BW compression={result["bandwidth_compression_ratio"]:.2f}x, '
                      f'target power={result["target_band_fraction_before"]:.1%} -> {result["target_band_fraction_after"]:.1%}', flush=True)
            except Exception as exc:
                failures += 1
                out.mkdir(parents=True, exist_ok=True)
                write_json(out / "parameters.json", {"success": False, "method": method, "error": str(exc)})
                rows.append({"case": path.stem, "method": method, "success": False, "error": str(exc)})
                print(f"  Failed: {exc}", file=sys.stderr, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {args.output.resolve()}", flush=True)
    return 1 if failures else 0
