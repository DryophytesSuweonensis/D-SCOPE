"""Portable dominant-emitter estimation and triangular-FM demodulation.

The physical model uses peak-to-peak sweep bandwidth. Synthetic and measured
examples retain their respective alpha-f representation conventions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time
import hashlib

import numpy as np
from PIL import Image
from scipy.signal import czt, medfilt

EPS = 1e-12


@dataclass(frozen=True)
class EstimationConfig:
    fs_hz: float = 67.0015e6
    alpha_min_hz: float = 1e3
    alpha_max_hz: float = 128e3
    alpha_step_hz: float = 1e3
    map_style: str = "revision"
    max_tau: int = 1920
    alpha_upscale: int = 30
    median_kernel: int = 9
    image_size: int = 3840
    confidence: float = 0.10
    iou: float = 0.10
    max_detections: int = 20
    target_class: int = 0
    device: str = "cpu"
    pso_particles: int = 100
    pso_iterations: int = 35
    pso_seed: int = 0
    objective_samples: int = 2000
    harmonics: int = 30
    fm_min_hz: float = 1e3
    fm_max_hz: float = 148e3
    fm_half_width_hz: float = 1e3
    bd_half_width_hz: float = 5e6
    czt_bins: int = 4096
    czt_max_samples: int = 262144
    czt_min_samples: int = 4096
    fc_max_samples: int = 1048576


def raised_cosine_front_end(iq: np.ndarray, fs_hz: float, bandwidth_hz: float,
                            beta: float = 0.1, center_hz: float = 0.0) -> np.ndarray:
    """Capture front end with an explicit double-precision FFT.

    The frequency-axis convention and final complex64 cast match the supplied
    examples. This operates on a frame without loading or trimming a capture.
    """
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if len(x) < 4 or bandwidth_hz <= 0 or not 0 < beta <= 1:
        raise ValueError("Raised-cosine filtering requires positive bandwidth and 0 < beta <= 1.")
    length = len(x)
    first = fs_hz * np.arange(int(length / 2) + 1) / length
    second = first[0:int(length / 2) - 1] - fs_hz / 2 + first[1]
    frequency = np.concatenate((first, second))
    period = 1 / bandwidth_hz
    response = np.zeros_like(frequency)
    absolute = np.abs(frequency - center_hz)
    pass_edge = (1 - beta) / (2 * period)
    stop_edge = (1 + beta) / (2 * period)
    response[absolute <= pass_edge] = 1.0
    transition = (absolute > pass_edge) & (absolute <= stop_edge)
    values = np.pi * period / beta * (absolute[transition] - pass_edge)
    response[transition] = 0.5 * (1 + np.cos(values))
    transformed = np.fft.fft(x.astype(np.complex128))
    if len(transformed) != len(response):
        length = min(len(transformed), len(response))
        transformed, response = transformed[:length], response[:length]
    return np.fft.ifft(transformed * response).astype(np.complex64)


def prepare_input_iq(iq: np.ndarray, fs_hz: float, metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply only an explicitly recorded front end to raw input I/Q.

    Custom records without front_end_filter are used directly. An explicitly
    prefiltered input is never filtered a second time.
    """
    raw = np.asarray(iq, dtype=np.complex64).reshape(-1)
    prefiltered = bool(metadata.get("prefiltered", False))
    input_stage = str(metadata.get("input_stage", "prefiltered_iq" if prefiltered else "raw_iq"))
    specification = metadata.get("front_end_filter")
    filtered = raw
    applied = None
    if specification and not prefiltered:
        if not isinstance(specification, dict) or specification.get("kind") != "raised_cosine":
            raise ValueError("front_end_filter must specify kind='raised_cosine'.")
        applied = {"kind": "raised_cosine", "bandwidth_hz": float(specification["bandwidth_hz"]),
                   "beta": float(specification.get("beta", 0.1)),
                   "center_hz": float(specification.get("center_hz", 0.0))}
        filtered = raised_cosine_front_end(raw, fs_hz, applied["bandwidth_hz"], applied["beta"], applied["center_hz"])
    raw_hash = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
    processing_hash = hashlib.sha256(filtered.tobytes(order="C")).hexdigest()
    if metadata.get("raw_iq_sha256") and metadata["raw_iq_sha256"] != raw_hash:
        raise ValueError("Raw I/Q fingerprint differs from metadata.")
    expected_processing_hash = metadata.get("processing_iq_sha256")
    processing_stage = "front_end_filtered_iq" if applied else input_stage
    input_preprocessing = [input_stage, "raised_cosine_filter", processing_stage] if applied else [input_stage]
    return filtered, {"input_stage": input_stage, "input_sample_count": len(raw),
                       "input_iq_sha256": raw_hash,
                       "processing_iq_stage": processing_stage,
                       "processing_sample_count": len(filtered),
                       "processing_iq_sha256": processing_hash,
                       "processing_matches_reference": processing_hash == expected_processing_hash if expected_processing_hash else None,
                       "front_end_filter_applied": applied,
                       "input_preprocessing": input_preprocessing,
                       "detector_preprocessing": input_preprocessing + ["phase_only", "instantaneous_frequency", "median_filter", "phase_reconstruction", "alpha_f_map"],
                       "demodulation_input_stage": processing_stage}


def phase_only(iq: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.angle(np.asarray(iq, dtype=np.complex64))).astype(np.complex64)


def instantaneous_frequency(iq: np.ndarray, fs_hz: float) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex128).reshape(-1)
    return fs_hz / (2 * np.pi) * np.angle(x[1:] * np.conj(x[:-1]))


def median_preprocess(iq: np.ndarray, cfg: EstimationConfig) -> np.ndarray:
    """Phase-only -> IF -> zero-padded median filter -> reconstructed phase."""
    trace = instantaneous_frequency(phase_only(iq), cfg.fs_hz)
    filtered = medfilt(trace, cfg.median_kernel)
    phase = 2 * np.pi * np.cumsum(np.asarray(filtered, dtype=float)) / cfg.fs_hz
    return np.exp(1j * phase).astype(np.complex64)


def alpha_f_frequency_axis(cfg: EstimationConfig) -> np.ndarray:
    n_tau = 2 * cfg.max_tau + 1
    if cfg.map_style == "legacy":
        # Preserve the legacy generator's inclusive endpoints, rather than
        # silently substituting the slightly different FFT-bin axis.
        return np.linspace(-cfg.fs_hz / 2, cfg.fs_hz / 2, n_tau, dtype=np.float32)
    return np.fft.fftshift(np.fft.fftfreq(n_tau, d=1 / cfg.fs_hz))


def make_alpha_f_image(iq: np.ndarray, cfg: EstimationConfig, lag_batch: int = 32) -> np.ndarray:
    """Compute a training-compatible alpha-f image with bounded working memory.

    The lag products and alpha sums are evaluated in batches using matrix
    multiplication. This preserves each experiment's mathematical definition;
    floating-point reductions can differ slightly across hardware/backends.
    """
    import torch

    device = torch.device("cpu" if cfg.device == "cpu" else
                          (cfg.device if str(cfg.device).startswith("cuda") else f"cuda:{cfg.device}"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu.")
    x = torch.as_tensor(np.asarray(iq, dtype=np.complex64), device=device).flatten()
    n_samples = x.numel()
    if n_samples <= cfg.max_tau:
        raise ValueError("Record must contain more than max_tau samples.")
    n = torch.arange(n_samples, device=device, dtype=torch.float32)
    t = n / cfg.fs_hz
    alphas = torch.arange(cfg.alpha_min_hz, cfg.alpha_max_hz + 1e-6,
                          cfg.alpha_step_hz, device=device, dtype=torch.float32)
    if cfg.map_style == "revision":
        exponent = torch.exp(-1j * 2.0 * np.pi * alphas[:, None] * t[None, :])
        window = torch.hann_window(n_samples, device=device, dtype=torch.float32)
        window = window / (window.mean() + EPS)
    elif cfg.map_style == "legacy":
        # Match the legacy scalar-alpha multiplication order in float32.
        exponent = torch.stack([torch.exp(-1j * 2.0 * np.pi * a * t) for a in alphas])
        window = None
    else:
        raise ValueError(f"Unknown map_style: {cfg.map_style}")
    taus = list(range(-cfg.max_tau, cfg.max_tau + 1))
    caf = torch.empty((len(alphas), len(taus)), dtype=torch.complex64, device=device)
    for start in range(0, len(taus), lag_batch):
        block = taus[start:start + lag_batch]
        products = torch.zeros((len(block), n_samples), dtype=torch.complex64, device=device)
        for j, tau in enumerate(block):
            if cfg.map_style == "revision":
                products[j] = x * torch.conj(torch.roll(x, shifts=tau)) * window
            elif tau >= 0:
                products[j, tau:] = x[tau:] * torch.conj(x[:n_samples - tau])
            else:
                products[j, :n_samples + tau] = x[:n_samples + tau] * torch.conj(x[-tau:])
        values = (exponent @ products.transpose(0, 1)) / n_samples
        if cfg.map_style == "revision":
            lag_seconds = torch.tensor(block, dtype=torch.float32, device=device) / cfg.fs_hz
            values *= torch.exp(1j * np.pi * alphas[:, None] * lag_seconds[None, :])
        caf[:, start:start + len(block)] = values
    magnitude = torch.abs(torch.fft.fftshift(torch.fft.fft(caf, dim=1), dim=1)).cpu().numpy()
    magnitude = np.asarray(magnitude, dtype=np.float32)
    normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + EPS)
    image = (255.0 * normalized).clip(0, 255).astype(np.uint8)
    return np.repeat(image, cfg.alpha_upscale, axis=0)


def triangular_phase_cycles(t: np.ndarray, bandwidth_hz: float, modulation_hz: float,
                            phi_rad: float, harmonics: int = 30) -> np.ndarray:
    """Integrated triangular FM; bandwidth_hz is the full peak-to-peak sweep."""
    result = np.zeros_like(np.asarray(t, dtype=np.float64))
    for m in range(1, harmonics + 1):
        k = 2 * m - 1
        denominator = k ** 3 * np.pi ** 3 * modulation_hz + EPS
        result += 2 * bandwidth_hz / denominator * np.sin(k * (2 * np.pi * modulation_hz * t + phi_rad))
    return result


def demodulate(iq: np.ndarray, bandwidth_hz: float, modulation_hz: float,
               phi_rad: float, cfg: EstimationConfig) -> np.ndarray:
    x = np.asarray(iq, dtype=np.complex128).reshape(-1)
    t = np.arange(x.size, dtype=np.float64) / cfg.fs_hz
    phase = triangular_phase_cycles(t, bandwidth_hz, modulation_hz, phi_rad, cfg.harmonics)
    return (x * np.exp(-1j * 2 * np.pi * phase)).astype(np.complex64)


def quadratic_peak_delta(values: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(values) - 1:
        return 0.0
    y0, y1, y2 = (float(values[idx + offset]) for offset in (-1, 0, 1))
    denominator = y0 - 2 * y1 + y2
    if not np.isfinite(denominator) or abs(denominator) < EPS:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))


def estimate_rate_fft(iq: np.ndarray, cfg: EstimationConfig) -> float:
    y = instantaneous_frequency(phase_only(iq), cfg.fs_hz)
    y = y[np.isfinite(y)]
    y = (y - y.mean()) * np.hanning(y.size)
    magnitude = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1 / cfg.fs_hz)
    valid = np.flatnonzero((freqs >= cfg.fm_min_hz) & (freqs <= cfg.fm_max_hz))
    if not valid.size:
        raise ValueError("No IF-FFT bins in the requested modulation-frequency interval.")
    k = int(valid[np.argmax(magnitude[valid])])
    return float(np.clip((k + quadratic_peak_delta(magnitude, k)) * cfg.fs_hz / y.size,
                         cfg.fm_min_hz, cfg.fm_max_hz))


def refine_rate_czt(iq: np.ndarray, coarse_hz: float, cfg: EstimationConfig) -> float:
    lo = max(cfg.fm_min_hz, coarse_hz - cfg.fm_half_width_hz)
    hi = min(cfg.fm_max_hz, coarse_hz + cfg.fm_half_width_hz)
    if hi <= lo:
        raise ValueError("Invalid CZT frequency interval.")
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)[:cfg.czt_max_samples]
    if x.size < cfg.czt_min_samples:
        raise ValueError(f"CZT requires at least {cfg.czt_min_samples} samples.")
    y = instantaneous_frequency(phase_only(x), cfg.fs_hz)
    y = y[np.isfinite(y)]
    y = (y - y.mean()) * np.hanning(y.size)
    grid = np.linspace(lo, hi, cfg.czt_bins, dtype=np.float64)
    step = float(grid[1] - grid[0])
    spectrum = czt(y, m=cfg.czt_bins, w=np.exp(-1j * 2 * np.pi * step / cfg.fs_hz),
                   a=np.exp(1j * 2 * np.pi * lo / cfg.fs_hz))
    magnitude = np.abs(spectrum)
    k = int(np.argmax(magnitude))
    return float(np.clip(grid[k] + quadratic_peak_delta(magnitude, k) * step, lo, hi))


def pso_bandwidth_phase(iq: np.ndarray, modulation_hz: float, bounds_hz: tuple[float, float],
                        cfg: EstimationConfig) -> dict[str, Any]:
    """The revision's fixed 100-particle x 35-iteration bounded PSO."""
    n = min(cfg.objective_samples, len(iq))
    if n < 64:
        raise ValueError("PSO requires at least 64 samples.")
    x = np.asarray(iq[:n], dtype=np.complex128)
    t = np.arange(n, dtype=np.float64) / cfg.fs_hz

    def score(theta: np.ndarray) -> float:
        phase = triangular_phase_cycles(t, theta[0], modulation_hz, theta[1], cfg.harmonics)
        power = np.abs(np.fft.fft(x * np.exp(-1j * 2 * np.pi * phase))) ** 2
        probability = power / (power.sum() + EPS)
        concentration = np.sum(probability ** 2)
        entropy = -np.sum(probability * np.log(probability + EPS)) / np.log(n)
        return float(concentration - entropy)

    lo = np.array([bounds_hz[0], 0.0], dtype=float)
    hi = np.array([bounds_hz[1], 2 * np.pi], dtype=float)
    span = hi - lo
    if np.any(span <= 0):
        raise ValueError("Invalid bandwidth/phase search bounds.")
    rng = np.random.default_rng(cfg.pso_seed)
    positions = lo[None, :] + rng.random((cfg.pso_particles, 2)) * span[None, :]
    velocities = np.zeros_like(positions)
    personal_best = positions.copy()
    personal_scores = np.array([score(p) for p in positions])
    best_idx = int(np.argmax(personal_scores))
    best = personal_best[best_idx].copy()
    best_score = float(personal_scores[best_idx])
    for _ in range(cfg.pso_iterations):
        r1, r2 = rng.random(positions.shape), rng.random(positions.shape)
        velocities = 0.70 * velocities + 1.50 * r1 * (personal_best - positions) + 1.50 * r2 * (best[None, :] - positions)
        positions += velocities
        hit = (positions < lo[None, :]) | (positions > hi[None, :])
        velocities[hit] *= -0.5
        positions = np.clip(positions, lo[None, :], hi[None, :])
        scores = np.array([score(p) for p in positions])
        improved = scores > personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        best_idx = int(np.argmax(personal_scores))
        if float(personal_scores[best_idx]) > best_score:
            best, best_score = personal_best[best_idx].copy(), float(personal_scores[best_idx])
    return {"bandwidth_hz": float(best[0]), "phi_rad": float(best[1] % (2 * np.pi)),
            "pso_score": best_score, "pso_iterations": cfg.pso_iterations,
            "pso_particles": cfg.pso_particles,
            "pso_objective_evaluations": cfg.pso_particles * (cfg.pso_iterations + 1),
            "bandwidth_search_low_hz": float(lo[0]), "bandwidth_search_high_hz": float(hi[0])}


def carrier_after_demodulation(iq: np.ndarray, bandwidth_hz: float, modulation_hz: float,
                              phi_rad: float, cfg: EstimationConfig) -> float:
    x = demodulate(iq[:cfg.fc_max_samples], bandwidth_hz, modulation_hz, phi_rad, cfg)
    spectrum = np.fft.fftshift(np.fft.fft(np.asarray(x, dtype=np.complex128) * np.hanning(len(x))))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(len(x), d=1 / cfg.fs_hz))
    k = int(np.argmax(power))
    return float(freqs[k] + quadratic_peak_delta(power, k) * (freqs[1] - freqs[0]))


def load_detector(weights: str | Path):
    from ultralytics import RTDETR

    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"Detector weights not found: {path}. Restore models/rtdetr_preprocessed.pt "
            "or supply a checkpoint with --weights /path/to/best.pt. The --method baseline option needs no weights."
        )
    return RTDETR(str(path))


def detect_coarse(model, image: np.ndarray, cfg: EstimationConfig) -> dict[str, Any]:
    import torch

    rgb = np.repeat(np.asarray(image, dtype=np.uint8)[:, :, None], 3, axis=2)
    start = time.perf_counter()
    with torch.inference_mode():
        prediction = model.predict(source=rgb, imgsz=cfg.image_size, conf=cfg.confidence,
                                   iou=cfg.iou, device=cfg.device, max_det=cfg.max_detections,
                                   verbose=False, save=False, stream=False)[0]
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No detector boxes found for this record.")
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    valid = np.flatnonzero(classes == cfg.target_class)
    if not valid.size:
        raise RuntimeError("No fundamental-ridge detection found for this record.")
    idx = int(valid[np.argmax(confidences[valid])])
    x1, y1, x2, y2 = boxes.xyxy[idx].detach().cpu().numpy().astype(float)
    h, w = image.shape[:2]
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid detector box after clipping.")
    freqs = alpha_f_frequency_axis(cfg)
    span = float(freqs[-1] - freqs[0])
    return {"fc_hz": float(freqs[0] + (x1 + x2) * 0.5 / w * span),
            "bandwidth_hz": float((x2 - x1) / w * span),
            "modulation_hz": float(cfg.alpha_min_hz + (y1 + y2) * 0.5 / h * (cfg.alpha_max_hz - cfg.alpha_min_hz)),
            "confidence": float(confidences[idx]), "box_xyxy": [float(v) for v in (x1, y1, x2, y2)],
            "detector_runtime_ms": (time.perf_counter() - start) * 1000}


def estimate(iq: np.ndarray, cfg: EstimationConfig, method: str = "proposed", model=None,
             image: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray | None]:
    """Estimate the dominant component, never consulting ground-truth metadata."""
    start = time.perf_counter()
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    if method == "proposed":
        if model is None:
            raise ValueError("The proposed estimator requires a detector model.")
        preprocessed = median_preprocess(x, cfg)
        if image is None:
            image = make_alpha_f_image(preprocessed, cfg)
        coarse = detect_coarse(model, image, cfg)
        modulation = refine_rate_czt(preprocessed, coarse["modulation_hz"], cfg)
        bounds = (max(1.0, coarse["bandwidth_hz"] - cfg.bd_half_width_hz),
                  min(cfg.fs_hz, coarse["bandwidth_hz"] + cfg.bd_half_width_hz))
        label = "RT-DETR + CZT + PSO"
    elif method == "baseline":
        coarse = {"modulation_hz": estimate_rate_fft(x, cfg)}
        modulation = refine_rate_czt(x, coarse["modulation_hz"], cfg)
        bounds = (0.5e6, 25e6)
        label = "IF-FFT-CZT-PSO"
    else:
        raise ValueError(f"Unknown method: {method}")
    optimized = pso_bandwidth_phase(x, modulation, bounds, cfg)
    fc = carrier_after_demodulation(x, optimized["bandwidth_hz"], modulation, optimized["phi_rad"], cfg)
    return {"method": label, "success": True, "fc_hz": fc, "modulation_hz": modulation,
            **optimized, "coarse": coarse, "runtime_ms": (time.perf_counter() - start) * 1000}, (image if method == "proposed" else None)
