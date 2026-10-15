"""Synthetic triangular-FM I/Q generation with reproducible noise and mixtures."""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

AMP_GAP_BIN_WIDTH_DB = 1.0

AMP_GAP_MIN_DB = 0.0

AMP_GAP_MAX_DB = 10.0

FS = 67.0015e6

DURATION_SEC = 0.002

FREQ_LIMITS = (-8e6, 8e6)

ALPHA_LIMITS = (5e3, 120e3)

BW_LIMITS = (5e6, 11e6)

N_COMPONENTS_RANGE = (1, 3)

NBI_PROB = 0.30

SNR_MIN = 0.0

SNR_MAX = 10.0

EPS = 1e-12

def make_gap_bins() -> List[Tuple[float, float]]:
    """Return amplitude-gap bins as [low, high) dB intervals."""
    bins: List[Tuple[float, float]] = []
    x = float(AMP_GAP_MIN_DB)
    while x < float(AMP_GAP_MAX_DB) - 1e-12:
        bins.append((x, min(x + float(AMP_GAP_BIN_WIDTH_DB), float(AMP_GAP_MAX_DB))))
        x += float(AMP_GAP_BIN_WIDTH_DB)
    return bins

def quantize_gap_bin(gap_db: float) -> str:
    """Convert an actual amplitude/pseudo gap to the configured bin label."""
    if not np.isfinite(gap_db):
        return "undefined"
    for low, high in make_gap_bins():
        if low <= gap_db < high or (math.isclose(gap_db, AMP_GAP_MAX_DB) and math.isclose(high, AMP_GAP_MAX_DB)):
            return f"{low:.0f}-{high:.0f} dB"
    return "out_of_range"

def triangular_wave(t: np.ndarray, alpha_hz: float, time_delay: float) -> np.ndarray:
    """Triangular waveform in [-1, 1]."""
    return (2.0 / np.pi) * np.arcsin(np.sin(2.0 * np.pi * alpha_hz * (t - time_delay)))

def generate_tfm_signal(fs: float, n_samples: int, params: Dict[str, float]) -> np.ndarray:
    """Generate one triangular frequency modulation (TFM) component."""
    t = np.arange(n_samples) / fs
    tri = triangular_wave(t, params["alpha"], params["time_delay"])
    instantaneous_freq = params["f_base"] + tri * params["f_range"]
    phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / fs
    return params["amp"] * np.exp(1j * (phase + params["phase_offset"]))

def generate_synthetic_iq_sample(
    sample_id: str,
    case: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Generate one noisy multi-component synthetic I/Q signal and exact GT metadata."""
    n_samples = int(round(FS * DURATION_SEC))
    if case is None:
        num_signals = random.randint(N_COMPONENTS_RANGE[0], N_COMPONENTS_RANGE[1])
        snr_db = random.uniform(SNR_MIN, SNR_MAX)
    else:
        num_signals = int(case["num_signals"])
        if num_signals == 1:
            # For the single-emitter regime, force the requested pseudo gap by
            # setting SNR to the selected bin. Because ref_amp=1,
            # pseudo_gap_db = 20*log10(A_signal/sigma_disturbance) = snr_db.
            snr_db = random.uniform(float(case["gap_low_db"]), float(case["gap_high_db"]))
        else:
            # For multi-emitter regimes, keep SNR independent from the
            # dominant-to-secondary amplitude gap.
            snr_db = random.uniform(SNR_MIN, SNR_MAX)

    mixed_signal = np.zeros(n_samples, dtype=np.complex128)
    signal_params: List[Dict[str, Any]] = []
    ref_amp = 1.0

    freq_min, freq_max = map(float, FREQ_LIMITS)

    for i in range(num_signals):
        bw = random.uniform(*BW_LIMITS)        # full sweep bandwidth
        f_range = 0.5 * bw                    # half sweep span
        center_low = freq_min + f_range
        center_high = freq_max - f_range
        if center_high <= center_low:
            raise ValueError(
                f"No valid center-frequency interval. FREQ_LIMITS={FREQ_LIMITS}, bw={bw:.3e} Hz."
            )

        if i > 0 and random.random() < 0.5:
            prev = signal_params[-1]
            overlap_ratio = random.uniform(0.2, 0.5)
            offset = (1.0 - overlap_ratio) * (f_range + prev["f_range"])
            direction = 1.0 if random.random() < 0.5 else -1.0
            candidate = prev["f_base"] + direction * offset
            if not (center_low <= candidate <= center_high):
                candidate_alt = prev["f_base"] - direction * offset
                if center_low <= candidate_alt <= center_high:
                    candidate = candidate_alt
            f_base = float(np.clip(candidate, center_low, center_high))
        else:
            f_base = random.uniform(center_low, center_high)

        if i == 0:
            # The first generated component is intentionally the dominant component.
            amp = ref_amp
            rel_amp_db = 0.0
        else:
            if case is not None and num_signals >= 2:
                # Force the strongest secondary component to create the requested
                # dominant-to-secondary amplitude gap. Additional secondaries are
                # made weaker so that the requested gap is preserved.
                gap_low = float(case["gap_low_db"])
                gap_high = float(case["gap_high_db"])
                target_gap_db = random.uniform(gap_low, gap_high)
                if i == 1:
                    rel_amp_db = -target_gap_db
                else:
                    extra_gap_db = random.uniform(0.5, 5.0)
                    rel_amp_db = -min(target_gap_db + extra_gap_db, AMP_GAP_MAX_DB + 5.0)
            else:
                # Legacy random generation: secondary components are 0--10 dB below the dominant.
                rel_amp_db = -random.uniform(AMP_GAP_MIN_DB, AMP_GAP_MAX_DB)
            amp = ref_amp * (10.0 ** (rel_amp_db / 20.0))

        params = {
            "component_id": f"{sample_id}_{i}",
            "component_index": i,
            "amp": float(amp),
            "rel_amp_db": float(rel_amp_db),
            "f_base": float(f_base),
            "f_range": float(f_range),
            "fspan": float(2.0 * f_range),
            "alpha": float(random.uniform(*ALPHA_LIMITS)),
            "time_delay": float(random.uniform(0.0, 100e-6)),
            "phase_offset": float(random.uniform(0.0, 2.0 * np.pi)),
        }
        mixed_signal += generate_tfm_signal(FS, n_samples, params)
        signal_params.append(params)

    # Noise + narrowband interference (NBI).
    ref_sig_power = ref_amp ** 2
    total_disturbance_power = ref_sig_power / (10.0 ** (snr_db / 10.0))

    interference = np.zeros(n_samples, dtype=np.complex128)
    has_nbi = 0
    nbi_freq = np.nan
    nbi_power = 0.0
    if random.random() < NBI_PROB:
        has_nbi = 1
        nbi_freq = random.uniform(FREQ_LIMITS[0], FREQ_LIMITS[1])
        nbi_ratio = random.uniform(0.01, 0.05)
        target_interf_power = total_disturbance_power * nbi_ratio
        interf_amp = math.sqrt(target_interf_power)
        t = np.arange(n_samples) / FS
        interference = interf_amp * np.exp(1j * (2.0 * np.pi * nbi_freq * t + random.uniform(0.0, 2.0 * np.pi)))
        nbi_power = float(np.mean(np.abs(interference) ** 2))

    remaining_noise_power = max(total_disturbance_power - nbi_power, 1e-15)
    noise = math.sqrt(remaining_noise_power / 2.0) * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    )

    final_signal = (mixed_signal + interference + noise).astype(np.complex64)

    amps = np.array([p["amp"] for p in signal_params], dtype=float)
    dom_idx = int(np.argmax(amps))
    dom = signal_params[dom_idx]
    sorted_amps = np.sort(amps)[::-1]
    amp_gap_db = np.nan if len(sorted_amps) < 2 else float(20.0 * np.log10((sorted_amps[0] + EPS) / (sorted_amps[1] + EPS)))

    # Single-emitter pseudo gap: treat the disturbance as a virtual second source.
    # Since disturbance is complex circular noise plus optional NBI, use RMS amplitude
    # sqrt(P_disturbance). With ref_amp=1, this equals the generated SNR in dB.
    disturbance_power = float(remaining_noise_power + nbi_power)
    disturbance_rms_amp = math.sqrt(max(disturbance_power, EPS))
    pseudo_gap_db = float(20.0 * np.log10((float(dom["amp"]) + EPS) / (disturbance_rms_amp + EPS)))

    if num_signals == 1:
        operating_gap_db = pseudo_gap_db
        operating_gap_definition = "signal_to_disturbance_pseudo_gap"
    else:
        operating_gap_db = amp_gap_db
        operating_gap_definition = "dominant_to_secondary_amplitude_gap"

    sample_meta = {
        "sample_id": sample_id,
        "snr_db": float(snr_db),
        "num_signals": int(num_signals),
        "has_nbi": int(has_nbi),
        "nbi_freq_hz": float(nbi_freq) if np.isfinite(nbi_freq) else np.nan,
        "nbi_power": float(nbi_power),
        "dominant_component_index": int(dom_idx),
        "amp_gap_db": amp_gap_db,
        "pseudo_gap_db": pseudo_gap_db,
        "operating_gap_db": operating_gap_db,
        "operating_gap_bin": quantize_gap_bin(operating_gap_db),
        "operating_gap_definition": operating_gap_definition,
        "amp_gap_bin": quantize_gap_bin(amp_gap_db),
        "target_case_id": str(case["case_id"]) if case is not None else "random",
        "target_amp_gap_bin": str(case["amp_gap_bin"]) if case is not None else quantize_gap_bin(operating_gap_db),
        "target_gap_low_db": float(case["gap_low_db"]) if case is not None and np.isfinite(case["gap_low_db"]) else np.nan,
        "target_gap_high_db": float(case["gap_high_db"]) if case is not None and np.isfinite(case["gap_high_db"]) else np.nan,
        "fc_D_gt": float(dom["f_base"]),
        "fspan_D_gt": float(dom["fspan"]),
        "frate_gt": float(dom["alpha"]),
    }
    return final_signal, signal_params, sample_meta
