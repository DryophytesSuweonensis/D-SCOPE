#!/usr/bin/env python3
"""Generate reproducible raw synthetic triangular-FM I/Q."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ssc_tfm import generation as gen


def clean(value):
    if isinstance(value, dict):
        return {key: clean(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(val) for val in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_record(emitters, seed, gap_low, gap_high, sample_id):
    random.seed(seed)
    np.random.seed(seed)
    case = {
        "num_signals": emitters,
        "gap_low_db": gap_low,
        "gap_high_db": gap_high,
        "case_id": f"em{emitters}_gap{gap_low:g}_{gap_high:g}dB",
        "amp_gap_bin": f"{gap_low:g}-{gap_high:g} dB",
    }
    iq, components, sample = gen.generate_synthetic_iq_sample(sample_id, case)
    dominant = components[int(sample["dominant_component_index"])]
    component_keys = ("component_index", "amp", "rel_amp_db", "f_base", "f_range",
                      "fspan", "alpha", "time_delay", "phase_offset")
    public_components = [
        {**{key: component[key] for key in component_keys},
         "is_dominant": int(index == sample["dominant_component_index"])}
        for index, component in enumerate(components)
    ]
    metadata = {
        "kind": "synthetic",
        "sample_id": sample_id,
        "n_samples": len(iq),
        "fs_hz": gen.FS,
        "num_emitters": emitters,
        "num_signals": emitters,
        "sample_seed": seed,
        "generation": {"seed": seed, "gap_low_db": gap_low, "gap_high_db": gap_high,
                       "duration_s": gen.DURATION_SEC},
        "input_stage": "raw_iq",
        "snr_db": sample["snr_db"],
        "map_style": "revision",
        "alpha_min_hz": 1000.0,
        "alpha_max_hz": 128000.0,
        "alpha_step_hz": 1000.0,
        "max_tau": 1920,
        "alpha_upscale": 30,
        "median_kernel": 9,
        "iq_sha256": hashlib.sha256(iq.tobytes()).hexdigest(),
        "components": public_components,
        "dominant_truth": {
            "fc_hz": dominant["f_base"],
            "bandwidth_hz": dominant["fspan"],
            "modulation_hz": dominant["alpha"],
        },
    }
    return iq, clean(metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["bundled", "new"], default="bundled",
                        help="bundled replays the seeds of the three curated examples")
    parser.add_argument("--emitters", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--gap-low-db", type=float, default=8.0)
    parser.add_argument("--gap-high-db", type=float, default=9.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/generated_synthetic")
    args = parser.parse_args()
    if not (0 <= args.gap_low_db < args.gap_high_db <= 10):
        parser.error("gap bounds must satisfy 0 <= low < high <= 10 dB")

    recipes = []
    if args.recipe == "bundled":
        for count in (1, 2, 3):
            source = ROOT / f"data/synthetic/emitter_{count}.npz"
            with np.load(source, allow_pickle=False) as record:
                metadata = json.loads(str(record["metadata_json"].item()))
            recipe = metadata["generation"]
            recipes.append((count, int(recipe["seed"]),
                            float(recipe["gap_low_db"]),
                            float(recipe["gap_high_db"]), metadata["sample_id"]))
    else:
        recipes.append((args.emitters, args.seed, args.gap_low_db,
                        args.gap_high_db, f"generated_emitter_{args.emitters}"))

    args.output.mkdir(parents=True, exist_ok=True)
    for count, seed, low, high, sample_id in recipes:
        path = args.output / f"emitter_{count}.npz"
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}; choose a fresh --output directory")
        iq, metadata = make_record(count, seed, low, high, sample_id)
        np.savez_compressed(path, iq=iq, fs_hz=np.float64(gen.FS),
                            metadata_json=json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        print(f"Saved {path}: {len(iq)} samples, {count} emitters, seed={seed}")


if __name__ == "__main__":
    main()
