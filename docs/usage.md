# Usage Guide

Measured signals are identified only as **A, B and C**. Hardware names,
original capture filenames and paths, source-code correspondence tables,
and historical experiment records are excluded from this release.

## Data and processing stages

`data/synthetic/` contains one raw I/Q example for each of 1, 2 and 3 emitters.
`data/measured/` contains **50,000 raw complex samples** for each of A, B and C.
All input records are stored **before software filtering, phase normalization,
IF median filtering and demodulation**.

The measured-data runner first applies a 30 MHz raised-cosine filter.
The proposed detector then uses phase-only I/Q → instantaneous frequency (IF)
→ median filtering → phase reconstruction → alpha–frequency mapping.
Demodulation operates on the band-filtered I/Q. Synthetic inputs do not use
this front-end filter.

NPZ files contain raw I/Q; PNG files contain alpha–frequency images derived
from that I/Q.

The following PNGs are provided for each of A, B and C:

- `monitor_X_alpha_f_before_median.png`: IF median filtering is skipped.
- `monitor_X_alpha_f_after_median.png`: IF median filtering uses kernel 9;
  this image is also the detector cache.

Both comparison branches use the same 30 MHz front-end filter, phase-only
normalization, IF calculation and phase reconstruction. Only the IF median
step differs. The IF calculation gives both comparison signals a length of
49,999 samples. The 50,000 raw samples in each NPZ remain unchanged.
Image intensities are normalized independently within each PNG.

[View the before/after comparison for A, B and C](assets/measured_median_comparison.png)

To regenerate the PNG pairs:

```bash
python scripts/export_measured_median_images.py --case all --device cpu
```

## Installation and execution

Use Python 3.11 and run these commands from the release folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/run_synthetic.py --case all --device cpu
python scripts/run_measured.py --case all --device cpu
python scripts/check_results.py --synthetic outputs/synthetic --measured outputs/measured
```

`--method baseline` selects IF-FFT-CZT-PSO. The default, `proposed`, selects
RT-DETR+CZT+PSO; `both` runs both methods. Use `--case` to select one record
and `--recompute-image` to rebuild its detector image from raw I/Q.
For GPU execution, install compatible CUDA-enabled PyTorch wheels and use
`--device 0`.

## Results and validation

`outputs/synthetic/` and `outputs/measured/` contain summary CSV files and
case/method subdirectories with parameter JSON, demodulated I/Q NPZ,
spectrum NPZ/CSV, and PNG/PDF figures. `expected_results/` contains anonymous
validation outputs.

Synthetic checks compare the dominant-emitter estimates with ground truth.
The code does not estimate every emitter simultaneously. Measured records
have no independent parameter ground truth, so their checks verify execution,
finite estimates, sample counts, and increased target-band power after
demodulation.

## Regenerate synthetic data

```bash
python scripts/generate_synthetic.py --recipe bundled
```

See the [README](../README.md) for new-data generation and additional options.
Arbitrary new records are not guaranteed to pass the accuracy thresholds.

## Upload to GitHub

Place this folder's contents at the repository root. The inference checkpoint
is about 22 MB. Use the Git commands in the README to upload this release;
the original working directory is not required.
