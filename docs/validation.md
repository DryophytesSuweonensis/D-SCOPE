# Validation

Validated on **2026-10-15**, Linux, Python 3.11.14, CPU with four to eight PyTorch
threads. Package versions are pinned in `requirements.txt`.

## Data

- Synthetic: three raw complex64 I/Q records, 134,003 samples each, with
  component truth and reproducible generator seeds.
- Measured: raw complex64 I/Q from A/B/C, exactly 50,000 samples each.
  No software filter or detector preprocessing is stored in these raw arrays.
- Six derived detector images are supplied for fast inference.
- Measured A/B/C also have explicit before/after IF median PNG pairs.
  Their input length and phase reconstruction agree; only median filtering
  differs. After-median PNGs serve as the detector caches.
- Anonymous data/configuration hashes and `SHA256SUMS` detect file changes.

The measured runner applies its front-end filter at execution time.
Complex128 FFT arithmetic followed by a complex64 output cast stabilizes
that filtering across NumPy versions. Cache consistency is checked against
provided raw and filtered I/Q hashes.

## Expected behavior

| Synthetic emitters | Center-frequency absolute error (Hz) | Bandwidth absolute error (Hz) | Modulation-rate absolute error (Hz) |
| --- | ---: | ---: | ---: |
| 1 | 48.747 | 3131.660 | 0.0283 |
| 2 | 6.841 | 60.160 | 0.4111 |
| 3 | 54.334 | 1086.287 | 0.2683 |

Both supplied methods pass the bundled synthetic joint accuracy checks:
center frequency and bandwidth ≤500 kHz absolute error, modulation rate
≤0.1% relative error. These checks use dominant-emitter ground truth.

Measured parameter ground truth is unavailable. Both methods complete on
A/B/C and increase target concentration after demodulation. For the proposed
method, target power before → after is A: 2.50% → 85.94%, B: 4.1% → 94.2%,
C: 3.72% → 83.02%. These values use the same final-carrier ±250 kHz interval
before and after. These curated examples do not establish a general ranking
between methods.

`expected_results/` contains anonymous parameter JSON, spectrum PNG/PDF and
demodulated I/Q. Small floating-point differences across environments are
expected; validation uses numerical criteria rather than exact estimate equality.

## Run validation

```bash
python scripts/verify_release.py
python scripts/run_synthetic.py --case all --method both --device cpu --output outputs/check_synthetic
python scripts/run_measured.py --case all --method both --device cpu --output outputs/check_measured
python scripts/check_results.py --synthetic outputs/check_synthetic --measured outputs/check_measured
```

`verify_release.py` checks the packaged files and expected outputs.
`check_results.py` checks newly generated outputs. Both return a nonzero exit
code when a check fails. The provided environment was used for CPU checks;
a fresh dependency installation and CUDA execution were not tested.
