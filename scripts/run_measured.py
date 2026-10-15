#!/usr/bin/env python3
"""Run measured parameter estimation and demodulation from any working directory."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssc_tfm.cli import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli("measured"))
