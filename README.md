# Frequency-Gated PatchTST

This repository contains the code for the paper "Lightweight Frequency Augmentation with Learnable Gating for Patch-Based Long-Term Time Series Forecasting".

## Main models

- `models/PatchTST.py`: baseline PatchTST.
- `models/PatchTST_FreqEmbed.py`: proposed learnable scalar gate, initialized to zero.
- `models/PatchTST_FreqEmbedFixed1.py`: fixed unit gate.
- `models/PatchTST_FreqOnly.py`: frequency-only representation.
- `models/PatchTST_RandomFreq.py`: random-frequency control.

## Installation

```bash
pip install -r requirements.txt
```

## Synthetic mechanism experiments

The controlled synthetic experiments are in `revision_synthetic/`.

```bash
cd revision_synthetic
bash run_synthetic_v2.sh
```

Seeds are listed in `revision_synthetic/seed_list.txt`, and the data conditions are described in `revision_synthetic/SNR_levels.md`.

## Benchmark experiments

The main benchmark commands follow the Time-Series-Library protocol. Example scripts are under `scripts/long_term_forecast/`.
