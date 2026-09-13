# JIIS revision: synthetic mechanism experiments

This folder contains everything needed to run the synthetic experiments for the revision.

## Files

- `synthetic_data_generator.py`: generates six synthetic datasets under different period/noise conditions.
- `SNR_levels.md`: condition table and expected mechanism.
- `run_synthetic_experiments.sh`: runs PatchTST, learnable-gate, and fixed-alpha models for each condition and seed.
- `seed_list.txt`: seeds 1-5.

## Steps

1. Generate the data:

   ```bash
   python synthetic_data_generator.py
   ```

2. Run the experiments:

   ```bash
   bash run_synthetic_experiments.sh
   ```

3. Collect MSE/MAE from the printed results, plus the learned gate value from the FreqEmbed checkpoint.

## Models

- `PatchTST`: baseline
- `PatchTST_FreqEmbed`: learnable scalar gate (alpha initialized to 0)
- `PatchTST_FreqEmbedFixed1`: fixed alpha = 1

Optional supplementary models, if needed:

- `PatchTST_FreqEmbedPos`: alpha constrained to be non-negative
- `PatchTST_FreqOnly`: frequency-only embedding (no time branch)
