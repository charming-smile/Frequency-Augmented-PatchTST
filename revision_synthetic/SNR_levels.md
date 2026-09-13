# Synthetic experiment conditions

Fixed settings:

- Timesteps: 20000
- Channels: 6
- Sampling: hourly
- Dominant period: 24 steps
- Prediction length: 96
- Look-back: 96

| Tag | Period strength | Noise std | Trend | Purpose |
|---|---|---|---|---|
| s0_no_period | 0.0 | 0.30 | 0.0 | no periodic structure |
| s1_weak_period | 0.2 | 0.30 | 0.0 | weak periodic structure |
| s2_mid_period | 0.5 | 0.30 | 0.0 | medium periodic structure |
| s3_strong_period | 0.8 | 0.30 | 0.0 | strong periodic structure |
| s4_strong_period_high_noise | 0.8 | 0.60 | 0.0 | strong period + strong noise |
| s5_strong_period_low_noise | 0.8 | 0.10 | 0.0 | strong period + weak noise |

Expected mechanism:

- In s0/s1, fixed alpha=1 should be worse than or similar to PatchTST.
- In s3/s5, learnable gate should be at least as good as PatchTST.
- Across conditions, learnable gate should not show clear negative transfer.
