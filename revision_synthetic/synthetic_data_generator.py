import os

import numpy as np
import pandas as pd


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)


CONDITIONS = {
    "s0_no_period": dict(period_strength=0.0, noise=0.30, trend=0.0),
    "s1_weak_period": dict(period_strength=0.2, noise=0.30, trend=0.0),
    "s2_mid_period": dict(period_strength=0.5, noise=0.30, trend=0.0),
    "s3_strong_period": dict(period_strength=0.8, noise=0.30, trend=0.0),
    "s4_strong_period_high_noise": dict(period_strength=0.8, noise=0.60, trend=0.0),
    "s5_strong_period_low_noise": dict(period_strength=0.8, noise=0.10, trend=0.0),
}


def generate_series(n, period, strength, noise, trend, channel_idx, seed):
    rng = np.random.default_rng(seed + channel_idx)
    t = np.arange(n)
    seasonal = strength * np.sin(2 * np.pi * t / period + channel_idx * 0.7)
    trend_component = trend * t
    eps = rng.normal(0, noise, size=n)
    return seasonal + trend_component + eps


def main():
    n = 20000
    channels = 6
    period = 24
    for tag, cfg in CONDITIONS.items():
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        data = {"date": dates}
        for c in range(1, channels + 1):
            data[f"x{c}"] = generate_series(
                n, period, cfg["period_strength"], cfg["noise"], cfg["trend"], c, seed=2026
            )
        df = pd.DataFrame(data)
        path = os.path.join(OUT_DIR, f"synthetic_{tag}.csv")
        df.to_csv(path, index=False)
        print("saved", path, "| period_strength=", cfg["period_strength"], "| noise=", cfg["noise"])


if __name__ == "__main__":
    main()
