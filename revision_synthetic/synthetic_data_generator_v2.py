import os

import numpy as np
import pandas as pd


OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "synthetic_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# Each condition controls multi-period strength, noise, trend, and amplitude modulation.
CONDITIONS = {
    "c0_no_period": dict(amp=0.00, noise=0.40, trend=0.0, periods=[24, 8, 6], modulated=False),
    "c1_weak_period": dict(amp=0.15, noise=0.35, trend=0.00005, periods=[24, 8, 6], modulated=False),
    "c2_strong_period": dict(amp=0.70, noise=0.10, trend=0.00005, periods=[24, 8, 6], modulated=False),
    "c3_strong_period_high_noise": dict(amp=0.70, noise=0.60, trend=0.00005, periods=[24, 8, 6], modulated=False),
    "c4_amplitude_modulated": dict(amp=0.60, noise=0.25, trend=0.00005, periods=[24, 8, 6], modulated=True),
    "c5_multi_short_period": dict(amp=0.55, noise=0.20, trend=0.0, periods=[8, 6, 4], modulated=False),
}


def generate_series(n, cfg, channel_idx, seed):
    rng = np.random.default_rng(seed + channel_idx)
    t = np.arange(n).astype(float)
    trend = cfg["trend"] * t
    signal = np.zeros(n)
    for j, period in enumerate(cfg["periods"]):
        amp = cfg["amp"] / (j + 1)
        if cfg["modulated"]:
            envelope = 0.55 + 0.45 * np.sin(2 * np.pi * t / 168 + channel_idx)
            amp = amp * envelope
        signal += amp * np.sin(2 * np.pi * t / period + channel_idx * 0.9 + j * 1.1)
    noise = rng.normal(0, cfg["noise"], size=n)
    return trend + signal + noise


def main():
    n = 20000
    channels = 6
    for tag, cfg in CONDITIONS.items():
        dates = pd.date_range("2020-01-01", periods=n, freq="h")
        data = {"date": dates}
        for c in range(1, channels + 1):
            data[f"x{c}"] = generate_series(n, cfg, c, seed=2026)
        df = pd.DataFrame(data)
        path = os.path.join(OUT_DIR, f"synthetic_v2_{tag}.csv")
        df.to_csv(path, index=False)
        print("saved", path, "| amp=", cfg["amp"], "| noise=", cfg["noise"], "| periods=", cfg["periods"])


if __name__ == "__main__":
    main()
