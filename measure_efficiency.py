"""Parameter counts and inference latency for PatchTST vs FreqEmbed."""
import argparse
import time

import torch

from models.PatchTST import Model as PatchTST
from models.PatchTST_FreqEmbed import Model as FreqEmbed


class Config:
    pass


def make_config(d_model):
    c = Config()
    c.task_name = "long_term_forecast"
    c.seq_len = 96
    c.pred_len = 96
    c.d_model = d_model
    c.d_ff = d_model * 4
    c.n_heads = 8
    c.e_layers = 2
    c.factor = 3
    c.dropout = 0.1
    c.activation = "gelu"
    c.enc_in = 7
    c.dec_in = 7
    c.c_out = 7
    return c


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, device, batch=64, iters=10):
    model.eval().to(device)
    x = torch.randn(batch, 96, 7, device=device)
    x_mark = torch.zeros(batch, 96, 4, device=device)
    dec = torch.zeros(batch, 96, 7, device=device)
    dec_mark = torch.zeros(batch, 96, 4, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x, x_mark, dec, dec_mark)
        start = time.time()
        for _ in range(iters):
            model(x, x_mark, dec, dec_mark)
        elapsed = time.time() - start
    return elapsed / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    results = []
    for name, cls in [("PatchTST", PatchTST), ("PatchTST_FreqEmbed", FreqEmbed)]:
        model = cls(make_config(args.d_model))
        params = count_params(model)
        latency = measure_latency(model, device, batch=args.batch, iters=args.iters)
        results.append((name, params, latency))
        print("{} params={} avg_infer_ms={:.3f}".format(
            name, params, latency * 1000
        ))

    p0, p1 = results[0][1], results[1][1]
    print("param_delta={:+.4f}%".format((p1 - p0) / p0 * 100))
    t0, t1 = results[0][2], results[1][2]
    print("latency_delta={:+.2f}%".format((t1 - t0) / t0 * 100))


if __name__ == "__main__":
    main()
