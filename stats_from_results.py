#!/usr/bin/env python3
"""Paired statistics for PatchTST vs PatchTST_FreqEmbed from result files.

Reads the time-series-library result format:

    long_term_forecast_<dataset>_<sl>_<pl>_<mid>_<model>_..._0
    mse:0.1234, mae:0.2345, dtw:Not calculated

and reports mean/std, mean difference, paired t-test and Wilcoxon signed-rank
test for every dataset/horizon with at least two paired seeds.
"""

import argparse
import re
from collections import defaultdict

import numpy as np
from scipy import stats


ENTRY_RE = re.compile(
    r"long_term_forecast_(?P<dataset>[A-Za-z0-9]+)_"
    r"(?P<sl>\d+)_(?P<pl>\d+)_(?P<mid>\S+?)_"
    r"(?P<model>PatchTST(?:_FreqEmbed)?)_[^\n]*?_0\s+"
    r"mse:(?P<mse>[0-9.eE+-]+),\s*mae:(?P<mae>[0-9.eE+-]+)"
)


def parse_result_file(path):
    """Return {model: {key: metric}} with keys (dataset, sl, pl, seed)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    pairs = {}
    for m in ENTRY_RE.finditer(text):
        model = m.group("model")
        if model != "PatchTST_FreqEmbed" and model != "PatchTST":
            continue
        mid = m.group("mid")
        seed = None
        seed_m = re.search(r"_s(\d+)", mid)
        if seed_m:
            seed = int(seed_m.group(1))
        elif re.search(r"freqembed(?:_|$)", mid):
            seed = 2021
        if seed is None:
            continue
        key = (m.group("dataset"), int(m.group("sl")), int(m.group("pl")), seed)
        metric = (float(m.group("mse")), float(m.group("mae")))
        pairs.setdefault(model, {})[key] = metric
    return pairs


def wilcoxon_p(diffs):
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 1.0
    return float(stats.wilcoxon(nonzero).pvalue)


def fmt_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", nargs="?", default="result_long_term_forecast.txt")
    parser.add_argument("--out", default="stats_summary.md")
    args = parser.parse_args()

    pairs = parse_result_file(args.result_file)
    base = pairs.get("PatchTST", {})
    freq = pairs.get("PatchTST_FreqEmbed", {})
    shared = sorted(set(base) & set(freq), key=lambda k: (k[0], k[2], k[3]))

    groups = defaultdict(list)
    for key in shared:
        groups[(key[0], key[1], key[2])].append(key)

    lines = [
        "# Paired statistics: PatchTST vs PatchTST_FreqEmbed (MSE)",
        "",
        "| Dataset | pred_len | seeds | Base mean | Freq mean | dMSE | t-test p | Wilcoxon p | wins |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for group_key in sorted(groups):
        dataset, sl, pl = group_key
        keys = sorted(groups[group_key], key=lambda k: k[3])
        base_mse = np.array([base[k][0] for k in keys])
        freq_mse = np.array([freq[k][0] for k in keys])
        diff = freq_mse - base_mse
        n = len(keys)
        wins = int((diff < 0).sum())
        t_p = float(stats.ttest_rel(freq_mse, base_mse).pvalue)
        w_p = wilcoxon_p(diff)
        lines.append(
            "| {} | {} | {} | {:.5f} | {:.5f} | {:.5f} | {} | {} | {}/{} |".format(
                dataset,
                pl,
                n,
                base_mse.mean(),
                freq_mse.mean(),
                diff.mean(),
                fmt_p(t_p),
                fmt_p(w_p),
                wins,
                n,
            )
        )

    lines += [
        "",
        "Notes:",
        "- dMSE = FreqEmbed - PatchTST; negative means FreqEmbed is better.",
        "- Wins counts seeds where FreqEmbed has lower MSE.",
        "- With n=5, a two-sided Wilcoxon p cannot be below 0.05;",
        "  n>=6 is required for a possible Wilcoxon significance star.",
        "- Statistical significance is secondary to reporting direction",
        "  consistency and mean effect size honestly.",
        "",
    ]

    report = "\n".join(lines)
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
