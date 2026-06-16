"""
Power analysis: given pilot results, determine sample size needed for main study.

Usage:
    python power_analysis.py --pilot_dir ./eval_output --output_dir ./eval_output
"""
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot_dir", default="./eval_output")
    parser.add_argument("--output_dir", default="./eval_output")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--power", type=float, default=0.80, help="Statistical power")
    parser.add_argument("--ci_halfwidth", type=float, default=0.04, help="Desired CI half-width")
    args = parser.parse_args()

    from scipy import stats

    z_alpha = stats.norm.ppf(1 - args.alpha / 2)      # e.g., 1.96 for 95%
    z_beta = stats.norm.ppf(args.power)                 # e.g., 0.84 for 80%

    models = ['llama_test', 'llada_test', 'llama_lora', 'llada_lora']

    # Collect pilot stds per metric across all models
    metric_stds = {}

    for model in models:
        try:
            df = pd.read_csv(f"{args.pilot_dir}/safety_scores_{model}.csv")
        except FileNotFoundError:
            continue

        for col in df.columns:
            if col == 'id':
                continue
            std = df[col].std()
            if col not in metric_stds:
                metric_stds[col] = []
            metric_stds[col].append(std)

    print("=" * 70)
    print("PILOT RESULTS — Per-metric standard deviations")
    print("=" * 70)

    for metric, stds in metric_stds.items():
        avg_std = np.mean(stds)
        max_std = np.max(stds)
        print(f"\n{metric}:")
        print(f"  Pilot std (avg across models): {avg_std:.4f}")
        print(f"  Pilot std (max across models): {max_std:.4f}  ← use this (conservative)")

        # Use max std for conservative estimate
        sigma = max_std

        # Per-model CI half-width: need n for desired precision
        n_ci = (z_alpha * sigma / args.ci_halfwidth) ** 2
        print(f"  For CI half-width ±{args.ci_halfwidth} (95%): n ≥ {n_ci:.0f}")

        # Two-model comparison: need n to detect delta
        print(f"  For two-model comparison ({int(args.power*100)}% power):")
        for delta in [0.05, 0.075, 0.10, 0.15]:
            n_delta = 2 * (z_alpha + z_beta) ** 2 * sigma ** 2 / delta ** 2
            print(f"    Detect Δ={delta:.2f}: n ≥ {n_delta:.0f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    # Use the most demanding metric
    max_sigma = max(np.max(s) for s in metric_stds.values())
    worst_metric = [m for m, s in metric_stds.items() if np.max(s) == max_sigma][0]
    n_recommended = int(np.ceil((z_alpha * max_sigma / 0.04) ** 2))
    print(f"Most variable metric: {worst_metric} (std={max_sigma:.4f})")
    print(f"For CI half-width ±0.04 (95%): n ≥ {n_recommended}")
    print(f"Round up → n = {((n_recommended + 24) // 25) * 25}")  # round to nearest 25


if __name__ == "__main__":
    main()
