"""
Bootstrap CI + paired tests for all 4 tiers.
Reads per-sample score CSVs, outputs means with 95% bootstrap CIs
and pairwise comparisons between key model pairs.
"""
import os
import numpy as np
import pandas as pd

RESULTS = r'c:\Users\leeze\Documents\GitHub\diffuclinic\results'
K = 1000
ALPHA = 0.05

MODEL_FILES = {
    'LLaMA ZS':      ('llama_test', 'safety', 'quality', 'pdsqi'),
    'LLaMA LoRA':    ('llama_lora', 'safety', 'quality', 'pdsqi'),
    'LLaDA ZS':      ('llada_test', 'safety', 'quality', 'pdsqi'),
    'LLaDA LoRA':    ('llada_lora', 'safety', 'quality', 'pdsqi'),
    'LLaDA EPD':     ('llada_epd', 'safety', 'quality', 'pdsqi'),
    'LAD EPD':       ('lad_epd', 'safety', 'quality', 'pdsqi'),
    'LAD LoRA':      ('lad_lora', 'safety', 'quality', 'pdsqi'),
}

# tini_test has no o3 scores, use R1 only for now — actually skip, it's #6

def load_scores(base, tier):
    """Load per-sample scores for a model."""
    if tier == 'safety':
        f = os.path.join(RESULTS, f'safety_scores_{base}.csv')
        df = pd.read_csv(f)
        return df.drop(columns=['id'])
    elif tier == 'quality':
        f = os.path.join(RESULTS, f'quality_scores_{base}.csv')
        df = pd.read_csv(f)
        return df.drop(columns=['id'])
    elif tier == 'pdsqi':
        scores = {}
        for judge in ['o3', 'R1']:
            if base == 'tini_test' and judge == 'o3':
                continue
            f = os.path.join(RESULTS, f'pdsqi_scores_{base}_{judge}.csv')
            df = pd.read_csv(f)
            dims = ['Accuracy', 'Completeness', 'Organization', 'Synthesis', 'Overall']
            for d in dims:
                scores[f'{d} ({judge})'] = df[d].values
        return pd.DataFrame(scores)
    return None


def bootstrap_ci(values, k=K, alpha=ALPHA):
    """Return (mean, lower, upper) via bootstrap percentile."""
    n = len(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(k):
        idx = rng.choice(n, n, replace=True)
        means.append(values[idx].mean())
    means = np.array(means)
    return float(np.mean(values)), float(np.percentile(means, alpha/2*100)), float(np.percentile(means, (1-alpha/2)*100))


def paired_bootstrap_test(a, b, k=K, alpha=ALPHA):
    """Paired bootstrap test: H0: mean(a) == mean(b). Returns (delta, lower, upper, significant)."""
    n = len(a)
    rng = np.random.RandomState(42)
    deltas = []
    for _ in range(k):
        idx = rng.choice(n, n, replace=True)
        deltas.append(a[idx].mean() - b[idx].mean())
    deltas = np.array(deltas)
    delta = a.mean() - b.mean()
    lo = np.percentile(deltas, alpha/2*100)
    hi = np.percentile(deltas, (1-alpha/2)*100)
    sig = 'ns' if (lo <= 0 <= hi) else ('*' if abs(delta) > 0 else '*')
    return delta, lo, hi, sig


print("=" * 90)
print("BOOTSTRAP CI (95%, k=1000) — All Models × All Metrics")
print("=" * 90)

all_results = {}

for model_name, (base, _, _, _) in MODEL_FILES.items():
    print(f"\n{'─'*80}")
    print(f"  {model_name}")
    print(f"{'─'*80}")
    model_results = {}

    for tier, tier_name in [('safety', 'Tier 1'), ('quality', 'Tier 2'), ('pdsqi', 'Tier 3')]:
        scores = load_scores(base, tier)
        if scores is None:
            continue
        for col in scores.columns:
            vals = scores[col].dropna().values
            if len(vals) == 0:
                continue
            mean, lo, hi = bootstrap_ci(vals)
            model_results[col] = (mean, lo, hi)
            star = ''
            print(f"  {col:<22s}  {mean:8.4f}  [{lo:8.4f}, {hi:8.4f}]{star}")

    all_results[model_name] = model_results

print("\n\n" + "=" * 90)
print("PAIRED BOOTSTRAP TESTS (k=1000, α=0.05)")
print("=" * 90)

pairs = [
    ('LLaMA ZS', 'LLaDA ZS', 'ZS: AR vs Diffusion'),
    ('LLaMA LoRA', 'LLaDA LoRA', 'LoRA: AR vs Diffusion'),
    ('LLaDA ZS', 'LLaDA LoRA', 'LLaDA: ZS vs LoRA'),
    ('LLaMA ZS', 'LLaMA LoRA', 'LLaMA: ZS vs LoRA'),
    ('LLaDA LoRA', 'LLaDA EPD', 'LLaDA: LoRA vs EPD'),
]

key_metrics = ['SummaC', 'QAFactEval', 'MedNER-F1',
               'ROUGE-L', 'BLEU', 'METEOR', 'BERTScore-F1',
               'Accuracy (o3)', 'Completeness (o3)', 'Organization (o3)',
               'Synthesis (o3)', 'Overall (o3)']

for ma, mb, label in pairs:
    print(f"\n{'─'*80}")
    print(f"  {label}  ({ma} vs {mb})")
    print(f"{'─'*80}")
    for metric in key_metrics:
        if metric in all_results.get(ma, {}) and metric in all_results.get(mb, {}):
            # Load raw values
            # Find which tier this metric belongs to
            for tier, tier_name in [('safety', 'Tier 1'), ('quality', 'Tier 2'), ('pdsqi', 'Tier 3')]:
                sa = load_scores(MODEL_FILES[ma][0], tier)
                sb = load_scores(MODEL_FILES[mb][0], tier)
                if sa is not None and metric in sa.columns:
                    a_vals = sa[metric].dropna().values
                    b_vals = sb[metric].dropna().values
                    delta, lo, hi, sig = paired_bootstrap_test(a_vals, b_vals)
                    sig_str = '***' if (lo > 0 or hi < 0) and abs(delta) > 0.1 else ('*' if lo > 0 or hi < 0 else 'ns')
                    print(f"  {metric:<22s}  Δ={delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig_str}")
                    break

print("\nDone.")
