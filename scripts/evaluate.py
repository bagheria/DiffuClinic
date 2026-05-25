"""
DiffuClinic Evaluation — Risk-Based Framework (Safety > Quality > Clinical Utility > Efficiency)

Currently implements Tier 2 (Quality). Tiers 1, 3, 4 to be added.

Usage:
    python scripts/evaluate.py --results_dir ./results --device cuda
    python scripts/evaluate.py --results_dir ./results --model llama_test --no-bertscore
"""
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ── Model registry ──────────────────────────────────────────
MODEL_FILES = {
    "llada_test": "llada_test.csv",
    "llada_lora": "llada_lora.csv",
    "llama_test": "llama_test.csv",
    "llama_lora": "llama_lora.csv",
}

MODEL_DISPLAY = {
    "llada_test": "LLaDA zero-shot (8B-Instruct)",
    "llada_lora": "LLaDA LoRA (8B-Base + clinical LoRA)",
    "llama_test": "LLaMA zero-shot (Llama-3.1-8B-Instruct)",
    "llama_lora": "LLaMA LoRA (Llama-3.1-8B + clinical LoRA)",
}


class QualityEvaluator:
    """ROUGE / BLEU / METEOR / BERTScore — ported from Evaulation/eva.ipynb."""

    def __init__(self, device="cuda", bert_batch_size=8):
        self.device = device

        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )

        print(f"Loading BERTScore (roberta-large) on {device}...")
        self.bertscorer = BERTScorer(
            lang="en",
            model_type="roberta-large",
            rescale_with_baseline=True,
            device=device,
            batch_size=bert_batch_size,
        )
        self.smoother = SmoothingFunction().method1

    def compute_rouge(self, pred, ref):
        scores = self.rouge_scorer.score(str(ref), str(pred))
        return {
            "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
            "ROUGE-2": round(scores['rouge2'].fmeasure, 4),
            "ROUGE-L": round(scores['rougeL'].fmeasure, 4),
        }

    def compute_bleu(self, pred, ref):
        pred_tokens = str(pred).split()
        ref_tokens = str(ref).split()
        results = {}
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0 / n] * n)
            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=weights, smoothing_function=self.smoother)
            results[f"BLEU-{n}"] = round(bleu, 4)
        bleu_avg = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoother)
        results["BLEU"] = round(bleu_avg, 4)
        return results

    def compute_meteor(self, pred, ref):
        pred_tokens = str(pred).split()
        ref_tokens = str(ref).split()
        return {"METEOR": round(meteor_score([ref_tokens], pred_tokens), 4)}

    def compute_bertscore_batch(self, preds, refs):
        P, R, F1 = self.bertscorer.score(preds, refs)
        return [round(s, 4) for s in F1.tolist()]

    def evaluate(self, preds, refs, desc="Evaluating"):
        results = []
        for pred, ref in tqdm(zip(preds, refs), total=len(preds), desc=f"{desc} (ROUGE/BLEU/METEOR)"):
            row = {}
            row.update(self.compute_rouge(pred, ref))
            row.update(self.compute_bleu(pred, ref))
            row.update(self.compute_meteor(pred, ref))
            results.append(row)

        df = pd.DataFrame(results)
        print(f"{desc}: computing BERTScore...")
        df["BERTScore-F1"] = self.compute_bertscore_batch(preds, refs)
        return df


def load_model_data(results_dir, model_name):
    """Load a model's prediction CSV, return (ids, predictions, references)."""
    filename = MODEL_FILES[model_name]
    path = os.path.join(results_dir, filename)
    df = pd.read_csv(path)
    preds = df['predicted_summary'].astype(str).tolist()
    refs = df['reference_summary'].astype(str).tolist()
    ids = df['id'].tolist()
    return ids, preds, refs


def main():
    parser = argparse.ArgumentParser(description="DiffuClinic Evaluation")
    parser.add_argument("--results_dir", default="./results", help="Directory with model CSVs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--model", default=None, help="Evaluate a single model (e.g. llada_test)")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (fast mode)")
    parser.add_argument("--bert-batch-size", type=int, default=8, help="BERTScore batch size (lower for smaller GPUs)")
    parser.add_argument("--output_dir", default=None, help="Output directory (defaults to results_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    models_to_run = [args.model] if args.model else list(MODEL_FILES.keys())

    print("=" * 60)
    print("DiffuClinic — Quality Evaluation (Tier 2)")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Device: {args.device} | BERTScore: {'off' if args.no_bertscore else 'on'} | Batch: {args.bert_batch_size}")
    print("=" * 60)

    # ── Init evaluator ──
    if args.no_bertscore:
        evaluator = None  # will compute ROUGE/BLEU/METEOR only
    else:
        evaluator = QualityEvaluator(device=args.device, bert_batch_size=args.bert_batch_size)

    # ── Run each model ──
    all_means = []

    for model_name in models_to_run:
        display = MODEL_DISPLAY.get(model_name, model_name)
        print(f"\n{'─' * 40}")
        print(f"  {display}")
        print(f"{'─' * 40}")

        ids, preds, refs = load_model_data(args.results_dir, model_name)
        print(f"  Samples: {len(preds)}")

        # ROUGE / BLEU / METEOR
        results_rows = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        smoother = SmoothingFunction().method1

        for pred, ref in tqdm(zip(preds, refs), total=len(preds), desc=f"  {model_name} (lexical)"):
            row = {}

            # ROUGE
            scores = scorer.score(str(ref), str(pred))
            row["ROUGE-1"] = round(scores['rouge1'].fmeasure, 4)
            row["ROUGE-2"] = round(scores['rouge2'].fmeasure, 4)
            row["ROUGE-L"] = round(scores['rougeL'].fmeasure, 4)

            # BLEU
            pred_tokens = str(pred).split()
            ref_tokens = str(ref).split()
            for n in [1, 2, 3, 4]:
                w = tuple([1.0 / n] * n)
                row[f"BLEU-{n}"] = round(sentence_bleu([ref_tokens], pred_tokens, weights=w, smoothing_function=smoother), 4)
            row["BLEU"] = round(sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother), 4)

            # METEOR
            row["METEOR"] = round(meteor_score([ref_tokens], pred_tokens), 4)

            results_rows.append(row)

        df = pd.DataFrame(results_rows)

        # BERTScore
        if evaluator is not None:
            print(f"  {model_name}: computing BERTScore...")
            df["BERTScore-F1"] = evaluator.compute_bertscore_batch(preds, refs)

        df.insert(0, 'id', ids)

        # Save per-sample
        out_path = os.path.join(output_dir, f"quality_scores_{model_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        # Aggregate
        metric_cols = [c for c in df.columns if c != 'id']
        mean_row = df[metric_cols].mean().to_dict()
        mean_row['model'] = model_name
        all_means.append(mean_row)

    # ── Summary table ──
    summary = pd.DataFrame(all_means)
    summary = summary.set_index('model').round(4)
    summary.index = [MODEL_DISPLAY.get(m, m) for m in summary.index]

    print("\n" + "=" * 60)
    print("QUALITY SUMMARY — 4 Models")
    print("=" * 60)
    print(summary.to_string())

    summary_path = os.path.join(output_dir, "quality_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
