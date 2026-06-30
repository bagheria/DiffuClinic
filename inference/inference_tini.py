"""
Batch inference for TINI-LAD (diffusion) on MultiClinSum test set.
Outputs CSV with columns: id, reference_summary, predicted_summary

Usage:
    python scripts/inference_tini.py \
        --checkpoint Ruurd/tini_model/diffusion-model-8B.pth \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --data_zip ./data/multiclinsum_test_en.zip \
        --sample_indices ./eval_output/sampled_indices_n175_seed43.txt \
        --output_dir ./results \
        --output_name tini_test
"""
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Suppress IPython display for batch mode ──
import inference.visualize as viz
viz.display = lambda *a, **kw: None
viz.clear_output = lambda *a, **kw: None
viz.HTML = lambda x: x
viz.Markdown = lambda x: x
viz.display_diffusion_output = lambda *a, **kw: None

from models.custom_transformer import CustomTransformerModel
from configs.model_config import CustomTransformerConfig
from transformers import AutoTokenizer
from utils.tokens import get_or_prompt_token

rng = np.random.default_rng()


def disable_dropout(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            setattr(model, name, torch.nn.Identity())
    return model


def filter_logits(logits, top_k=0, top_p=1.0, temperature=1.0):
    original_shape = logits.shape
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    logits = logits.clone()
    if temperature == 0.0:
        # Greedy: return one-hot-like logits (only argmax survives)
        max_ids = logits.argmax(dim=-1, keepdim=True)
        mask = torch.zeros_like(logits).scatter_(-1, max_ids, 1.0)
        logits = torch.where(mask.bool(), logits, torch.full_like(logits, float('-inf')))
    elif temperature != 1.0:
        logits = logits / temperature
    if top_k > 0 and top_k < logits.size(-1):
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        thresholds = topk_vals[:, -1].unsqueeze(-1)
        logits = torch.where(logits < thresholds, torch.full_like(logits, float("-inf")), logits)
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = probs.cumsum(dim=-1)
        mask = cum_probs > top_p
        mask[:, 0] = False
        scatter_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(dim=-1, index=sorted_indices, src=mask)
        logits = torch.where(scatter_mask, torch.full_like(logits, float("-inf")), logits)
    if original_shape[0] == 1:
        logits = logits.unsqueeze(0)
    return logits


def get_noising_schedule(i, max_it, sharpness=5.0):
    x = i / max_it
    return (np.exp(-sharpness * x) - np.exp(-sharpness)) / (1 - np.exp(-sharpness))


def noisify_answer(input_ids, answer_start, threshold=1.0, is_unmasked=None, mask_token_id=128002):
    noised = input_ids.copy()
    total_len = len(input_ids)
    candidates = [i for i in range(answer_start, total_len) if is_unmasked is None or not is_unmasked[i]]
    num_to_add = int(threshold * total_len)
    if num_to_add > 0 and len(candidates) > 0:
        newly_masked = rng.choice(candidates, size=min(num_to_add, len(candidates)), replace=False)
        for idx in newly_masked:
            noised[idx] = mask_token_id
    return noised


def generate_diffusion_text(model, input_ids, answer_start, top_k=0, top_p=1.0,
                            temperature=1.0, eos_token_id=None, eos_boost=0.0):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)
        logits = model(input_ids=input_tensor)["logits"]
        if eos_token_id is not None and eos_boost != 0.0:
            logits[:, :, eos_token_id] += eos_boost
        filtered_logits = filter_logits(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        probs = torch.softmax(filtered_logits, dim=-1).squeeze()
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidences = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
    return input_ids[:answer_start] + sampled[answer_start:].tolist(), confidences


def generate_summary(source_text, model, tokenizer, max_it=16, noise_start=0.5,
                     noising_sharpness=5.0, max_length=256, top_k=100, top_p=1.0,
                     temperature=1.0, add_tokens=256):
    """Generate clinical summary using TINI diffusion."""
    eos_token_id = tokenizer.eos_token_id

    prompt = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a clinical summarization assistant. Generate a concise clinical summary "
        "based on the provided medical text.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{source_text.strip()}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=4096)
    marker = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False)

    def find_marker(ids, marker):
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return None

    answer_start = find_marker(input_ids, marker)
    if answer_start is None:
        raise ValueError("Assistant marker not found in prompt.")

    mask_token = tokenizer.encode("MASK", add_special_tokens=False)[0]

    if len(input_ids) < max_length:
        input_ids += [mask_token] * (max_length - len(input_ids))

    current_tokens = input_ids[:answer_start] + [mask_token] * add_tokens

    last_clean_answers = []
    t0 = time.time()
    for step in range(max_it):
        if not len(current_tokens) >= max_length:
            current_tokens += [mask_token] * add_tokens

        current_tokens, confidence_scores = generate_diffusion_text(
            model, current_tokens, answer_start,
            top_k=top_k, top_p=top_p, temperature=temperature,
            eos_token_id=eos_token_id, eos_boost=0.0
        )

        answer_tokens = current_tokens[answer_start:]
        if eos_token_id in answer_tokens:
            answer_tokens = answer_tokens[:answer_tokens.index(eos_token_id)]
        answer_tokens = [t for t in answer_tokens if t != eos_token_id]

        last_clean_answers.append(answer_tokens)
        if len(last_clean_answers) > 3:
            last_clean_answers.pop(0)
            if all(ans == last_clean_answers[0] for ans in last_clean_answers):
                break

        if step < max_it - 1:
            threshold = noise_start * get_noising_schedule(step, max_it, sharpness=noising_sharpness)
            current_tokens = noisify_answer(current_tokens, answer_start, threshold=threshold,
                                            mask_token_id=mask_token)

    elapsed = time.time() - t0
    actual_iterations = step + 1
    return tokenizer.decode(current_tokens[answer_start:], skip_special_tokens=True).strip(), elapsed, actual_iterations


def load_data_from_zip(zip_path):
    """Load MultiClinSum data from zip. Returns (ids, sources, references)."""
    import zipfile
    sources, refs = [], []
    with zipfile.ZipFile(zip_path, 'r') as z:
        fulltext_files = sorted([f for f in z.namelist() if '/fulltext/' in f and f.endswith('.txt')])
        summary_files = sorted([f for f in z.namelist() if '/summaries/' in f and f.endswith('.txt')])

        for f in fulltext_files:
            sources.append(z.read(f).decode('utf-8'))
        for f in summary_files:
            refs.append(z.read(f).decode('utf-8'))

    ids = list(range(len(sources)))
    return ids, sources, refs


def main():
    parser = argparse.ArgumentParser(description="TINI-LAD batch inference for MultiClinSum")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base tokenizer model")
    parser.add_argument("--data_zip", default="./data/multiclinsum_test_en.zip")
    parser.add_argument("--sample_indices", default=None, help="Path to sampled indices file (optional)")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--output_name", default="tini_test")
    parser.add_argument("--max_it", type=int, default=64, help="Diffusion iterations (aligned with LLaDA)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_tokens", type=int, default=128, help="Tokens to add per iteration (aligned with LLaDA gen_length)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Greedy decoding (aligned with LLaDA)")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--hf_token", default=None, help="Hugging Face token")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading tokenizer from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True,
        token=args.hf_token or os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float32
    )
    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
    model = disable_dropout(model)
    for m in model.modules():
        if hasattr(m, "lora_A") and not hasattr(m, "lora_variant"):
            m.lora_variant = {}
    if hasattr(model, "llama") and hasattr(model.llama, "base_model"):
        model.llama.base_model.has_active_enabled_adapter = False
        model.llama.base_model.enable_adapters = lambda: None
        model.llama.base_model.disable_adapters = lambda: None
    model.to(torch.bfloat16)
    model.to(device)
    model.eval()
    print("Model loaded.")

    # ── Load data ──
    print(f"Loading data from {args.data_zip} ...")
    all_ids, all_sources, all_refs = load_data_from_zip(args.data_zip)
    print(f"Total samples: {len(all_ids)}")

    # ── Sample indices ──
    if args.sample_indices:
        with open(args.sample_indices) as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        print(f"Running on {len(indices)} sampled instances from {args.sample_indices}")
    else:
        indices = list(range(len(all_ids)))
        print(f"Running on all {len(indices)} instances")

    # ── Run inference ──
    results = []
    metrics = []
    start_time = time.time()
    for idx in tqdm(indices, desc="TINI inference"):
        source = all_sources[idx]
        ref = all_refs[idx]
        try:
            pred, elapsed_sample, actual_iters = generate_summary(
                source, model, tokenizer,
                max_it=args.max_it, max_length=args.max_length,
                add_tokens=args.add_tokens,
                temperature=args.temperature, top_k=args.top_k
            )
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            pred, elapsed_sample, actual_iters = "", 0, 0
        results.append({"id": idx, "reference_summary": ref, "predicted_summary": pred})
        metrics.append({
            "id": idx,
            "generation_time_seconds": round(elapsed_sample, 3),
            "iterations": actual_iters,
        })

    total_elapsed = time.time() - start_time
    print(f"\nDone. {len(results)} samples in {total_elapsed:.0f}s ({total_elapsed/len(results):.1f}s/sample)")

    # ── Save ──
    df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    # ── Save metrics ──
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(args.output_dir, f"{args.output_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    avg_latency = metrics_df['generation_time_seconds'].mean()
    avg_iters = metrics_df['iterations'].mean()
    print(f"Metrics saved → {metrics_path}")
    print(f"Avg latency: {avg_latency:.2f}s | Avg iterations: {avg_iters:.1f}")


if __name__ == "__main__":
    main()
