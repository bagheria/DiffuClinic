"""
Run inference with LLaDA (diffusion language model) for clinical summarization.

Usage:
    python inference_llada.py \
        --model_path "GSAI-ML/LLaDA-8B-Instruct" \
        --data_path "data/multiclinsum_test_en.zip" \
        --output_dir "./results" \
        --output_name "llada"

Supports both base and fine-tuned LLaDA models.
"""
import os
import sys
import time
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from dataclasses import asdict
import gc

from utils import load_data_from_zip, find_latest_checkpoint

PROMPT_TEMPLATE = """Please generate a concise clinical summary based on the following medical dialogue:

{full_text}

Clinical Summary:"""


# ============================================================
# Diffusion generation helpers (ported from inference.ipynb)
# ============================================================

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128,
             block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence',
             mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


# ============================================================
# Summary generation
# ============================================================

def summarize(model, tokenizer, full_text, device, cache_ref,
              gen_length=128, steps=64, block_length=16,
              temperature=0., cfg_scale=0., remasking='low_confidence'):
    prompt = PROMPT_TEMPLATE.format(full_text=full_text)
    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        formatted_prompt = prompt

    encoded = tokenizer([formatted_prompt], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    if cache_ref[0] is not None:
        try:
            cache_ref[0]().reset_cache(input_ids.shape[1])
        except Exception:
            cache_ref[0] = None  # disable on first failure

    t0 = time.time()
    out = generate(
        model, input_ids, attention_mask,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=temperature, cfg_scale=cfg_scale, remasking=remasking,
    )
    elapsed = time.time() - t0
    pred_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    total_diffusion_steps = steps * (gen_length // block_length)
    return pred_text, elapsed, total_diffusion_steps


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLaDA diffusion model inference for clinical summarization")
    parser.add_argument("--model_path", required=True, help="Model ID or local path")
    parser.add_argument("--data_path", required=True, help="Path to MultiClinSum zip file")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--output_name", default="llada_summaries", help="Output CSV base name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--no_dllm_cache", action="store_true", help="Disable dLLM-Cache acceleration")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch, ignore existing checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LLaDA Inference: {args.model_path}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # ---- Load data ----
    df = load_data_from_zip(args.data_path)

    # ---- Load model ----
    import transformers
    print(f"Transformers version: {transformers.__version__}")

    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=args.device
    ).eval()

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print(f"LoRA adapter loaded and merged from {args.lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # ---- dLLM-Cache (optional) ----
    cache_ref = [None]  # mutable so summarize() can disable on failure
    if not args.no_dllm_cache:
        try:
            if not os.path.exists('dLLM-cache'):
                import subprocess
                subprocess.run(['git', 'clone', 'https://github.com/maomaocun/dLLM-cache.git'], check=True)
            sys.path.append('dLLM-cache')
            from dllm_cache.cache import dLLMCache, dLLMCacheConfig
            from dllm_cache.hooks import register_cache_LLaDA

            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                prompt_interval_steps=100, gen_interval_steps=7, transfer_ratio=0.25,
            )))
            register_cache_LLaDA(model, "model.transformer.blocks")
            cache_ref[0] = dLLMCache
            print("dLLM-Cache acceleration enabled")
        except Exception as e:
            print(f"dLLM-Cache not available ({e}), continuing without it")

    print("Model loaded.")

    # ---- Setup resume ----
    start_idx = 0
    results = []
    if not args.no_resume:
        start_idx, results = find_latest_checkpoint(args.output_dir)
    total = len(df)

    # ---- Inference loop ----
    metrics = []
    print(f"\nProcessing {total} samples (starting from {start_idx})...")
    for idx in tqdm(range(start_idx, total), desc="Generating", initial=start_idx, total=total):
        row = df.iloc[idx]
        try:
            pred, elapsed, diffusion_steps = summarize(
                model, tokenizer, row['Full_Text'], args.device, cache_ref,
                gen_length=args.gen_length, steps=args.steps,
                block_length=args.block_length, temperature=args.temperature,
                cfg_scale=args.cfg_scale, remasking=args.remasking,
            )
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            pred, elapsed, diffusion_steps = "", 0, 0

        results.append({
            'id': idx,
            'reference_summary': row['Summary'],
            'predicted_summary': pred,
        })
        metrics.append({
            'id': idx,
            'generation_time_seconds': round(elapsed, 3),
            'diffusion_steps': diffusion_steps,
        })

        if (idx + 1) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(
                f'{args.output_dir}/checkpoint_{idx + 1}.csv', index=False
            )
            torch.cuda.empty_cache()
            gc.collect()

    # ---- Save final ----
    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(metrics)

    csv_path = f'{args.output_dir}/{args.output_name}.csv'
    results_df.to_csv(csv_path, index=False)
    metrics_df.to_csv(f'{args.output_dir}/{args.output_name}_metrics.csv', index=False)
    print(f"\nSaved {len(results_df)} results to {csv_path}")
    print(f"Saved metrics to {args.output_dir}/{args.output_name}_metrics.csv")

    successful = sum(1 for r in results if r['predicted_summary'])
    avg_len = results_df['predicted_summary'].str.len().mean()
    avg_latency = metrics_df['generation_time_seconds'].mean()
    avg_steps = metrics_df['diffusion_steps'].mean()
    print(f"Successful: {successful}/{len(results_df)}")
    print(f"Avg summary length: {avg_len:.0f} chars")
    print(f"Avg latency: {avg_latency:.2f}s | Avg diffusion steps: {avg_steps:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
