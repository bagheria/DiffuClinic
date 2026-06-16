"""Entity-pinned LLaDA inference - Li's block-wise denoising loop plus a decaying logit bias
on the selected source entities"""
import argparse
import gc
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_utils import find_latest_checkpoint, load_data_from_zip
from utilities.entity_selector import EntitySelector
from utilities.extractor import extract_all

PROMPT_TEMPLATE = "Summarize this clinical note: {full_text}\nSummary: "


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
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, anchor_ids=None, lambda_max=2.0, annealing_power=1.0):

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    total_steps = num_blocks * steps
    global_step = 0

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

            if anchor_ids is not None and lambda_max > 0.0:
                bonus = lambda_max * (1.0 - global_step / total_steps) ** annealing_power
                if bonus > 0.0:
                    logits[:, prompt.shape[1]:, anchor_ids] += bonus
            global_step += 1

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


def build_anchor_ids(selected, tokenizer, device):
    ids = []
    seen = set()
    for entity in selected:
        tokens = tokenizer.encode(" " + entity, add_special_tokens=False)
        if not tokens:
            continue
        anchor = tokens[0]
        if anchor not in seen:
            seen.add(anchor)
            ids.append(anchor)
    if not ids:
        return None
    return torch.tensor(ids, dtype=torch.long, device=device)


def summarize(model, tokenizer, full_text, device, cache_ref, selector, gen_length=128, steps=64, block_length=16, temperature=0., cfg_scale=0., remasking='low_confidence', lambda_max=2.0, annealing_power=1.0):
    prompt = PROMPT_TEMPLATE.format(full_text=full_text)

    encoded = tokenizer([prompt], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    entities = extract_all(full_text)
    selected = selector.select(entities, source_text=full_text, gen_length=gen_length, tokenizer=tokenizer)
    anchor_ids = build_anchor_ids(selected, tokenizer, device)

    if cache_ref[0] is not None:
        try:
            cache_ref[0]().reset_cache(input_ids.shape[1])
        except Exception:
            cache_ref[0] = None

    t0 = time.time()
    out = generate(model, input_ids, attention_mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=temperature, cfg_scale=cfg_scale, remasking=remasking, anchor_ids=anchor_ids, lambda_max=lambda_max, annealing_power=annealing_power)
    elapsed = time.time() - t0
    pred_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    num_blocks = gen_length // block_length
    diffusion_steps = num_blocks * (steps // num_blocks)
    return pred_text, elapsed, diffusion_steps, len(selected)


def main():
    parser = argparse.ArgumentParser(description="Entity-pinned LLaDA inference for clinical summarization")
    parser.add_argument("--data_path", required=True, help="Path to MultiClinSum test zip file")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base", help="Base model ID or local path")
    parser.add_argument("--lora_path", default="CondeSoulrack/llada-clinical-summary-lora-ls", help="LoRA adapter ID or path")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--output_name", default="llada_entity_pinned", help="Output CSV base name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--lambda_max", type=float, default=2.0, help="Max entity-bias bonus; 0.0 disables pinning (no-pin control)")
    parser.add_argument("--annealing_power", type=float, default=1.0, help="Exponent on the (1 - t/T) bias decay")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--idf_weights", default="utilities/idf_weights.json", help="Fitted IDF table for the entity selector")
    parser.add_argument("--no_dllm_cache", action="store_true", help="Disable dLLM-Cache (recommended for pinned runs)")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch, ignore existing checkpoints")
    parser.add_argument("--sample_indices", default=None, help="Path to file with pre-sampled indices (one per line)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"LLaDA Entity-Pinned Inference")
    print(f"Model: {args.model_path}")
    print(f"LoRA: {args.lora_path}")
    print(f"gen_length={args.gen_length} steps={args.steps} block_length={args.block_length}")
    print(f"lambda_max={args.lambda_max} (annealing_power={args.annealing_power})")
    print(f"dLLM-Cache: {'OFF' if args.no_dllm_cache else 'ON'}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

    df = load_data_from_zip(args.data_path)

    import transformers
    print(f"Transformers version: {transformers.__version__}")

    torch.cuda.empty_cache()
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=args.device, token=hf_token).eval()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.lora_path, token=hf_token)
    model = model.merge_and_unload()
    print(f"LoRA adapter loaded and merged from {args.lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    selector = EntitySelector.load(args.idf_weights)
    print(f"Entity selector loaded from {args.idf_weights}")

    cache_ref = [None]
    if not args.no_dllm_cache:
        try:
            if not os.path.exists('dLLM-cache'):
                import subprocess
                subprocess.run(['git', 'clone', 'https://github.com/maomaocun/dLLM-cache.git'], check=True)
            sys.path.append('dLLM-cache')
            from dllm_cache.cache import dLLMCache, dLLMCacheConfig
            from dllm_cache.hooks import register_cache_LLaDA

            dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=100, gen_interval_steps=7, transfer_ratio=0.25)))
            register_cache_LLaDA(model, "model.transformer.blocks")
            cache_ref[0] = dLLMCache
            print("dLLM-Cache acceleration enabled")
        except Exception as e:
            print(f"dLLM-Cache not available ({e}), continuing without it")

    print("Model loaded.")

    if args.sample_indices:
        with open(args.sample_indices) as f:
            run_indices = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(run_indices)} sample indices from {args.sample_indices}")
    else:
        run_indices = list(range(len(df)))

    start_pos = 0
    results = []
    if not args.no_resume:
        start_pos, results = find_latest_checkpoint(args.output_dir, args.output_name)

    metrics = []
    print(f"\nProcessing {len(run_indices)} samples (starting from position {start_pos})...")
    for pos in tqdm(range(start_pos, len(run_indices)), desc="Generating", initial=start_pos, total=len(run_indices)):
        idx = run_indices[pos]
        row = df.iloc[idx]
        try:
            pred, elapsed, diffusion_steps, n_pinned = summarize(model, tokenizer, row['Full_Text'], args.device, cache_ref, selector, gen_length=args.gen_length, steps=args.steps, block_length=args.block_length, temperature=args.temperature, cfg_scale=args.cfg_scale, remasking=args.remasking, lambda_max=args.lambda_max, annealing_power=args.annealing_power)
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            pred, elapsed, diffusion_steps, n_pinned = "", 0, 0, 0

        results.append({'id': idx, 'reference_summary': row['Summary'], 'predicted_summary': pred})
        metrics.append({'id': idx, 'generation_time_seconds': round(elapsed, 3), 'diffusion_steps': diffusion_steps, 'n_entities_pinned': n_pinned})

        if (pos + 1) % args.checkpoint_every == 0:
            pd.DataFrame(results).to_csv(f'{args.output_dir}/checkpoint_{args.output_name}_{pos + 1}.csv', index=False)
            torch.cuda.empty_cache()
            gc.collect()

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
    avg_pinned = metrics_df['n_entities_pinned'].mean()
    print(f"Successful: {successful}/{len(results_df)}")
    print(f"Avg summary length: {avg_len:.0f} chars")
    print(f"Avg latency: {avg_latency:.2f}s | Avg diffusion steps: {avg_steps:.1f}")
    print(f"Avg entities pinned: {avg_pinned:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
