"""
Run inference with Llama (autoregressive model) for clinical summarization.

Usage:
    # Base model
    python inference_llama.py \
        --model_path "meta-llama/Llama-3.1-8B-Instruct" \
        --data_path "data/multiclinsum_test_en.zip" \
        --output_dir "./results" --output_name "llama"

    # With LoRA adapter
    python inference_llama.py \
        --model_path "meta-llama/Llama-3.1-8B-Instruct" \
        --lora_path "./llama_base_medical_lora" \
        --prompt_template clinical_note \
        --data_path "data/multiclinsum_test_en.zip" \
        --output_dir "./results" --output_name "llama_ft"
"""
import os
import time
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from utils import load_data_from_zip, find_latest_checkpoint

# Prompt for base instruct model
PROMPT_DIALOGUE = """Please generate a concise clinical summary based on the following medical dialogue:

{full_text}

Clinical Summary:"""

# Prompt for LoRA fine-tuned model
PROMPT_CLINICAL_NOTE = "Summarize this clinical note: {full_text}\nSummary: "

PROMPT_MAP = {
    "dialogue": PROMPT_DIALOGUE,
    "clinical_note": PROMPT_CLINICAL_NOTE,
}


def summarize(model, tokenizer, full_text, device, prompt_template, max_new_tokens=128, temperature=0.0, do_sample=False):
    prompt = prompt_template.format(full_text=full_text)
    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    generated_ids = outputs[0][input_ids.shape[1]:]
    pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return pred_text, elapsed, len(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Llama AR model inference for clinical summarization")
    parser.add_argument("--model_path", required=True, help="Model ID or local path")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--data_path", required=True, help="Path to MultiClinSum zip file")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--output_name", default="llama_summaries", help="Output CSV base name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--prompt_template", default="dialogue", choices=["dialogue", "clinical_note"],
                        help="Prompt format: 'dialogue' for base model, 'clinical_note' for fine-tuned")
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch, ignore existing checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Llama Inference: {args.model_path}")
    if args.lora_path:
        print(f"LoRA adapter: {args.lora_path}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    prompt_template = PROMPT_MAP[args.prompt_template]

    # ---- Load data ----
    df = load_data_from_zip(args.data_path)

    # ---- Load model ----
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.float16,
            device_map=args.device, attn_implementation="flash_attention_2",
        )
        print("Flash Attention 2 enabled")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=args.device
        )
        print("Flash Attention 2 not available, using default attention")

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print("LoRA adapter loaded and merged")

    model.eval()
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
            pred, elapsed, num_tokens = summarize(
                model, tokenizer, row['Full_Text'], args.device, prompt_template,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                do_sample=args.do_sample,
            )
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            pred, elapsed, num_tokens = "", 0, 0

        results.append({
            'id': idx,
            'reference_summary': row['Summary'],
            'predicted_summary': pred,
        })
        metrics.append({
            'id': idx,
            'generation_time_seconds': round(elapsed, 3),
            'tokens_generated': num_tokens,
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
    avg_tokens = metrics_df['tokens_generated'].mean()
    total_tokens = metrics_df['tokens_generated'].sum()
    print(f"Successful: {successful}/{len(results_df)}")
    print(f"Avg summary length: {avg_len:.0f} chars")
    print(f"Avg latency: {avg_latency:.2f}s | Avg tokens: {avg_tokens:.1f} | Total tokens: {total_tokens}")
    print("Done.")


if __name__ == "__main__":
    main()
