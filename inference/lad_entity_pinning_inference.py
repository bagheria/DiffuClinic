"""Entity-pinned LAD inference - a pickled CustomTransformerModel decoded by iterative
re-noising, plus a decaying logit bias on the selected source entities"""
import argparse
import gc
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared_utils import find_latest_checkpoint, load_data_from_zip
from utilities.entity_selector import EntitySelector
from utilities.extractor import extract_all

PROMPT_TEMPLATE = "User: Summarize this clinical note: {full_text}\nAssistant:"

# bare "Assistant:" - a leading newline merges into the previous token so the marker would never match
ANSWER_MARKER = "Assistant:"

rng = np.random.default_rng()


_LAD_ROOT = os.path.join(_REPO_ROOT, "third_party", "lad-code")


def _ensure_ladcode_on_path() -> None:
    if not os.path.isdir(_LAD_ROOT):
        raise FileNotFoundError(
            f"lad-code submodule not found at {_LAD_ROOT}. Run "
            "`git submodule update --init third_party/lad-code`."
        )
    if _LAD_ROOT not in sys.path:
        sys.path.insert(0, _LAD_ROOT)


def _register_ladcode_classes_for_unpickling() -> None:
    _ensure_ladcode_on_path()
    # register lad-code's models as a namespace so the pickle's class names resolve (this repo's own models/ shadows it)
    if "models" not in sys.modules:
        models_package = types.ModuleType("models")
        models_package.__path__ = [os.path.join(_LAD_ROOT, "models")]
        sys.modules["models"] = models_package
    import models.custom_transformer
    import configs.model_config

    import __main__
    from models.custom_transformer import CustomTransformerModel
    from configs.model_config import CustomTransformerConfig
    if not hasattr(__main__, "CustomTransformerModel"):
        __main__.CustomTransformerModel = CustomTransformerModel
    if not hasattr(__main__, "CustomTransformerConfig"):
        __main__.CustomTransformerConfig = CustomTransformerConfig


def disable_dropout(model):
    names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)]
    for name in names:
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], torch.nn.Identity())
    return model


def filter_logits(logits, top_k=0, top_p=1.0, temperature=1.0):
    original_shape = logits.shape
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    logits = logits.clone()
    if temperature == 0.0:
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


def generate_diffusion_text(model, input_ids, answer_start, top_k=0, top_p=1.0, temperature=1.0, eos_token_id=None, eos_boost=0.0, anchor_ids=None, lambda_max=0.0, annealing_power=1.0, step=0, max_it=1):
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.device)
        logits = model(input_ids=input_tensor)["logits"]
        if eos_token_id is not None and eos_boost != 0.0:
            logits[:, :, eos_token_id] += eos_boost

        if anchor_ids is not None and lambda_max > 0.0:
            bonus = lambda_max * (1.0 - step / max_it) ** annealing_power
            if bonus > 0.0:
                logits[:, answer_start:, anchor_ids] += bonus

        filtered_logits = filter_logits(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        probs = torch.softmax(filtered_logits, dim=-1).squeeze()
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidences = probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
    return input_ids[:answer_start] + sampled[answer_start:].tolist(), confidences


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


def resolve_prompt(source_text, tokenizer):
    prompt = PROMPT_TEMPLATE.format(full_text=source_text.strip())
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=4096)
    marker = tokenizer.encode(ANSWER_MARKER, add_special_tokens=False)
    answer_start = None
    for i in range(len(input_ids) - len(marker) + 1):
        if input_ids[i:i + len(marker)] == marker:
            answer_start = i + len(marker)
            break
    if answer_start is None:
        raise ValueError(
            f"Answer marker {ANSWER_MARKER!r} (ids={marker}) not found in prompt. "
            f"Prompt ids: {input_ids}"
        )
    return input_ids, answer_start


def generate_summary(source_text, model, tokenizer, selector, device, max_it=64, noise_start=0.5, noising_sharpness=5.0, max_length=512, top_k=100, top_p=1.0, temperature=0.0, add_tokens=128, lambda_max=2.0, annealing_power=1.0):
    eos_token_id = tokenizer.eos_token_id

    input_ids, answer_start = resolve_prompt(source_text, tokenizer)
    mask_token = tokenizer.encode("MASK", add_special_tokens=False)[0]

    entities = extract_all(source_text)
    selected = selector.select(entities, source_text=source_text, gen_length=add_tokens, tokenizer=tokenizer)
    anchor_ids = build_anchor_ids(selected, tokenizer, device)

    if len(input_ids) < max_length:
        input_ids += [mask_token] * (max_length - len(input_ids))

    current_tokens = input_ids[:answer_start] + [mask_token] * add_tokens

    last_clean_answers = []
    t0 = time.time()
    step = 0
    for step in range(max_it):
        if not len(current_tokens) >= max_length:
            current_tokens += [mask_token] * add_tokens

        current_tokens, confidence_scores = generate_diffusion_text(model, current_tokens, answer_start, top_k=top_k, top_p=top_p, temperature=temperature, eos_token_id=eos_token_id, eos_boost=0.0, anchor_ids=anchor_ids, lambda_max=lambda_max, annealing_power=annealing_power, step=step, max_it=max_it)

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
            current_tokens = noisify_answer(current_tokens, answer_start, threshold=threshold, mask_token_id=mask_token)

    elapsed = time.time() - t0
    actual_iterations = step + 1
    pred_text = tokenizer.decode(current_tokens[answer_start:], skip_special_tokens=True).strip()
    return pred_text, elapsed, actual_iterations, len(selected)


def load_lad_model(checkpoint_path, device):
    _register_ladcode_classes_for_unpickling()
    print(f"Loading pickled LAD model from {checkpoint_path} ...")
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    return model


def load_gate(model, tokenizer, device, seq_len=32):
    examples = [
        "User: Summarize this clinical note: 49M with chest pain and raised troponin.\n"
        "Assistant: 49M with chest pain and elevated troponin.",
        "User: Summarize this clinical note: 8F with fever and cough for three days.\n"
        "Assistant: 8F with fever and cough.",
    ]
    rows = []
    for text in examples:
        toks = tokenizer(text, add_special_tokens=False)["input_ids"][:seq_len]
        toks = toks + [tokenizer.eos_token_id] * (seq_len - len(toks))
        rows.append(toks)
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids)["logits"]
    finite = bool(torch.isfinite(logits).all().item())
    print(f"[load gate] logits shape {tuple(logits.shape)} | all-finite={finite}")
    if not finite:
        raise RuntimeError("LAD load gate failed: forward produced non-finite logits.")
    return logits.shape


def main():
    parser = argparse.ArgumentParser(description="Entity-pinned LAD inference for clinical summarization")
    parser.add_argument("--data_path", required=True, help="Path to MultiClinSum test zip file")
    parser.add_argument("--model_repo", default="CondeSoulrack/LAD-clinical-summary-lora-ls", help="HF repo holding the pickled fine-tuned model")
    parser.add_argument("--model_file", default="lad-finetuned.pth", help="Pickled model filename in the repo")
    parser.add_argument("--checkpoint_path", default=None, help="Local .pth override (skips the HF download)")
    parser.add_argument("--tokenizer_repo", default="CondeSoulrack/LAD-clinical-summary-lora-ls", help="HF repo holding the tokenizer (ships the LAD 'MASK' token)")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--output_name", default="lad_entity_pinned", help="Output CSV base name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_it", type=int, default=64, help="Diffusion iterations")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_tokens", type=int, default=128, help="Answer-span length")
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy argmax (Li's LAD baseline)")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--noise_start", type=float, default=0.5)
    parser.add_argument("--noising_sharpness", type=float, default=5.0)
    parser.add_argument("--lambda_max", type=float, default=2.0, help="Max entity-bias bonus (PROVISIONAL; calibrate on smoke). 0.0 = no-pin control")
    parser.add_argument("--annealing_power", type=float, default=1.0, help="Exponent on the (1 - step/max_it) decay")
    parser.add_argument("--idf_weights", default="utilities/idf_weights.json", help="Fitted IDF table for the selector")
    parser.add_argument("--seed", type=int, default=43, help="Seeds numpy (re-noising/sampling) and torch")
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--no_resume", action="store_true", help="Start from scratch, ignore checkpoints")
    parser.add_argument("--sample_indices", default=None, help="File of indices (one per line); smoke mode")
    parser.add_argument("--self_test", action="store_true", help="Load tokenizer+model, run the gates (MASK id, load gate, marker proof), then exit")
    args = parser.parse_args()

    global rng
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 60)
    print("LAD Entity-Pinned Inference")
    print(f"Model: {args.model_repo}/{args.model_file}" if not args.checkpoint_path else f"Model: {args.checkpoint_path}")
    print(f"Tokenizer: {args.tokenizer_repo}")
    print(f"max_it={args.max_it} max_length={args.max_length} add_tokens={args.add_tokens}")
    print(f"temperature={args.temperature} top_k={args.top_k} top_p={args.top_p} " f"noise_start={args.noise_start} noising_sharpness={args.noising_sharpness}")
    print(f"lambda_max={args.lambda_max} annealing_power={args.annealing_power} seed={args.seed}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    import transformers
    print(f"transformers {transformers.__version__}")
    print("=" * 60)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_repo, use_fast=True, token=hf_token)

    mask_ids = tokenizer.encode("MASK", add_special_tokens=False)
    print(f"[MASK] tokenizer.encode('MASK', add_special_tokens=False) = {mask_ids}")
    if len(mask_ids) != 1:
        print("STOP: 'MASK' did not tokenize to a single id. The LAD mask token is wrong; " "check the tokenizer repo.")
        sys.exit(2)
    print(f"[MASK] LAD mask token id = {mask_ids[0]} | eos id = {tokenizer.eos_token_id}")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id=args.model_repo, filename=args.model_file, token=hf_token)
    model = load_lad_model(checkpoint_path, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"Model loaded: {type(model).__name__} | params {n_params:,} | dtypes {sorted(dtypes)} " f"| device {next(model.parameters()).device}")

    load_gate(model, tokenizer, args.device)

    df = load_data_from_zip(args.data_path)

    if args.self_test:
        ids, answer_start = resolve_prompt(df.iloc[0]["Full_Text"], tokenizer)
        marker = tokenizer.encode(ANSWER_MARKER, add_special_tokens=False)
        print("=" * 60)
        print(f"[marker proof] marker {ANSWER_MARKER!r} ids = {marker}")
        print(f"[marker proof] prompt token count = {len(ids)} | answer_start = {answer_start}")
        print(f"[marker proof] ids around marker = {ids[max(0, answer_start - len(marker) - 3):answer_start + 2]}")
        print(f"[marker proof] decoded tail = {tokenizer.decode(ids[max(0, answer_start - 8):answer_start])!r}")
        print("Self-test passed (MASK id, load gate, marker proof).")
        return

    selector = EntitySelector.load(args.idf_weights)
    print(f"Entity selector loaded from {args.idf_weights}")

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
            pred, elapsed, iters, n_pinned = generate_summary(row["Full_Text"], model, tokenizer, selector, args.device, max_it=args.max_it, noise_start=args.noise_start, noising_sharpness=args.noising_sharpness, max_length=args.max_length, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, add_tokens=args.add_tokens, lambda_max=args.lambda_max, annealing_power=args.annealing_power)
        except Exception as e:
            print(f"\nError on sample {idx}: {e}")
            pred, elapsed, iters, n_pinned = "", 0, 0, 0

        results.append({"id": idx, "reference_summary": row["Summary"], "predicted_summary": pred})
        metrics.append({"id": idx, "generation_time_seconds": round(elapsed, 3), "iterations": iters, "n_entities_pinned": n_pinned})

        if (pos + 1) % args.checkpoint_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            pd.DataFrame(results).to_csv(f"{args.output_dir}/checkpoint_{args.output_name}_{pos + 1}.csv", index=False)
            torch.cuda.empty_cache()
            gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    metrics_df = pd.DataFrame(metrics)

    csv_path = f"{args.output_dir}/{args.output_name}.csv"
    results_df.to_csv(csv_path, index=False)
    metrics_df.to_csv(f"{args.output_dir}/{args.output_name}_metrics.csv", index=False)
    print(f"\nSaved {len(results_df)} results to {csv_path}")
    print(f"Saved metrics to {args.output_dir}/{args.output_name}_metrics.csv")

    successful = sum(1 for r in results if r["predicted_summary"])
    if len(results_df):
        avg_len = results_df["predicted_summary"].str.len().mean()
        avg_latency = metrics_df["generation_time_seconds"].mean()
        avg_iters = metrics_df["iterations"].mean()
        avg_pinned = metrics_df["n_entities_pinned"].mean()
        print(f"Successful: {successful}/{len(results_df)}")
        print(f"Avg summary length: {avg_len:.0f} chars")
        print(f"Avg latency: {avg_latency:.2f}s | Avg iterations: {avg_iters:.1f}")
        print(f"Avg entities pinned: {avg_pinned:.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
