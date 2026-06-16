
import argparse
import os

import torch
import yaml
from peft import get_peft_model

from training.loop import train as run_training_loop
from training.paradigms import get_paradigm


def _resolve_hf_token(cli_token):
    return cli_token or os.environ.get("HF_TOKEN")


def _log_environment():
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model by paradigm.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (else the HF_TOKEN env var).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["hf_token"] = _resolve_hf_token(args.hf_token)

    _log_environment()
    paradigm = get_paradigm(config["paradigm"])

    model, tokenizer = paradigm.load_model_and_tokenizer(config)

    if not getattr(paradigm, "WRAPS_OWN_LORA", False):
        model = get_peft_model(model, paradigm.build_lora_config(config))
        model.print_trainable_parameters()

    dataset = paradigm.build_dataset(config, tokenizer)

    collate_fn = None
    if hasattr(paradigm, "build_collator"):
        collate_fn = paradigm.build_collator(config, tokenizer)

    training = config["training"]
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=training["batch_size"], shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    config["pad_token_id"] = tokenizer.pad_token_id

    run_training_loop(model, train_loader, paradigm.compute_loss, config, tokenizer)

    if hasattr(paradigm, "save_final"):
        paradigm.save_final(model, tokenizer, config)

    push = config.get("push_to_hub", {})
    if push.get("enabled", False):
        repo_id = push["repo_id"]
        model.push_to_hub(repo_id, token=config["hf_token"])
        tokenizer.push_to_hub(repo_id, token=config["hf_token"])
        print(f"Pushed -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
