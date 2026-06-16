
import os
import sys
import types

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download

from training.data import load_multiclinsum

WRAPS_OWN_LORA = True

_LAD_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "lad-code")


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
    if "models" not in sys.modules:
        models_package = types.ModuleType("models")
        models_package.__path__ = [os.path.join(_LAD_ROOT, "models")]
        sys.modules["models"] = models_package
    import models.custom_transformer
    import configs.model_config
    import models.custom_transformer
    import configs.model_config

    import __main__
    from models.custom_transformer import CustomTransformerModel
    from configs.model_config import CustomTransformerConfig
    if not hasattr(__main__, "CustomTransformerModel"):
        __main__.CustomTransformerModel = CustomTransformerModel
    if not hasattr(__main__, "CustomTransformerConfig"):
        __main__.CustomTransformerConfig = CustomTransformerConfig


class _DiffusionDataCollator:

    def __call__(self, features):
        input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
        labels = torch.stack([torch.tensor(f["labels"], dtype=torch.long) for f in features])
        return {"input_ids": input_ids, "labels": labels}


class _LadDataset(Dataset):

    def __init__(self, input_ids, labels) -> None:
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def build_lora_config(config: dict) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias="none", task_type=None)


def _load_and_merge(config: dict):
    _register_ladcode_classes_for_unpickling()

    local_path = config.get("checkpoint_path")
    if local_path is None:
        local_path = hf_hub_download(repo_id=config["checkpoint_repo"], filename=config["checkpoint_file"], token=config.get("hf_token"))

    model = torch.load(local_path, map_location="cpu", weights_only=False)

    for module in model.modules():
        if hasattr(module, "lora_A") and not hasattr(module, "lora_variant"):
            module.lora_variant = {}

    if hasattr(model, "llama") and hasattr(model.llama, "merge_and_unload"):
        model.llama = model.llama.merge_and_unload()
    elif hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    else:
        raise RuntimeError(
            "No PeftModel to merge: neither model.llama nor model exposes "
            "merge_and_unload. The checkpoint layout is not what lad.py expects."
        )

    return model


def _add_fresh_adapter_and_freeze(model, config: dict):
    inner = model.llama if hasattr(model, "llama") else model
    wrapped = get_peft_model(inner, build_lora_config(config))
    if hasattr(model, "llama"):
        model.llama = wrapped
    else:
        model = wrapped

    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    return model


def load_model_and_tokenizer(config: dict):
    model = _load_and_merge(config)
    model = _add_fresh_adapter_and_freeze(model, config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True, token=config.get("hf_token"))

    model.train()

    if config.get("fp32_trainable", True):
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainable_dtype = next((p.dtype for p in model.parameters() if p.requires_grad), None)
    frozen_dtype = next((p.dtype for p in model.parameters() if not p.requires_grad), None)
    print(f"trainable dtype: {trainable_dtype} | frozen base dtype: {frozen_dtype}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError(
            "Loaded LAD model has no trainable parameters after adding the fresh r=8 "
            "adapter. Check that build_lora_config targeted modules present in the merged "
            "base and that the merge did not strip the q/v/k/o projections."
        )
    print(f"LAD trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    return model, tokenizer


def build_dataset(config: dict, tokenizer):
    prompt_format = config.get("prompt_format", "lad_assistant")
    if prompt_format != "lad_assistant":
        raise NotImplementedError(
            "prompt_format='unified' is not wired: lad-code's Noiser hardcodes the "
            "'Assistant:' marker to locate the answer span, so unifying to the other "
            "paradigms' prompt would require a lad-code change. Use "
            "prompt_format: lad_assistant or patch third_party/lad-code."
        )

    _ensure_ladcode_on_path()
    from data.noise import Noiser

    df = load_multiclinsum(config)
    max_len = config["max_length"]
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id

    assistant_marker = "\nAssistant:"
    input_ids_list, labels_list = [], []
    skipped = 0
    for _, row in df.iterrows():
        instruction = f"Summarize this clinical note: {row['Full_Text']}"
        prefix = f"User: {instruction.strip()}{assistant_marker}"
        prompt = f"{prefix} {row['Summary'].strip()}"

        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if len(prefix_ids) >= max_len:
            skipped += 1
            continue

        tokenized = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tokenized = tokenized[:max_len]
        tokenized = tokenized + [pad_token] * (max_len - len(tokenized))

        input_ids_list.append(tokenized)
        labels_list.append(list(tokenized))

    print(f"LAD dataset: {len(input_ids_list)} usable, {skipped} skipped (prompt >= max_length).")

    noiser = Noiser(tokenizer)
    corrupted = noiser.corrupt_batch({"input_ids": input_ids_list, "labels": labels_list})
    return _LadDataset(corrupted["input_ids"], corrupted["labels"])


def build_collator(config: dict, tokenizer):
    return _DiffusionDataCollator()


def compute_loss(model, batch, config: dict):
    return model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"]


def save_final(model, tokenizer, config: dict) -> None:
    output_dir = config["output_dir"]
    final_path = os.path.join(output_dir, "lad-finetuned.pth")
    torch.save(model, final_path)
    tokenizer.save_pretrained(output_dir)
    print(f"LAD final model pickled -> {final_path}")
