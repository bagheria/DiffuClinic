
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

from training.data import ClinicalSumDataset, load_multiclinsum

WRAPS_OWN_LORA = False


def load_model_and_tokenizer(config: dict):
    token = config.get("hf_token")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], token=token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16, device_map="auto", token=token)

    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_lora_config(config: dict) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias="none")


def build_dataset(config: dict, tokenizer):
    df = load_multiclinsum(config)
    return ClinicalSumDataset(df, tokenizer, config["max_length"])


def compute_loss(model, batch, config: dict):
    input_ids = batch["input_ids"]
    prompt_lengths = batch["prompt_length"]
    pad_id = config["pad_token_id"]

    labels = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100

    attention_mask = (input_ids != pad_id).long()
    labels[attention_mask == 0] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss
