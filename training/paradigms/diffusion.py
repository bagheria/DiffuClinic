
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig

from training.data import ClinicalSumDataset, load_multiclinsum

WRAPS_OWN_LORA = False


def load_model_and_tokenizer(config: dict):
    token = config.get("hf_token")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True, token=token)

    model = AutoModel.from_pretrained(config["model_name"], trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", token=token)

    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    return model, tokenizer


def build_lora_config(config: dict) -> LoraConfig:
    lora = config["lora"]
    return LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=lora["target_modules"], bias="none")


def build_dataset(config: dict, tokenizer):
    df = load_multiclinsum(config)
    return ClinicalSumDataset(df, tokenizer, config["max_length"])


def forward_process(input_ids, mask_token_id: int, eps: float = 1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    return noisy_batch, masked_indices, p_mask


def compute_loss(model, batch, config: dict):
    input_ids = batch["input_ids"]
    prompt_lengths = batch["prompt_length"]
    mask_token_id = config["mask_token_id"]
    eps = config["diffusion"]["forward_eps"]

    noisy_batch, _, p_mask = forward_process(input_ids, mask_token_id, eps)

    positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device)
    positions = positions.unsqueeze(0).expand(noisy_batch.shape[0], -1)
    prompt_mask = positions < prompt_lengths.unsqueeze(1)
    noisy_batch[prompt_mask] = input_ids[prompt_mask]

    answer_lengths = torch.sum((~prompt_mask).to(torch.int64), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.expand_as(noisy_batch)

    masked_indices = (noisy_batch == mask_token_id)

    logits = model(input_ids=noisy_batch).logits

    token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction="none") / p_mask[masked_indices]

    return torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
