
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MASK_ID = 126336


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base)
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


def _anchor_token_ids(entity_strings: list[str], tokenizer) -> list[int]:
    anchors: list[int] = []
    seen: set[int] = set()
    for entity in entity_strings:
        ids = tokenizer.encode(entity, add_special_tokens=False)
        if not ids:
            logger.debug("Entity %r tokenized to nothing, skipping", entity)
            continue
        anchor = ids[0]
        if anchor not in seen:
            seen.add(anchor)
            anchors.append(anchor)
    return anchors


@torch.no_grad()
def entity_pinned_generate(model, tokenizer, prompt_ids: torch.Tensor, entity_strings: list[str], gen_length: int = 128, steps: int = 128, mask_id: int = DEFAULT_MASK_ID, lambda_max: float = 2.0, annealing_power: float = 1.0, temperature: float = 0.0, remasking: str = "low_confidence", device: str = "cuda") -> str:
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    anchors = _anchor_token_ids(entity_strings, tokenizer)
    anchor_tensor = torch.tensor(anchors, dtype=torch.long, device=device) if anchors else None
    logger.debug("Pinning %d anchor tokens over %d steps", len(anchors), steps)

    num_transfer_tokens = get_num_transfer_tokens(x[:, prompt_len:] == mask_id, steps)

    for i in range(steps):
        mask_index = x == mask_id
        logits = model(x).logits

        if anchor_tensor is not None:
            progress = i / steps
            bonus = lambda_max * (1.0 - progress) ** annealing_power
            if bonus > 0.0:
                logits[:, prompt_len:, anchor_tensor] += bonus

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if remasking == "low_confidence":
            probs = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise ValueError(f"Unknown remasking strategy: {remasking!r}")

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i]))
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

    generated = x[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def entity_pinned_generate_sweep(model, tokenizer, prompt_ids: torch.Tensor, entity_strings: list[str], lambda_max_grid: list[float], annealing_power_grid: list[float], gen_length: int = 128, steps: int = 128, mask_id: int = DEFAULT_MASK_ID, device: str = "cuda", **generate_kwargs) -> dict[tuple[float, float], str]:
    results: dict[tuple[float, float], str] = {}
    for lambda_max in lambda_max_grid:
        for annealing_power in annealing_power_grid:
            logger.debug("Sweep cell lambda_max=%s annealing_power=%s", lambda_max, annealing_power)
            results[(lambda_max, annealing_power)] = entity_pinned_generate(model=model, tokenizer=tokenizer, prompt_ids=prompt_ids, entity_strings=entity_strings, gen_length=gen_length, steps=steps, mask_id=mask_id, lambda_max=lambda_max, annealing_power=annealing_power, device=device, **generate_kwargs)
    return results


class _BiasProbe:

    def __init__(self, vocab_size: int, device: str) -> None:
        self.vocab_size = vocab_size
        self.device = device

    def __call__(self, x: torch.Tensor):
        batch, seq = x.shape
        logits = torch.zeros(batch, seq, self.vocab_size, device=self.device)
        for token in torch.unique(x):
            tok = int(token)
            if 0 <= tok < self.vocab_size:
                logits[:, :, tok] += 0.1

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    class _CharTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 256 for c in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(i)) for i in ids if int(i) != _SMOKE_MASK_ID % 256)

    _SMOKE_MASK_ID = 255
    _VOCAB = 256
    _DEVICE = "cpu"

    tokenizer = _CharTokenizer()
    model = _BiasProbe(vocab_size=_VOCAB, device=_DEVICE)
    prompt = torch.tensor([tokenizer.encode("summarize: ")], dtype=torch.long)

    summary = entity_pinned_generate(model=model, tokenizer=tokenizer, prompt_ids=prompt, entity_strings=["metformin", "diabetes"], gen_length=16, steps=16, mask_id=_SMOKE_MASK_ID, lambda_max=5.0, annealing_power=1.0, device=_DEVICE)
    logger.info("Smoke-test generated %d chars: %r", len(summary), summary)
