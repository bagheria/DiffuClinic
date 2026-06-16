import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import LoraConfig, get_peft_model
from configs.model_config import CustomTransformerConfig

class CustomTransformerModel(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config, hf_token=None):
        super().__init__(config)
        self.llama = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B",
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        self.llama.resize_token_embeddings(config.vocab_size)

        for param in self.llama.parameters():
            param.requires_grad = False
        for param in self.llama.lm_head.parameters():
            param.requires_grad = True

        lora_config = LoraConfig(
            r=1024, lora_alpha=1024, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none", task_type=None
        )
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        # assert seq_len == self.config.prediction_chunk, (
        #     f"Expected input length {self.config.prediction_chunk}, got {seq_len}")

        device = input_ids.device
        masking_type = getattr(self.config, "masking_type", "bidirectional")

        if masking_type == 'bidirectional':
            base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        elif masking_type == 'bidirectional_masked':
            base_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
            base_mask.fill_diagonal_(False)
        elif masking_type == 'unidirectional':
            base_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        else:
            raise ValueError(f"Unknown masking type: {masking_type}")

        attention_mask = base_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, seq_len, seq_len).clone()
        attention_mask = attention_mask.to(dtype=torch.float32)


        with autocast("cuda", dtype=torch.float16):
            outputs = self.llama(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                **kwargs
            )

        logits = outputs.logits[:, :, :self.config.vocab_size].view(batch_size, seq_len, self.config.vocab_size)
        loss = None

        if labels is not None:
            assert labels.shape == (batch_size, seq_len)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
