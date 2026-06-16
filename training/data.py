
import os

import pandas as pd
import torch
from torch.utils.data import Dataset


_TRAIN_SPLITS = ("multiclinsum_gs_train_en", "multiclinsum_large-scale_train_en")


def load_split(split_name: str, data_dir: str) -> pd.DataFrame:
    fulltext_dir = os.path.join(data_dir, split_name, "fulltext")
    summary_dir = os.path.join(data_dir, split_name, "summaries")
    records = []
    for file_name in os.listdir(fulltext_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(fulltext_dir, file_name), "r", encoding="utf-8") as f:
                full_text = f.read()
            summary_name = file_name.replace(".txt", "_sum.txt")
            with open(os.path.join(summary_dir, summary_name), "r", encoding="utf-8") as f:
                summary_text = f.read()
            records.append({"Full_Text": full_text, "Summary": summary_text})
    return pd.DataFrame(records)


def load_multiclinsum(config: dict) -> pd.DataFrame:
    data_dir = config["data_dir"]
    frames = [load_split(split, data_dir) for split in _TRAIN_SPLITS]
    train_df = pd.concat(frames, ignore_index=True)
    print(f"Total training on: {len(train_df)} examples")
    return train_df


class ClinicalSumDataset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:
        self.samples = []
        eos_id = tokenizer.eos_token_id
        skipped = 0

        for _, row in df.iterrows():
            prompt_text = f"Summarize this clinical note: {row['Full_Text']}\nSummary: "
            response_text = row["Summary"]

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            response_ids = tokenizer.encode(response_text, add_special_tokens=False)
            response_ids = response_ids + [eos_id]

            if len(prompt_ids) >= max_length:
                skipped += 1
                continue

            full_ids = (prompt_ids + response_ids)[:max_length]
            pad_len = max_length - len(full_ids)
            full_ids = full_ids + [eos_id] * pad_len

            self.samples.append({"input_ids": torch.tensor(full_ids, dtype=torch.long), "prompt_length": torch.tensor(len(prompt_ids), dtype=torch.long)})

        print(f"Dataset: {len(self.samples)} usable, {skipped} skipped (prompt > max_length).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
