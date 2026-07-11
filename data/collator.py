"""
Batch Collator
"""
from typing import List, Dict
import torch

class Collator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    def __call__(self, batch_chunks):
        max_len = max(len(c) for c in batch_chunks)
        ids = []
        for c in batch_chunks:
            if len(c) == max_len:
                ids.append(c)
            else:
                ids.append(c + [self.pad_token_id] * (max_len - len(c)))
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "labels": t.clone(), "attention_mask": torch.ones_like(t, dtype=torch.bool)}