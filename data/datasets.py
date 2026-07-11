"""
Dataset Implementations
"""
import random
from functools import lru_cache
import torch
from torch.utils.data import Dataset, IterableDataset

class MapDataset(Dataset):
    def __init__(self, backend, tokenizer, chunker, filter_fn=None, cache_size=512):
        self.backend = backend
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.filter_fn = filter_fn or (lambda doc: True)
        self.doc_indices = []
        if backend.supports_length():
            for i in range(len(backend)):
                if self.filter_fn(backend[i]):
                    self.doc_indices.append(i)
        self._tokenize = lru_cache(maxsize=cache_size)(self._tokenize_doc)
    def _tokenize_doc(self, idx):
        doc = self.backend[idx]
        text = doc.get(self.backend.text_column, "") or doc.get(list(doc.keys())[0], "")
        return self.tokenizer.encode(text)
    def __len__(self):
        return len(self.doc_indices) if self.doc_indices else len(self.backend)
    def __getitem__(self, idx):
        doc_idx = random.choice(self.doc_indices) if self.doc_indices else (idx % len(self.backend))
        tokens = self._tokenize(doc_idx)
        bs = self.chunker.block_size
        if len(tokens) > bs:
            s = random.randint(0, len(tokens) - bs)
            return tokens[s:s + bs]
        return tokens + [self.tokenizer.pad_token_id] * (bs - len(tokens))

class StreamingDataset(IterableDataset):
    def __init__(self, backend, tokenizer, chunker, filter_fn=None):
        self.backend = backend
        self.tokenizer = tokenizer
        self.chunker = chunker
        self.filter_fn = filter_fn or (lambda doc: True)
    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        nw = wi.num_workers if wi else 1
        for i, doc in enumerate(self.backend.iter_documents()):
            if i % nw != wid: continue
            if not self.filter_fn(doc): continue
            text = doc.get(self.backend.text_column, "") or doc.get(list(doc.keys())[0], "")
            if not text: continue
            yield from self.chunker.chunk(self.tokenizer.encode(text))