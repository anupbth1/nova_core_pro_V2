"""
Data Backends
"""
from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
import random

class BaseBackend(ABC):
    @abstractmethod
    def iter_documents(self) -> Iterator[Dict[str, Any]]:
        pass
    def supports_random_access(self) -> bool:
        return False
    def supports_length(self) -> bool:
        return False
    def __len__(self) -> int:
        raise NotImplementedError()
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

class HFArrowBackend(BaseBackend):
    def __init__(self, dataset_name, subset=None, split="train", text_column="text", max_samples=None):
        from datasets import load_dataset
        kwargs = {"split": split}
        if subset: kwargs["name"] = subset
        self.dataset = load_dataset(dataset_name, **kwargs, streaming=False)
        self.text_column = text_column
        self._len = min(max_samples or len(self.dataset), len(self.dataset))
    def iter_documents(self):
        for i in range(self._len):
            yield self.dataset[i]
    def supports_random_access(self): return True
    def supports_length(self): return True
    def __len__(self): return self._len
    def __getitem__(self, idx):
        if idx >= self._len: raise IndexError()
        return self.dataset[idx]

class HFStreamingBackend(BaseBackend):
    def __init__(self, dataset_name, subset=None, split="train", text_column="text", shuffle_buffer=10000):
        from datasets import load_dataset
        kwargs = {"split": split, "streaming": True}
        if subset: kwargs["name"] = subset
        self.dataset = load_dataset(dataset_name, **kwargs)
        self.text_column = text_column
        self.shuffle_buffer = shuffle_buffer
    def iter_documents(self):
        buffer = []
        for ex in self.dataset:
            buffer.append(ex)
            if len(buffer) >= self.shuffle_buffer:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)
        random.shuffle(buffer)
        yield from buffer