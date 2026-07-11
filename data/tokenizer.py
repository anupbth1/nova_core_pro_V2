"""
Tokenizer Interface and Implementations
"""
from abc import ABC, abstractmethod
from typing import List

class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[int]: pass
    @abstractmethod
    def decode(self, ids: List[int]) -> str: pass
    @property
    @abstractmethod
    def vocab_size(self) -> int: pass
    @property
    @abstractmethod
    def pad_token_id(self) -> int: pass
    @property
    @abstractmethod
    def eos_token_id(self) -> int: pass

class CharTokenizer(BaseTokenizer):
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{} \n\t-_=/\\@#$%^&*~|<>+"
        self._char_map = {c: i + 1 for i, c in enumerate(chars)}
        self._char_map["<PAD>"] = 0
        self._vocab_size = len(chars) + 1
    def encode(self, text): return [self._char_map.get(c, 0) for c in text]
    def decode(self, ids):
        rev = {v: k for k, v in self._char_map.items()}
        return "".join(rev.get(i, "") for i in ids)
    @property
    def vocab_size(self): return self._vocab_size
    @property
    def pad_token_id(self): return 0
    @property
    def eos_token_id(self): return 0

class HFTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_name="gpt2", add_pad=True):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if add_pad and self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token or "[PAD]"
    def encode(self, text): return self._tokenizer.encode(text)
    def decode(self, ids): return self._tokenizer.decode(ids)
    @property
    def vocab_size(self): return self._tokenizer.vocab_size
    @property
    def pad_token_id(self): return self._tokenizer.pad_token_id or 0
    @property
    def eos_token_id(self): return self._tokenizer.eos_token_id or 0