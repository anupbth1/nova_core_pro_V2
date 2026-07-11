"""
Chunking Strategies
"""
from abc import ABC, abstractmethod
from typing import Iterator, List

class Chunker(ABC):
    @abstractmethod
    def chunk(self, tokens: List[int]) -> Iterator[List[int]]: pass

class SlidingWindowChunker(Chunker):
    def __init__(self, block_size: int, overlap: bool = True):
        self.block_size = block_size
        self.stride = block_size // 2 if overlap else block_size
    def chunk(self, tokens):
        if len(tokens) < self.block_size: return
        for i in range(0, len(tokens) - self.block_size + 1, self.stride):
            yield tokens[i:i + self.block_size]