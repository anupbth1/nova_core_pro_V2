"""
Data Pipeline Configuration
Dataclasses for all pipeline components.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BackendConfig:
    type: str = "hf_arrow"
    dataset_name: str = "roneneldan/TinyStories"
    subset: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    shuffle_buffer: int = 10000
    max_samples: Optional[int] = None

@dataclass
class TokenizerConfig:
    type: str = "char"
    tokenizer_name: str = "gpt2"
    add_pad: bool = True
    cache_size: int = 512

@dataclass
class ChunkConfig:
    type: str = "sliding_window"
    block_size: int = 256
    overlap: bool = True

@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    seed: int = 42

@dataclass
class FilterConfig:
    min_text_len: Optional[int] = None
    max_text_len: Optional[int] = None

@dataclass
class PipelineConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    chunker: ChunkConfig = field(default_factory=ChunkConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    streaming: bool = False