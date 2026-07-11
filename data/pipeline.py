"""
Data Pipeline
"""
from torch.utils.data import DataLoader
from .config import PipelineConfig
from .backend import HFArrowBackend, HFStreamingBackend
from .tokenizer import CharTokenizer, HFTokenizer
from .filters import DocumentFilter, MinLengthFilter, MaxLengthFilter
from .chunker import SlidingWindowChunker
from .collator import Collator
from .datasets import MapDataset, StreamingDataset

class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backend = self._create_backend()
        self.filter = self._create_filter()
        self.tokenizer = self._create_tokenizer()
        self.chunker = self._create_chunker()
        self.collator = Collator(pad_token_id=self.tokenizer.pad_token_id)
        self.dataset = self._create_dataset()
    def _create_backend(self):
        c = self.config.backend
        if self.config.streaming:
            return HFStreamingBackend(c.dataset_name, c.subset, c.split, c.text_column, c.shuffle_buffer)
        return HFArrowBackend(c.dataset_name, c.subset, c.split, c.text_column, c.max_samples)
    def _create_filter(self):
        c = self.config.filter
        df = DocumentFilter()
        if c.min_text_len is not None:
            df.add(MinLengthFilter(c.min_text_len, self.config.backend.text_column))
        if c.max_text_len is not None:
            df.add(MaxLengthFilter(c.max_text_len, self.config.backend.text_column))
        return df
    def _create_tokenizer(self):
        c = self.config.tokenizer
        return HFTokenizer(c.tokenizer_name, c.add_pad) if c.type == "hf" else CharTokenizer()
    def _create_chunker(self):
        c = self.config.chunker
        return SlidingWindowChunker(c.block_size, c.overlap)
    def _create_dataset(self):
        if self.config.streaming:
            return StreamingDataset(self.backend, self.tokenizer, self.chunker, self.filter)
        return MapDataset(self.backend, self.tokenizer, self.chunker, self.filter, self.config.tokenizer.cache_size)
    def create_dataloader(self):
        c = self.config.loader
        return DataLoader(
            self.dataset, batch_size=c.batch_size,
            shuffle=False if self.config.streaming else True,
            num_workers=c.num_workers, collate_fn=self.collator,
            pin_memory=c.pin_memory,
            persistent_workers=c.persistent_workers and c.num_workers > 0,
            prefetch_factor=c.prefetch_factor if c.num_workers > 0 else None)