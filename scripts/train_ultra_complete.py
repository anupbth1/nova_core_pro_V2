"""
============================================================================
NovaLM-ULTRA v1.0 - COMPLETE TRAINING PIPELINE
============================================================================
The ULTIMATE script with EVERY parameter explained.
Train on ANY HuggingFace dataset with ANY configuration.

Author: NovaLM Team
License: MIT

============================================================================
EXAMPLES:
============================================================================

# ===== QUICK TEST (1 epoch, tiny model) =====
python train_ultra_complete.py --dataset roneneldan/TinyStories --model-name "TestNova" --dim 256 --layers 4 --epochs 1

# ===== SMALL LLM (balanced config) =====
python train_ultra_complete.py --dataset roneneldan/TinyStories --model-name "MiniNova" --dim 512 --layers 8 --heads 16 --epochs 3

# ===== MEDIUM LLM (with HF tokenizer) =====
python train_ultra_complete.py --dataset HuggingFaceFW/fineweb --dataset-subset sample-10BT --model-name "NovaMedium" --dim 768 --layers 12 --heads 12 --kv-heads 4 --vocab-size 32000 --tokenizer hf --hf-tokenizer gpt2 --epochs 5 --batch-size 16 --block-size 512 --mixed-precision fp16 --save-every 5000

# ===== LARGE LLM (multi-GPU) =====
python train_ultra_complete.py --dataset HuggingFaceFW/fineweb-edu --model-name "NovaLarge" --dim 2048 --layers 24 --heads 32 --kv-heads 8 --vocab-size 50277 --tokenizer hf --hf-tokenizer meta-llama/Llama-2-7b-hf --epochs 10 --batch-size 8 --block-size 2048 --multi-gpu --gpu-ids 0,1,2,3 --mixed-precision bf16 --wandb --wandb-project nova-llm

# ===== ULTRA LLM (max power) =====
python train_ultra_complete.py --dataset HuggingFaceFW/fineweb-edu --model-name "NovaUltra" --dim 4096 --layers 64 --heads 64 --kv-heads 8 --vocab-size 50277 --weight-share 16 --block-size 8192 --multi-gpu --mixed-precision bf16 --compile --wandb

# ===== CUSTOM DATASET (your own data) =====
python train_ultra_complete.py --dataset my_dataset --text-column content --model-name "MyCustomLLM" --dim 512 --layers 8 --epochs 5

# ===== WITH FILTERING =====
python train_ultra_complete.py --dataset roneneldan/TinyStories --max-samples 50000 --min-text-length 100 --max-text-length 10000 --model-name "FilteredNova"

# ===== RESUME TRAINING =====
python train_ultra_complete.py --resume checkpoints/MyFirstLLM/latest.pt --epochs 10

# ===== GENERATE TEXT AFTER TRAINING =====
python train_ultra_complete.py --resume checkpoints/MyFirstLLM/best_model.pt --generate --prompt "Once upon a time"
"""

import sys
import os
import argparse
import math
import time
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import threading

# Add project to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# Import data pipeline
try:
    from data import Pipeline, PipelineConfig, BackendConfig, TokenizerConfig, ChunkConfig, LoaderConfig, FilterConfig
    HAS_PIPELINE = True
except ImportError as e:
    HAS_PIPELINE = False
    print(f"⚠️  Data pipeline not found ({e}). Using fallback HFDatasetWrapper.")

# Try importing NovaLM-ULTRA (handle hyphen in folder name)
try:
    import importlib
    ultra_config = importlib.import_module("ultra_config")
    ultra_model = importlib.import_module("ultra_model")
    UltraConfig = ultra_config.UltraConfig
    create_ultra_model = ultra_model.create_ultra_model
    NovaUltraModel = ultra_model.NovaUltraModel
    HAS_NOVA = True
    print("✅ NovaLM-ULTRA module loaded successfully!")
except ImportError as e:
    HAS_NOVA = False
    print(f"⚠️  NovaLM-ULTRA not found ({e}). Will create model manually.")

# ============================================================================
# SECTION 1: COMPLETE CONFIGURATION WITH ALL PARAMETERS
# ============================================================================

@dataclass
class AllConfig:
    """
    ULTIMATE CONFIGURATION - Every parameter you could ever need.
    
    === MODEL ARCHITECTURE ===
    dim:          Model dimension (default: 512). Higher = more capacity.
                 [256=fast, 512=small, 768=medium, 1024=large, 2048=xl, 4096=xxl]
    num_layers:   Number of transformer layers (default: 8). More = deeper.
                 [4=shallow, 8=medium, 12=deep, 24=very deep, 64=ultra deep]
    num_heads:    Number of attention heads (default: 16). More = finer attention.
                 Must divide dim evenly. Common: dim/64 or dim/128.
    n_kv_heads:   KV heads for GQA (default: auto). 4x memory savings vs full MHA.
                 [1=max savings, 4=good, 8=balanced, 16=full quality]
    head_dim:     Dimension per head (default: dim/num_heads). Usually 64 or 128.
    ffn_dim:      FFN hidden dimension (default: dim*4). Usually 4x dim.
                 [dim*2=fast, dim*3.5=balanced, dim*4=standard, dim*8=SwiGLU]
    vocab_size:   Vocabulary size (default: auto from tokenizer).
                 [1000=char, 10000=small, 32000=medium, 50277=gpt2, 128000=llama3]
    max_seq_len:  Maximum sequence length (default: 1024).
                 [256=short, 512=medium, 1024=long, 2048=xl, 8192=ultra, 131072=max]
    weight_share: Weight sharing iterations (default: 4).
                 Nx effective depth with same params. [1=no sharing, 4=good, 8=deep, 16=ultra]
    weight_tying: Share embedding and output head weights (default: True). Saves vocab*dim params.
    """
    # Model architecture
    dim: int = 512
    num_layers: int = 8
    num_heads: int = 16
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    ffn_dim: Optional[int] = None
    vocab_size: int = 10000
    max_seq_len: int = 1024
    weight_share: int = 4
    weight_tying: bool = True
    
    """
    === ARCHITECTURE COMPONENTS ===
    attn_type:    Attention type (default: "gqa"). ["gqa"=standard, "mla"=deepseek latent]
    use_rwkv:     Use RWKV-7 TimeMix linear attention (default: True). Fast CPU inference.
    rwkv_freq:    Apply RWKV every N layers (default: 2). 
    use_ssm:      Use Selective SSM Mamba-style (default: True). Long-range dependencies.
    ssm_freq:     Apply SSM every N layers (default: 4).
    ssm_state:    SSM state dimension (default: 16). Higher = more memory.
    use_memory:   Use Titans Neural Memory (default: True). Surprise-based storage.
    memory_slots: Number of memory slots (default: 256). More = more capacity.
    memory_topk:  Top-k memory retrieval (default: 16).
    memory_freq:  Apply memory every N layers (default: 4).
    use_router:   Use Adaptive Conditional Router (default: True). Per-token compute.
    router_dim:   Router hidden dimension (default: 256).
    use_rope:     Use Rotary Position Embeddings (default: True). From Llama/Mistral.
    """
    # Architecture components
    attn_type: str = "gqa"
    use_rwkv: bool = True
    rwkv_freq: int = 2
    use_ssm: bool = True
    ssm_freq: int = 4
    ssm_state: int = 16
    use_memory: bool = True
    memory_slots: int = 256
    memory_topk: int = 16
    memory_freq: int = 4
    use_router: bool = True
    router_dim: int = 256
    use_rope: bool = True
    
    """
    === MODEL NAMING ===
    model_name:   Name for your LLM (default: "NovaLM-ULTRA").
                 Used for checkpoint directories and saving.
    """
    model_name: str = "NovaLM-ULTRA"
    
    """
    === DATASET CONFIGURATION ===
    dataset:         HuggingFace dataset name (default: "roneneldan/TinyStories").
                    [TinyStories, fineweb, fineweb-edu, c4, wikitext, etc.]
    dataset_subset:  Dataset subset/config (default: None). 
                    [sample-10BT, default, etc.]
    dataset_split:   Dataset split to use (default: "train").
                    ["train", "validation", "test", "train[:10%]", etc.]
    val_dataset:     Separate validation dataset (default: same as train).
    val_split:       Validation split name (default: "validation").
    val_size:        Validation fraction if no val split (default: 0.1).
    text_column:     Column containing text (default: "text").
                    ["text", "content", "article", "code", etc.]
    max_samples:     Maximum samples to load (default: None = all).
                    Use for testing: --max-samples 10000
    min_text_len:    Minimum text length filter (default: None).
                    Skip shorter texts: --min-text-length 100
    max_text_len:    Maximum text length filter (default: None).
                    Skip longer texts: --max-text-length 100000
    streaming:       Use streaming mode for huge datasets (default: False).
                    Saves RAM but slower for first epoch.
    cache_dir:       Dataset cache directory (default: None = HF default).
    """
    # Dataset
    dataset: str = "roneneldan/TinyStories"
    dataset_subset: Optional[str] = None
    dataset_split: str = "train"
    val_dataset: Optional[str] = None
    val_split: str = "validation"
    val_size: float = 0.1
    text_column: str = "text"
    max_samples: Optional[int] = None
    min_text_len: Optional[int] = None
    max_text_len: Optional[int] = None
    streaming: bool = False
    cache_dir: Optional[str] = None
    
    """
    === TOKENIZER CONFIGURATION ===
    tokenizer:      Tokenizer type (default: "char"). 
                   ["char"=simple, "hf"=huggingface]
    hf_tokenizer:   HF tokenizer name (default: "gpt2").
                   ["gpt2", "bert-base-uncased", "meta-llama/Llama-2-7b-hf", etc.]
    add_pad_token:  Add padding token if missing (default: True).
    """
    # Tokenizer
    tokenizer: str = "char"
    hf_tokenizer: str = "gpt2"
    add_pad_token: bool = True
    
    """
    === TRAINING HYPERPARAMETERS ===
    epochs:         Number of training epochs (default: 3).
                   [1=test, 3=quick, 10=standard, 100=full]
    batch_size:     Batch size per device (default: 8).
                   [4=small GPU, 8=default, 16=good, 32=large, 64=huge]
    block_size:     Sequence length per sample (default: 256).
                   [128=fast, 256=default, 512=good, 1024=long, 2048=xl, 8192=ultra]
    lr:             Peak learning rate (default: 3e-4).
                   [1e-4=stable, 3e-4=default, 5e-4=fast, 1e-3=aggressive]
    min_lr:         Minimum learning rate (default: 3e-5).
                   Usually 10% of peak lr.
    warmup_steps:   Linear warmup steps (default: 100).
                   [0=none, 100=default, 1000=slow warmup]
    weight_decay:   AdamW weight decay (default: 0.1).
                   [0.01=light, 0.1=standard, 1.0=strong]
    beta1:          Adam beta1 (default: 0.9). Momentum.
    beta2:          Adam beta2 (default: 0.95). RMS prop.
    grad_clip:      Max gradient norm (default: 1.0).
                   [0.5=strict, 1.0=default, 5.0=loose, None=off]
    dropout:        Dropout rate (default: 0.0).
                   [0.0=none, 0.1=light, 0.2=standard]
    label_smoothing: Label smoothing for loss (default: 0.0).
                    [0.0=none, 0.1=good for generalization]
    lr_scheduler:   Learning rate schedule (default: "cosine").
                   ["cosine", "linear", "constant", "cosine_restarts"]
    """
    # Training
    epochs: int = 3
    batch_size: int = 8
    block_size: int = 256
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float = 0.0
    label_smoothing: float = 0.0
    lr_scheduler: str = "cosine"
    
    """
    === HARDWARE CONFIGURATION ===
    device:          Device to use (default: "auto").
                    ["auto", "cpu", "cuda", "mps"]
    num_workers:     DataLoader workers (default: 0 for Windows).
                    [0=Windows safe, 2=good, 4=fast, 8=max]
    mixed_precision: Mixed precision mode (default: "fp16").
                    ["fp16"=fast GPU, "bf16"=better GPU, "no"=full fp32]
    compile:         Use torch.compile (default: False). 20-30% faster.
                    Requires PyTorch 2.0+. May have issues on Windows.
    multi_gpu:       Use all available GPUs (default: False).
    gpu_ids:         Specific GPU IDs to use (default: None = all).
                    ["0", "0,1", "0,1,2,3"]
    cpu_threads:     Number of CPU threads for OMP (default: None = auto).
                    Set higher for CPU training: --cpu-threads 8
    """
    # Hardware
    device: str = "auto"
    num_workers: int = 0
    mixed_precision: str = "fp16"
    compile: bool = False
    multi_gpu: bool = False
    gpu_ids: Optional[str] = None
    cpu_threads: Optional[int] = None
    
    """
    === LOGGING & CHECKPOINTS ===
    save_dir:       Directory to save checkpoints (default: "checkpoints").
    save_every:     Save checkpoint every N steps (default: 1000).
    save_optimizer: Save optimizer state (default: True). Needed for resume.
    log_interval:   Log every N steps (default: 10).
    wandb:          Use Weights & Biases logging (default: False).
    wandb_project:  W&B project name (default: "novaultra").
    wandb_run:      W&B run name (default: model_name + timestamp).
    eval_every:     Evaluate every N steps (default: 500).
    eval_steps:     Number of evaluation steps (default: 100).
    """
    # Logging
    save_dir: str = "checkpoints"
    save_every: int = 1000
    save_optimizer: bool = True
    log_interval: int = 10
    wandb: bool = False
    wandb_project: str = "novaultra"
    wandb_run: Optional[str] = None
    eval_every: int = 500
    eval_steps: int = 100
    
    """
    === RESUME & GENERATION ===
    resume:         Path to checkpoint to resume from (default: None).
    reset_optimizer: Reset optimizer on resume (default: False).
    reset_scheduler: Reset scheduler on resume (default: False).
    generate:       Generate text after training (default: False).
    prompt:         Prompt for generation (default: "Once upon a time").
    gen_tokens:     Number of tokens to generate (default: 100).
    gen_temperature: Generation temperature (default: 0.7).
    gen_top_k:      Top-k sampling (default: 50).
    gen_top_p:      Top-p nucleus sampling (default: 0.9).
    """
    # Resume & Generation
    resume: Optional[str] = None
    reset_optimizer: bool = False
    reset_scheduler: bool = False
    generate: bool = False
    prompt: str = "Once upon a time"
    gen_tokens: int = 100
    gen_temperature: float = 0.7
    gen_top_k: int = 50
    gen_top_p: float = 0.9


# ============================================================================
# SECTION 2: TOKENIZER (Character + HuggingFace)
# ============================================================================

def create_char_tokenizer():
    """Create a simple character-level tokenizer."""
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}\"' \n\t-_=/\\@#$%^&*~`|<>+"
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad/unk
    char_to_idx["<PAD>"] = 0
    char_to_idx["<UNK>"] = 0
    
    class SimpleTokenizer:
        def __init__(self, char_map, vocab_size):
            self.char_map = char_map
            self._vocab_size = vocab_size + 1  # +1 for padding
            self.pad_token_id = 0
            self.eos_token_id = 0
            self.bos_token_id = 0
        
        @property
        def vocab_size(self):
            return self._vocab_size
        
        def encode(self, text: str) -> List[int]:
            return [self.char_map.get(c, 0) for c in text]
        
        def decode(self, ids: List[int]) -> str:
            rev_map = {v: k for k, v in self.char_map.items()}
            return "".join(rev_map.get(i, "") for i in ids)
    
    return SimpleTokenizer(char_to_idx, len(chars))


def create_hf_tokenizer(tokenizer_name: str = "gpt2", add_pad: bool = True):
    """Create a HuggingFace tokenizer."""
    from transformers import AutoTokenizer
    print(f"  Loading HF tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if add_pad and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token or "[CLS]"
    print(f"  Vocab size: {tokenizer.vocab_size}")
    return tokenizer


# ============================================================================
# SECTION 3: DATASET HANDLING (With ALL Filters)
# ============================================================================

class HFDatasetWrapper(Dataset):
    """
    Complete HuggingFace Dataset wrapper with:
    - Text filtering (min/max length)
    - Tokenization
    - Sliding window chunking
    - Overlapping chunks (block_size // 2 stride)
    """
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        block_size: int = 256,
        text_column: str = "text",
        min_text_len: Optional[int] = None,
        max_text_len: Optional[int] = None,
        max_samples: Optional[int] = None,
        overlap: bool = True,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        
        # Apply filters
        if min_text_len or max_text_len or max_samples:
            dataset = []
            for i, example in enumerate(hf_dataset):
                if max_samples and i >= max_samples:
                    break
                text = example.get(text_column, "")
                if min_text_len and len(text) < min_text_len:
                    continue
                if max_text_len and len(text) > max_text_len:
                    continue
                dataset.append(example)
            hf_dataset = dataset
            print(f"  After filtering: {len(hf_dataset)} samples")
        
        # Tokenize and chunk
        self.tokens = []
        stride = block_size // 2 if overlap else block_size
        
        print("  Tokenizing dataset...")
        for example in tqdm(hf_dataset, desc="Tokenizing"):
            text = example.get(text_column, "")
            if not text:
                continue
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size, stride):
                chunk = tokens[i:i + block_size]
                if len(chunk) == block_size:
                    self.tokens.append(chunk)
        
        print(f"  Created {len(self.tokens):,} training chunks")
        if self.tokens:
            self.data = torch.tensor(self.tokens, dtype=torch.long)
        else:
            self.data = torch.zeros((1, block_size), dtype=torch.long)
            print("  ⚠️  No valid chunks! Using dummy data.")
    
    def __len__(self):
        return len(self.tokens) if self.tokens else 1
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {"input_ids": chunk, "labels": chunk}


# ============================================================================
# SECTION 4: MODEL CREATION (NovaLM-ULTRA or Custom)
# ============================================================================

def create_model(config: AllConfig) -> nn.Module:
    """
    Create the model based on configuration.
    Uses NovaLM-ULTRA if available, otherwise creates a minimal version.
    """
    print(f"\n{'='*60}")
    print(f"🤖 CREATING MODEL: {config.model_name}")
    print(f"{'='*60}")
    
    if HAS_NOVA:
        # Use NovaLM-ULTRA architecture
        ultra_config = UltraConfig(
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            n_kv_heads=config.n_kv_heads,
            ffn_dim=config.ffn_dim,
            vocab_size=config.vocab_size,
            max_seq_len=config.block_size,
            weight_tying=config.weight_tying,
            weight_share_iterations=config.weight_share,
            use_adaptive_router=config.use_router,
            router_hidden_dim=config.router_dim,
            attn_type=config.attn_type,
            use_rope=config.use_rope,
            use_rwkv=config.use_rwkv,
            rwkv_frequency=config.rwkv_freq,
            use_titans_memory=config.use_memory,
            memory_slots=config.memory_slots,
            memory_top_k=config.memory_topk,
            memory_frequency=config.memory_freq,
            use_ssm_scan=config.use_ssm,
            ssm_state_dim=config.ssm_state,
            ssm_frequency=config.ssm_freq,
            use_fp32=(config.mixed_precision == "no"),
            use_bf16=(config.mixed_precision == "bf16"),
            use_fp16=(config.mixed_precision == "fp16"),
            batch_size=config.batch_size,
            learning_rate=config.lr,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            grad_clip=config.grad_clip,
        )
        model = create_ultra_model(ultra_config)
        print(f"  ✅ Created NovaLM-ULTRA model!")
    else:
        # Create minimal transformer as fallback
        print("  ⚠️  NovaLM-ULTRA not available! Creating minimal transformer.")
        model = MinimalTransformer(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim or config.dim * 4,
            max_seq_len=config.block_size,
            dropout=config.dropout,
        )
        print(f"  ✅ Created minimal transformer model!")
    
    # Print model info
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    if hasattr(model, 'get_effective_params'):
        print(f"  Effective parameters: {model.get_effective_params():,}")
    
    return model


class MinimalTransformer(nn.Module):
    """Minimal transformer fallback (if NovaLM-ULTRA not available)."""
    def __init__(self, vocab_size, dim=512, num_layers=8, num_heads=16, 
                 ffn_dim=2048, max_seq_len=1024, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=num_heads, dim_feedforward=ffn_dim,
                dropout=dropout, activation=F.silu, batch_first=True,
                norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, x, state=None, return_state=False):
        # Create causal mask
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits, None if return_state else None
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.7, 
                 top_k=50, top_p=0.9, repetition_penalty=1.1):
        self.eval()
        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits, _ = self(generated[:, -self.dim:])
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < vals[:, -1:]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cum_probs > top_p
                sorted_idx_to_remove[:, 1:] = sorted_idx_to_remove[:, :-1].clone()
                sorted_idx_to_remove[:, 0] = False
                indices_to_remove = sorted_idx_to_remove.scatter(1, sorted_idx, sorted_idx_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == 0:
                break
        return generated


# ============================================================================
# SECTION 5: TRAINING LOOP (Complete with ALL features)
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: AllConfig,
) -> nn.Module:
    """
    Complete training loop with:
    - AdamW with weight decay separation
    - Linear warmup + Cosine annealing
    - Mixed precision (fp16/bf16)
    - Gradient clipping
    - Multi-GPU support
    - WandB logging
    - Checkpoint saving/resuming
    - Evaluation loop
    - Text generation demo
    """
    # ===== DEVICE SETUP =====
    device_str = config.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    
    print(f"📱 Device: {device_str}")
    
    # CPU threads
    if config.cpu_threads and device_str == "cpu":
        torch.set_num_threads(config.cpu_threads)
        print(f"   CPU threads: {config.cpu_threads}")
    
    # Multi-GPU
    use_ddp = False
    if config.multi_gpu and device_str == "cuda":
        if config.gpu_ids:
            gpu_list = [int(x) for x in config.gpu_ids.split(",")]
            print(f"   Using GPUs: {gpu_list}")
        else:
            gpu_list = list(range(torch.cuda.device_count()))
            print(f"   Using GPUs: {gpu_list}")
        
        if len(gpu_list) > 1:
            use_ddp = True
    
    # Move model to device
    device = torch.device(device_str)
    model = model.to(device)
    
    if use_ddp:
        # DataParallel fallback (simpler than DDP)
        model = nn.DataParallel(model, device_ids=gpu_list)
        print(f"   Using DataParallel with {len(gpu_list)} GPUs")
    
    # ===== MIXED PRECISION =====
    scaler = None
    amp_dtype = None
    amp_device = device_str
    
    if config.mixed_precision == "fp16" and device_str == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        amp_dtype = torch.float16
        print("   Mixed precision: fp16")
    elif config.mixed_precision == "bf16" and device_str == "cuda":
        amp_dtype = torch.bfloat16
        print("   Mixed precision: bf16")
    elif config.mixed_precision == "fp16" and device_str == "mps":
        amp_dtype = torch.float16
        print("   Mixed precision: fp16 (MPS)")
    else:
        print("   Mixed precision: off (fp32)")
    
    # ===== COMPILE =====
    if config.compile and device_str == "cuda" and not use_ddp:
        try:
            model = torch.compile(model)
            print("   Using torch.compile (20-30% faster)")
        except Exception as e:
            print(f"   torch.compile failed: {e}. Skipping.")
    
    # ===== OPTIMIZER =====
    # Separate weight decay
    decay_params = []
    no_decay_params = []
    raw_model = model.module if use_ddp else model
    
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )
    
    # ===== LR SCHEDULER =====
    total_steps = len(train_loader) * config.epochs
    
    if config.lr_scheduler == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, 
                          total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - config.warmup_steps, 
                                   eta_min=config.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[config.warmup_steps])
    elif config.lr_scheduler == "linear":
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, 
                          total_iters=config.warmup_steps)
        linear = LinearLR(optimizer, start_factor=1.0, end_factor=config.min_lr / config.lr,
                          total_iters=total_steps - config.warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, linear],
                                 milestones=[config.warmup_steps])
    elif config.lr_scheduler == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.warmup_steps * 4,
                                                T_mult=2, eta_min=config.min_lr)
    else:  # constant
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, 
                          total_iters=config.warmup_steps)
        scheduler = warmup
    
    # ===== RESUME FROM CHECKPOINT =====
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    
    if config.resume and os.path.exists(config.resume):
        print(f"📂 Resuming from: {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model_state"])
        
        if not config.reset_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if not config.reset_scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler and "scaler_state" in checkpoint and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        print(f"   Resumed at epoch {start_epoch}, step {global_step}")
    
    # ===== SAVE DIRECTORY =====
    save_dir = Path(config.save_dir) / config.model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints saved to: {save_dir}")
    
    # ===== WANDB =====
    if config.wandb:
        try:
            import wandb
            run_name = config.wandb_run or f"{config.model_name}-{datetime.now():%Y%m%d_%H%M%S}"
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "dim": config.dim,
                    "num_layers": config.num_layers,
                    "num_heads": config.num_heads,
                    "n_kv_heads": config.n_kv_heads,
                    "vocab_size": config.vocab_size,
                    "batch_size": config.batch_size,
                    "block_size": config.block_size,
                    "epochs": config.epochs,
                    "lr": config.lr,
                    "weight_decay": config.weight_decay,
                    "model_name": config.model_name,
                    "dataset": config.dataset,
                    "tokenizer": config.tokenizer,
                }
            )
            print("📊 WandB logging enabled")
        except ImportError:
            print("⚠️  wandb not installed. Skipping.")
            config.wandb = False
    
    # ===== TRAINING LOOP =====
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING STARTED")
    print(f"   Model: {config.model_name}")
    print(f"   Dataset: {config.dataset}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Block size: {config.block_size}")
    print(f"   Total steps: {total_steps}")
    print(f"   Epochs: {config.epochs}")
    print(f"{'='*60}\n")
    
    raw_model.train()
    total_start_time = time.time()
    
    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        label_smoothing=config.label_smoothing,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            elif amp_dtype:
                with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        label_smoothing=config.label_smoothing,
                    )
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
                optimizer.step()
            else:
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    label_smoothing=config.label_smoothing,
                )
                loss.backward()
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), config.grad_clip)
                optimizer.step()
            
            # Step scheduler
            if config.lr_scheduler != "constant":
                scheduler.step()
            
            global_step += 1
            epoch_loss += loss.item()
            epoch_tokens += input_ids.numel()
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
            })
            
            # Logging
            if global_step % config.log_interval == 0:
                elapsed = time.time() - total_start_time
                tok_per_sec = epoch_tokens / (time.time() - epoch_start + 1e-8)
                
                if config.wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": current_lr,
                        "train/epoch": epoch + (batch_idx + 1) / len(train_loader),
                        "train/tokens_per_sec": tok_per_sec,
                        "train/step": global_step,
                    })
            
            # Save checkpoint
            if global_step % config.save_every == 0:
                ckpt_path = save_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(raw_model, optimizer, scheduler, scaler, 
                               epoch, global_step, best_loss, ckpt_path, config)
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch+1} completed:")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        print(f"   Tokens/sec: {epoch_tokens / (epoch_time + 1e-8):.0f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save epoch checkpoint
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        save_checkpoint(raw_model, optimizer, scheduler, scaler,
                       epoch, global_step, best_loss, ckpt_path, config)
        
        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            torch.save({
                "model_state": raw_model.state_dict(),
                "config": {
                    "dim": config.dim, "num_layers": config.num_layers,
                    "num_heads": config.num_heads, "n_kv_heads": config.n_kv_heads,
                    "vocab_size": config.vocab_size, "ffn_dim": config.ffn_dim or config.dim * 4,
                    "weight_share": config.weight_share,
                    "model_name": config.model_name,
                },
            }, best_path)
            print(f"🏆 New best model! Saved to: {best_path}")
        
        # Save latest
        latest_path = save_dir / "latest.pt"
        save_checkpoint(raw_model, optimizer, scheduler, scaler,
                       epoch, global_step, best_loss, latest_path, config)
        
        # Evaluation
        if val_loader and (global_step % config.eval_every < len(train_loader) or config.epochs == 1):
            raw_model.eval()
            val_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validating", leave=False):
                    if val_steps >= config.eval_steps:
                        break
                    val_ids = val_batch["input_ids"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    val_logits, _ = model(val_ids)
                    v_loss = F.cross_entropy(
                        val_logits.view(-1, val_logits.size(-1)),
                        val_labels.view(-1),
                    )
                    val_loss += v_loss.item()
                    val_steps += 1
            
            val_loss /= max(val_steps, 1)
            print(f"   Validation Loss: {val_loss:.4f}")
            
            if config.wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/step": global_step})
            
            raw_model.train()
    
    # ===== TRAINING COMPLETE =====
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Model saved to: {save_dir}")
    print(f"{'='*60}")
    
    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save({
        "model_state": raw_model.state_dict(),
        "config": {
            "dim": config.dim, "num_layers": config.num_layers,
            "num_heads": config.num_heads, "n_kv_heads": config.n_kv_heads,
            "vocab_size": config.vocab_size, "ffn_dim": config.ffn_dim or config.dim * 4,
            "weight_share": config.weight_share,
            "model_name": config.model_name,
        },
    }, final_path)
    
    if config.wandb:
        import wandb
        wandb.finish()
    
    return raw_model


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, best_loss, path, config):
    """Save a complete checkpoint."""
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if config.save_optimizer else None,
        "scheduler_state": scheduler.state_dict() if config.save_optimizer else None,
        "scaler_state": scaler.state_dict() if (scaler and config.save_optimizer) else None,
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "model_name": config.model_name,
        "config": {
            "dim": config.dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "n_kv_heads": config.n_kv_heads,
            "vocab_size": config.vocab_size,
            "ffn_dim": config.ffn_dim or config.dim * 4,
            "weight_share": config.weight_share,
        },
    }
    torch.save(checkpoint, path)
    print(f"💾 Saved: {path}")


# ============================================================================
# SECTION 6: TEXT GENERATION
# ============================================================================

def generate_text(model, tokenizer, config: AllConfig):
    """Generate text using the trained model."""
    print(f"\n{'='*60}")
    print(f"✍️ GENERATING TEXT")
    print(f"{'='*60}")
    
    # Encode prompt
    prompt_ids = torch.tensor([tokenizer.encode(config.prompt)], dtype=torch.long)
    print(f"   Prompt: '{config.prompt}'")
    
    # Generate
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            prompt_ids,
            max_new_tokens=config.gen_tokens,
            temperature=config.gen_temperature,
            top_k=config.gen_top_k,
            top_p=config.gen_top_p,
        )
    
    # Decode
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(f"\n   Generated:")
    print(f"   {'─'*60}")
    print(f"   {output_text}")
    print(f"   {'─'*60}")
    
    return output_text


# ============================================================================
# SECTION 7: ARGUMENT PARSER (ALL parameters with Hindi help)
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with ALL parameters and explanations."""
    parser = argparse.ArgumentParser(
        description="""
╔══════════════════════════════════════════════════════════════╗
║     NovaLM-ULTRA v1.0 - COMPLETE TRAINING PIPELINE          ║
║     सभी पैरामीटर्स के साथ पूरा ट्रेनिंग स्क्रिप्ट            ║
╚══════════════════════════════════════════════════════════════╝

EXAMPLES:
  # Quick test (fast config, 1 epoch)
  python train_ultra_complete.py --dataset roneneldan/TinyStories --dim 256 --layers 4 --epochs 1
  
  # Full training with HF tokenizer
  python train_ultra_complete.py --dataset HuggingFaceFW/fineweb --dim 768 --layers 12 --heads 12 --tokenizer hf --hf-tokenizer gpt2 --epochs 5
  
  # Multi-GPU training
  python train_ultra_complete.py --dataset HuggingFaceFW/fineweb-edu --dim 2048 --layers 24 --heads 32 --multi-gpu --mixed-precision bf16
  
  # Custom dataset with filtering
  python train_ultra_complete.py --dataset my_dataset --text-column content --min-text-length 100 --max-samples 100000
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ===== MODEL ARCHITECTURE =====
    model_group = parser.add_argument_group("🏗️  MODEL ARCHITECTURE (मॉडल आर्किटेक्चर)")
    model_group.add_argument("--model-name", type=str, default="NovaLM-ULTRA",
                            help="Your LLM का नाम (default: NovaLM-ULTRA)")
    model_group.add_argument("--dim", type=int, default=512,
                            help="Model dimension - मॉडल का आकार (256=fast, 512=small, 768=medium, 1024=large, 2048=xl, 4096=xxl)")
    model_group.add_argument("--layers", type=int, default=8,
                            help="Number of layers - लेयर्स की संख्या (4=shallow, 8=medium, 12=deep, 24=very deep)")
    model_group.add_argument("--heads", type=int, default=16,
                            help="Attention heads - अटेंशन हेड्स (must divide dim, 4/8/16/32/64)")
    model_group.add_argument("--kv-heads", type=int, default=None,
                            help="KV heads for GQA - KV हेड्स (1=max savings, 4=good, 8=balanced, None=auto)")
    model_group.add_argument("--head-dim", type=int, default=None,
                            help="Dimension per head - हर हेड का आयाम (usually 64 or 128)")
    model_group.add_argument("--ffn-dim", type=int, default=None,
                            help="FFN dimension - FFN आयाम (default: dim*4, higher = more capacity)")
    model_group.add_argument("--vocab-size", type=int, default=10000,
                            help="Vocabulary size - वोकैब का आकार (1000=char, 10000=small, 32000=medium, 50277=gpt2)")
    model_group.add_argument("--block-size", type=int, default=256,
                            help="Sequence/block length - सीक्वेंस लंबाई (128=fast, 256=default, 512=good, 1024=long)")
    model_group.add_argument("--weight-share", type=int, default=4,
                            help="Weight sharing - वेट शेयरिंग (Nx depth at 1x params, 1=off, 4=good, 8=deep)")
    model_group.add_argument("--no-weight-tying", action="store_true",
                            help="Disable weight tying - वेट टाइइंग बंद करें (embed và head alag rakhne ke liye)")
    
    # ===== ARCHITECTURE COMPONENTS =====
    comp_group = parser.add_argument_group("🧩  ARCHITECTURE COMPONENTS (आर्किटेक्चर कम्पोनेंट्स)")
    comp_group.add_argument("--attn-type", type=str, default="gqa", choices=["gqa", "mla"],
                           help="Attention type (gqa=fast, mla=deepseek latent)")
    comp_group.add_argument("--use-rwkv", action="store_true",
                           help="Enable RWKV-7 TimeMix (default: on) - RWKV चालू करें")
    comp_group.add_argument("--no-rwkv", action="store_true",
                           help="Disable RWKV-7 TimeMix - RWKV बंद करें")
    comp_group.add_argument("--rwkv-freq", type=int, default=2,
                           help="RWKV every N layers (2=har doosri layer mein)")
    comp_group.add_argument("--use-ssm", action="store_true",
                           help="Enable Selective SSM (default: on) - SSM चालू करें")
    comp_group.add_argument("--no-ssm", action="store_true",
                           help="Disable Selective SSM (Mamba) - SSM बंद करें")
    comp_group.add_argument("--ssm-freq", type=int, default=4,
                           help="SSM every N layers")
    comp_group.add_argument("--ssm-state", type=int, default=16,
                           help="SSM state dimension - SSM स्टेट डायमेंशन")
    comp_group.add_argument("--use-memory", action="store_true",
                           help="Enable Titans Neural Memory (default: on) - मेमोरी चालू करें")
    comp_group.add_argument("--no-memory", action="store_true",
                           help="Disable Titans Neural Memory - मेमोरी बंद करें")
    comp_group.add_argument("--memory-slots", type=int, default=256,
                           help="Memory slots - मेमोरी स्लॉट्स (more = more capacity)")
    comp_group.add_argument("--memory-topk", type=int, default=16,
                           help="Top-k memory retrieval")
    comp_group.add_argument("--memory-freq", type=int, default=4,
                           help="Memory every N layers")
    comp_group.add_argument("--use-router", action="store_true",
                           help="Enable Adaptive Router (default: on) - राउटर चालू करें")
    comp_group.add_argument("--no-router", action="store_true",
                           help="Disable Adaptive Router - राउटर बंद करें")
    comp_group.add_argument("--router-dim", type=int, default=256,
                           help="Router hidden dimension")
    comp_group.add_argument("--use-rope", action="store_true",
                           help="Enable RoPE embeddings (default: on) - RoPE चालू करें")
    comp_group.add_argument("--no-rope", action="store_true",
                           help="Disable RoPE embeddings - RoPE बंद करें")
    
    # ===== DATASET =====
    data_group = parser.add_argument_group("📚  DATASET (डेटासेट)")
    data_group.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                           help="HF dataset name - डेटासेट का नाम (TinyStories, fineweb, c4, wikitext, etc.)")
    data_group.add_argument("--dataset-subset", type=str, default=None,
                           help="Dataset subset/config (sample-10BT, default, etc.)")
    data_group.add_argument("--dataset-split", type=str, default="train",
                           help="Dataset split (train, validation, test, train[:10%%])")
    data_group.add_argument("--text-column", type=str, default="text",
                           help="Text column name (text, content, article, code)")
    data_group.add_argument("--val-dataset", type=str, default=None,
                           help="Separate validation dataset name")
    data_group.add_argument("--val-split", type=str, default="validation",
                           help="Validation split name")
    data_group.add_argument("--val-size", type=float, default=0.1,
                           help="Validation fraction (0.1 = 10%% data for validation)")
    data_group.add_argument("--max-samples", type=int, default=None,
                           help="Max samples to load (for testing: --max-samples 10000)")
    data_group.add_argument("--min-text-length", type=int, default=None,
                           help="Filter: minimum text length (skip chhote texts)")
    data_group.add_argument("--max-text-length", type=int, default=None,
                           help="Filter: maximum text length (skip bade texts)")
    data_group.add_argument("--streaming", action="store_true",
                           help="Use streaming mode (for huge datasets)")
    data_group.add_argument("--cache-dir", type=str, default=None,
                           help="Dataset cache directory")
    
    # ===== TOKENIZER =====
    tok_group = parser.add_argument_group("🔤  TOKENIZER (टोकनाइज़र)")
    tok_group.add_argument("--tokenizer", type=str, default="char", choices=["char", "hf"],
                          help="Tokenizer type (char=simple, hf=huggingface)")
    tok_group.add_argument("--hf-tokenizer", type=str, default="gpt2",
                          help="HF tokenizer name (gpt2, bert-base-uncased, meta-llama/Llama-2-7b-hf)")
    tok_group.add_argument("--no-pad-token", action="store_true",
                          help="Don't add padding token")
    
    # ===== TRAINING =====
    train_group = parser.add_argument_group("🎯  TRAINING (ट्रेनिंग)")
    train_group.add_argument("--epochs", type=int, default=3,
                            help="Number of epochs - एपॉक्स की संख्या (1=test, 3=quick, 10=standard)")
    train_group.add_argument("--batch-size", type=int, default=8,
                            help="Batch size - बैच साइज़ (4=small GPU, 8=default, 16=good, 32=large)")
    train_group.add_argument("--lr", type=float, default=3e-4,
                            help="Learning rate - लर्निंग रेट (1e-4=stable, 3e-4=default, 5e-4=fast)")
    train_group.add_argument("--min-lr", type=float, default=3e-5,
                            help="Minimum learning rate (usually 10%% of peak)")
    train_group.add_argument("--warmup-steps", type=int, default=100,
                            help="Warmup steps (0=none, 100=default, 1000=slow)")
    train_group.add_argument("--weight-decay", type=float, default=0.1,
                            help="Weight decay (0.01=light, 0.1=standard, 1.0=strong)")
    train_group.add_argument("--beta1", type=float, default=0.9,
                            help="Adam beta1 (momentum)")
    train_group.add_argument("--beta2", type=float, default=0.95,
                            help="Adam beta2 (RMS)")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                            help="Gradient clipping (0.5=strict, 1.0=default, 0=off)")
    train_group.add_argument("--dropout", type=float, default=0.0,
                            help="Dropout rate (0.0=none, 0.1=light, 0.2=standard)")
    train_group.add_argument("--label-smoothing", type=float, default=0.0,
                            help="Label smoothing (0.0=off, 0.1=good)")
    train_group.add_argument("--lr-scheduler", type=str, default="cosine",
                            choices=["cosine", "linear", "constant", "cosine_restarts"],
                            help="LR scheduler type")
    
    # ===== HARDWARE =====
    hw_group = parser.add_argument_group("💻  HARDWARE (हार्डवेयर)")
    hw_group.add_argument("--device", type=str, default="auto",
                         choices=["auto", "cpu", "cuda", "mps"],
                         help="Device to use (auto=detect, cpu, cuda=gpu, mps=apple)")
    hw_group.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader workers (0=Windows safe, 2=good, 4=fast)")
    hw_group.add_argument("--mixed-precision", type=str, default="fp16",
                         choices=["fp16", "bf16", "no"],
                         help="Mixed precision (fp16=fast, bf16=better, no=full fp32)")
    hw_group.add_argument("--compile", action="store_true",
                         help="Use torch.compile (20-30%% faster, PyTorch 2.0+)")
    hw_group.add_argument("--multi-gpu", action="store_true",
                         help="Use all available GPUs")
    hw_group.add_argument("--gpu-ids", type=str, default=None,
                         help="Specific GPU IDs (0,1,2,3)")
    hw_group.add_argument("--cpu-threads", type=int, default=None,
                         help="CPU threads for OMP (8=good for CPU training)")
    
    # ===== LOGGING =====
    log_group = parser.add_argument_group("📊  LOGGING (लॉगिंग)")
    log_group.add_argument("--save-dir", type=str, default="checkpoints",
                          help="Save directory for checkpoints")
    log_group.add_argument("--save-every", type=int, default=1000,
                          help="Save checkpoint every N steps")
    log_group.add_argument("--no-save-optimizer", action="store_true",
                          help="Don't save optimizer state (smaller files)")
    log_group.add_argument("--log-interval", type=int, default=10,
                          help="Log every N steps")
    log_group.add_argument("--wandb", action="store_true",
                          help="Enable Weights & Biases logging")
    log_group.add_argument("--wandb-project", type=str, default="novaultra",
                          help="W&B project name")
    log_group.add_argument("--wandb-run", type=str, default=None,
                          help="W&B run name")
    log_group.add_argument("--eval-every", type=int, default=500,
                          help="Evaluate every N steps")
    log_group.add_argument("--eval-steps", type=int, default=100,
                          help="Number of eval steps per evaluation")
    
    # ===== RESUME & GENERATION =====
    misc_group = parser.add_argument_group("🔄  RESUME & GENERATION (रिज़्यूम और जनरेशन)")
    misc_group.add_argument("--resume", type=str, default=None,
                           help="Resume from checkpoint path")
    misc_group.add_argument("--reset-optimizer", action="store_true",
                           help="Reset optimizer on resume")
    misc_group.add_argument("--reset-scheduler", action="store_true",
                           help="Reset scheduler on resume")
    misc_group.add_argument("--generate", action="store_true",
                           help="Generate text after training")
    misc_group.add_argument("--prompt", type=str, default="Once upon a time",
                           help="Prompt for text generation")
    misc_group.add_argument("--gen-tokens", type=int, default=100,
                           help="Number of tokens to generate")
    misc_group.add_argument("--gen-temperature", type=float, default=0.7,
                           help="Generation temperature (0.1=deterministic, 1.0=creative)")
    misc_group.add_argument("--gen-top-k", type=int, default=50,
                           help="Top-k sampling (0=off, 50=default)")
    misc_group.add_argument("--gen-top-p", type=float, default=0.9,
                           help="Top-p nucleus sampling (1.0=off, 0.9=default)")
    
    return parser


# ============================================================================
# SECTION 8: MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point - Complete training pipeline."""
    parser = create_parser()
    args = parser.parse_args()
    
    # ===== SET CPU THREADS =====
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    
    # ===== CONVERT ARGS TO CONFIG =====
    config = AllConfig(
        # Model
        model_name=args.model_name,
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        n_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        ffn_dim=args.ffn_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.block_size,
        weight_share=args.weight_share,
        weight_tying=not args.no_weight_tying,
        # Components
        attn_type=args.attn_type,
        use_rwkv=not args.no_rwkv if args.no_rwkv else (args.use_rwkv if args.use_rwkv else True),
        rwkv_freq=args.rwkv_freq,
        use_ssm=not args.no_ssm if args.no_ssm else (args.use_ssm if args.use_ssm else True),
        ssm_freq=args.ssm_freq,
        ssm_state=args.ssm_state,
        use_memory=not args.no_memory if args.no_memory else (args.use_memory if args.use_memory else True),
        memory_slots=args.memory_slots,
        memory_topk=args.memory_topk,
        memory_freq=args.memory_freq,
        use_router=not args.no_router if args.no_router else (args.use_router if args.use_router else True),
        router_dim=args.router_dim,
        use_rope=not args.no_rope if args.no_rope else (args.use_rope if args.use_rope else True),
        # Dataset
        dataset=args.dataset,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        val_dataset=args.val_dataset,
        val_split=args.val_split,
        val_size=args.val_size,
        max_samples=args.max_samples,
        min_text_len=args.min_text_length,
        max_text_len=args.max_text_length,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        # Tokenizer
        tokenizer=args.tokenizer,
        hf_tokenizer=args.hf_tokenizer,
        add_pad_token=not args.no_pad_token,
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        lr_scheduler=args.lr_scheduler,
        # Hardware
        device=args.device,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        compile=args.compile,
        multi_gpu=args.multi_gpu,
        gpu_ids=args.gpu_ids,
        cpu_threads=args.cpu_threads,
        # Logging
        save_dir=args.save_dir,
        save_every=args.save_every,
        save_optimizer=not args.no_save_optimizer,
        log_interval=args.log_interval,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        # Resume & Generation
        resume=args.resume,
        reset_optimizer=args.reset_optimizer,
        reset_scheduler=args.reset_scheduler,
        generate=args.generate,
        prompt=args.prompt,
        gen_tokens=args.gen_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_k=args.gen_top_k,
        gen_top_p=args.gen_top_p,
    )
    
    # ===== PRINT CONFIG SUMMARY =====
    print("\n" + "█"*60)
    print()
    print(f"  ███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ██╗     ███╗   ███╗")
    print(f"  ████╗  ██║██╔═══██╗██║   ██║██╔══██╗██║     ████╗ ████║")
    print(f"  ██╔██╗ ██║██║   ██║██║   ██║███████║██║     ██╔████╔██║")
    print(f"  ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║██║     ██║╚██╔╝██║")
    print(f"  ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗██║ ╚═╝ ██║")
    print(f"  ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝")
    print(f"  NovaLM-COMPLETE TRAINING PIPELINE")
    print("█"*60)
    print(f"\n📋 CONFIGURATION:")
    print(f"   Model: {config.model_name}")
    print(f"   dim={config.dim}, layers={config.num_layers}, heads={config.num_heads}")
    print(f"   kv_heads={config.n_kv_heads or 'auto'}, vocab={config.vocab_size}")
    print(f"   block_size={config.block_size}, weight_share={config.weight_share}")
    print(f"   Dataset: {config.dataset}")
    print(f"   Epochs: {config.epochs}, Batch: {config.batch_size}")
    print(f"   Device: auto, Mixed: {config.mixed_precision}")
    print(f"   RWKV={'on' if config.use_rwkv else 'off'}, SSM={'on' if config.use_ssm else 'off'}")
    print(f"   Memory={'on' if config.use_memory else 'off'}, Router={'on' if config.use_router else 'off'}")
    
    # ===== LOAD DATASET (Fast Download Tips) =====
    print(f"\n{'='*60}")
    print(f"📚 LOADING DATASET: {config.dataset}")
    print(f"{'='*60}")
    
    # ===== REAL-TIME DOWNLOAD PROGRESS TRACKER =====
    print(f"  🔍 Checking if dataset is cached locally...")
    
    # Check if already cached
    is_cached = False
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names
        try:
            cached_info = get_dataset_split_names(config.dataset, config.dataset_subset)
            is_cached = True
        except:
            pass
    except:
        pass
    
    if not is_cached and not config.streaming:
        print(f"  ⏳ Dataset NOT cached — will download (this may be slow from India/Asia).")
        print(f"  {'━'*60}")
        print(f"  💡 TIPS FOR FASTER DOWNLOAD:")
        print(f"    1. Use HuggingFace mirror (fast in Asia):")
        print(f"       set HF_ENDPOINT=https://hf-mirror.com")
        print(f"    2. Use --streaming mode (no download wait):")
        print(f"       python train_ultra_complete.py ... --streaming")
        print(f"    3. Use a different dataset (TinyStories = 200MB):")
        print(f"       python train_ultra_complete.py --dataset roneneldan/TinyStories")
        print(f"  {'━'*60}")
        print()
    
    try:
        from datasets import load_dataset
        
        dataset_kwargs = {"split": config.dataset_split}
        if config.dataset_subset:
            dataset_kwargs["name"] = config.dataset_subset
        if config.streaming:
            dataset_kwargs["streaming"] = True
        if config.cache_dir:
            dataset_kwargs["cache_dir"] = config.cache_dir
        
        # ===== FASTER DOWNLOAD WITH HF MIRROR =====
        # If HF_ENDPOINT env var is set (e.g. hf-mirror.com), use it; otherwise check
        # for environment variable or use datasets default
        hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if hf_endpoint:
            print(f"  🌐 Using HF endpoint: {hf_endpoint}")
        
        print(f"  📥 Downloading dataset (this may take time for large datasets)...")
        print(f"     HuggingFace's native progress bar will show below:")
        print()
        
        hf_dataset = load_dataset(config.dataset, **dataset_kwargs)
        
        print(f"\n  ✅ Loaded: {config.dataset}")
        if hasattr(hf_dataset, '__len__'):
            print(f"  📊 Samples: {len(hf_dataset):,}")
        else:
            print(f"  📊 Samples: streaming (unknown count)")
        
        # Sample (streaming-safe)
        try:
            if config.streaming:
                sample_iter = iter(hf_dataset)
                first_sample = next(sample_iter)
                sample_text = first_sample.get(config.text_column, "")
            else:
                sample_text = hf_dataset[0].get(config.text_column, "")
            print(f"  📝 Sample: {str(sample_text)[:150]}...")
        except Exception as sample_err:
            print(f"  📝 Sample: (could not preview: {sample_err})")
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        print("   pip install datasets")
        return
    
    # ===== CREATE TOKENIZER =====
    print(f"\n{'='*60}")
    print(f"🔤 CREATING TOKENIZER")
    print(f"{'='*60}")
    
    if config.tokenizer == "hf":
        tokenizer = create_hf_tokenizer(config.hf_tokenizer, config.add_pad_token)
    else:
        tokenizer = create_char_tokenizer()
    
    config.vocab_size = tokenizer.vocab_size
    
    # ===== PREPARE DATASETS =====
    print(f"\n{'='*60}")
    print(f"🔨 PREPARING DATA")
    print(f"{'='*60}")
    
    # Use new Pipeline if available, otherwise fall back to HFDatasetWrapper
    if HAS_PIPELINE:
        # Build pipeline config from AllConfig
        pipeline_config = PipelineConfig(
            streaming=config.streaming,
            backend=BackendConfig(
                dataset_name=config.dataset,
                subset=config.dataset_subset,
                split=config.dataset_split,
                text_column=config.text_column,
                shuffle_buffer=10000,
                max_samples=config.max_samples,
            ),
            tokenizer=TokenizerConfig(
                type=config.tokenizer,
                tokenizer_name=config.hf_tokenizer,
                add_pad=config.add_pad_token,
                cache_size=512,
            ),
            chunker=ChunkConfig(
                block_size=config.block_size,
                overlap=True,
            ),
            loader=LoaderConfig(
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.device == "cuda",
                prefetch_factor=2,
                seed=42,
            ),
            filter=FilterConfig(
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
            ),
        )
        
        pipeline = Pipeline(pipeline_config)
        train_loader = pipeline.create_dataloader()
        
        if config.streaming and hasattr(pipeline.backend, 'iter_documents'):
            first = next(iter(pipeline.backend.iter_documents()))
        else:
            first = pipeline.backend[0]
        print(f"  📝 Sample: {str(first.get(config.text_column, ''))[:150]}...")
        
        # Validation - use separate pipeline if val_dataset specified
        val_loader = None
        if config.val_dataset:
            try:
                val_config = PipelineConfig(
                    streaming=False,
                    backend=BackendConfig(
                        dataset_name=config.val_dataset,
                        subset=None,
                        split=config.val_split,
                        text_column=config.text_column,
                    ),
                    tokenizer=TokenizerConfig(
                        type=config.tokenizer,
                        tokenizer_name=config.hf_tokenizer,
                        add_pad=config.add_pad_token,
                    ),
                    chunker=ChunkConfig(
                        block_size=config.block_size,
                        overlap=True,
                    ),
                    loader=LoaderConfig(
                        batch_size=config.batch_size,
                        num_workers=0,
                    ),
                )
                val_pipeline = Pipeline(val_config)
                val_loader = val_pipeline.create_dataloader()
                print(f"  📊 Validation dataset: {config.val_dataset}")
            except Exception as e:
                print(f"  ⚠️  Could not load validation dataset: {e}")
    else:
        # Fallback to old HFDatasetWrapper (for backward compatibility)
        train_dataset = HFDatasetWrapper(
            hf_dataset, tokenizer,
            block_size=config.block_size,
            text_column=config.text_column,
            min_text_len=config.min_text_len,
            max_text_len=config.max_text_len,
            max_samples=config.max_samples,
        )
        
        # Validation split
        val_loader = None
        if config.val_dataset:
            try:
                val_hf = load_dataset(config.val_dataset, split=config.val_split)
                val_dataset = HFDatasetWrapper(
                    val_hf, tokenizer,
                    block_size=config.block_size,
                    text_column=config.text_column,
                )
                if len(val_dataset) > 0:
                    val_loader = DataLoader(
                        val_dataset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers,
                    )
            except:
                pass
        
        if val_loader is None and config.val_size > 0:
            # Split train into train/val
            val_len = int(len(train_dataset) * config.val_size)
            train_len = len(train_dataset) - val_len
            if val_len > 0 and train_len > 0:
                train_subset, val_subset = random_split(
                    train_dataset, [train_len, val_len],
                    generator=torch.Generator().manual_seed(42),
                )
                
                from torch.utils.data import Subset
                class SubsetWrapper(Dataset):
                    def __init__(self, subset):
                        self.subset = subset
                    def __len__(self):
                        return len(self.subset)
                    def __getitem__(self, idx):
                        return self.subset[idx]
                
                train_dataset = SubsetWrapper(train_subset)
                val_dataset = SubsetWrapper(val_subset)
                
                val_loader = DataLoader(
                    val_dataset, batch_size=config.batch_size,
                    shuffle=False, num_workers=config.num_workers,
                )
                print(f"  Validation split: {val_len} samples")
        
        # Train loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.device == "cuda",
        )
    
    # ===== CREATE MODEL =====
    model = create_model(config)
    
    # ===== TRAIN =====
    model = train_model(model, train_loader, val_loader, config)
    
    # ===== GENERATE =====
    if config.generate:
        generate_text(model, tokenizer, config)
    
    print(f"\n{'='*60}")
    print(f"✅ ALL DONE! 🎉")
    print(f"   Model: {config.model_name}")
    print(f"   Checkpoints: {Path(config.save_dir) / config.model_name}")
    print(f"{'='*60}")
    
    # Print next steps
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. Generate more text:")
    print(f"      python {__file__} --resume {Path(config.save_dir) / config.model_name / 'latest.pt'} --generate --prompt 'Your prompt here'")
    print(f"   2. Continue training:")
    print(f"      python {__file__} --resume {Path(config.save_dir) / config.model_name / 'latest.pt'} --epochs 10")
    print(f"   3. Run example script:")
    print(f"      python scripts/example_usage.py --checkpoint {Path(config.save_dir) / config.model_name / 'best_model.pt'}")
    print()


if __name__ == "__main__":
    main()