"""
NovaLM-ULTRA v1.0 - Train on HuggingFace Datasets
=================================================
Complete training pipeline for training NovaLM-ULTRA on any HF dataset.

Usage:
    # Train on TinyStories (small, good for testing)
    python train_hf_dataset.py --dataset roneneldan/TinyStories --config fast
    
    # Train on fineweb-edu (medium scale)
    python train_hf_dataset.py --dataset HuggingFaceFW/fineweb-edu --config balanced --subset sample-10BT
    
    # Resume from checkpoint
    python train_hf_dataset.py --resume checkpoints/step_1000.pt
    
    # Custom config
    python train_hf_dataset.py --dataset my_dataset --dim 512 --layers 8 --epochs 5
"""
import sys
import os
import argparse
import math
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm

# Add NovaLM-ULTRA to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# ============================================================================
# Import NovaLM-ULTRA
# ============================================================================
try:
    from NovaLM_ULTRA import create_ultra_model, UltraConfig
    from NovaLM_ULTRA import NovaUltraModel, UltraBlock
    print("✅ NovaLM-ULTRA imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import NovaLM-ULTRA: {e}")
    print("   Make sure you've installed it: pip install -e NovaLM-ULTRA/")
    print("   Or run from the NovaLM-ULTRA directory.")
    sys.exit(1)


# ============================================================================
# HF Dataset Wrapper
# ============================================================================
class HFDatasetWrapper(Dataset):
    """
    Wraps a HuggingFace dataset for training NovaLM-ULTRA.
    Handles tokenization, chunking, and streaming.
    """
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        block_size: int = 256,
        text_column: str = "text",
        shuffle_buffer: int = 10000,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_column = text_column
        self.shuffle_buffer = shuffle_buffer
        
        # Pre-tokenize everything into chunks
        self.tokens = []
        print("  Tokenizing dataset...")
        for example in tqdm(self.dataset, desc="Tokenizing"):
            text = example.get(self.text_column, "")
            if text:
                tokens = tokenizer.encode(text)
                # Chunk into block_size pieces
                for i in range(0, len(tokens) - block_size, block_size // 2):
                    chunk = tokens[i:i + block_size]
                    if len(chunk) == block_size:
                        self.tokens.append(chunk)
        
        print(f"  Created {len(self.tokens):,} training chunks")
        
        # Convert to tensor for fast access
        if self.tokens:
            self.data = torch.tensor(self.tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {
            "input_ids": chunk,
            "labels": chunk,
        }


# ============================================================================
# Tokenizer Setup
# ============================================================================
def create_tokenizer(
    vocab_size: int = 10000,
    use_hf_tokenizer: bool = False,
    hf_tokenizer_name: str = "bert-base-uncased",
):
    """
    Creates a tokenizer for the model.
    
    Two options:
    1. Simple character-level tokenizer (default, no extra deps)
    2. HF tokenizer (better quality, needs transformers)
    """
    if use_hf_tokenizer:
        from transformers import AutoTokenizer
        print(f"  Loading HF tokenizer: {hf_tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        print(f"  Vocab size: {tokenizer.vocab_size}")
        return tokenizer
    else:
        # Simple character-level tokenizer
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}\"' \n\t-_=/\\@#$%^&*~`|<>+"
        char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad/unk
        char_to_idx["<PAD>"] = 0
        char_to_idx["<UNK>"] = 0
        
        class SimpleTokenizer:
            def __init__(self, char_map, vocab_size):
                self.char_map = char_map
                self._vocab_size = vocab_size
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
        
        print(f"  Created char-level tokenizer with vocab size: {len(chars)}")
        return SimpleTokenizer(char_to_idx, len(chars))


# ============================================================================
# Training Function
# ============================================================================
def train(
    model: NovaUltraModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 3,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    save_dir: str = "checkpoints",
    device: str = "auto",
    use_mixed_precision: bool = True,
    resume_path: Optional[str] = None,
):
    """
    Complete training loop with:
    - AdamW optimizer with weight decay
    - Linear warmup + Cosine annealing LR schedule
    - Gradient clipping
    - Mixed precision (fp16/bf16)
    - Checkpoint saving
    """
    # --- Device setup ---
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"📱 Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # --- Mixed precision setup ---
    scaler = None
    amp_dtype = None
    if use_mixed_precision and device == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        amp_dtype = torch.float16
        print("  Using mixed precision (fp16)")
    elif use_mixed_precision and device == "mps":
        amp_dtype = torch.float16
        print("  Using mixed precision (fp16 on MPS)")
    elif device == "cpu":
        print("  Using full precision (fp32)")
    
    # --- Optimizer ---
    # Separate params with/without weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
    )
    
    # --- LR Scheduler ---
    total_steps = len(train_loader) * epochs
    
    # Warmup phase
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    
    # Cosine decay phase
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    
    # --- Resume from checkpoint ---
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    
    if resume_path and os.path.exists(resume_path):
        print(f"📂 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))
        if scaler and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")
    
    # --- Create save directory ---
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Training Loop ---
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING STARTED")
    print(f"   Model params: {model.total_params:,}")
    print(f"   Effective params: {model.get_effective_params():,}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   Total steps: {total_steps}")
    print(f"   Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    model.train()
    total_start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward + backward with optional mixed precision
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            elif amp_dtype:
                with torch.autocast(device_type=device, dtype=amp_dtype):
                    logits, _ = model(input_ids)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            else:
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Logging
            if global_step % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - total_start_time
                tokens_per_sec = (global_step * train_loader.batch_size * input_ids.size(1)) / elapsed
                tqdm.write(
                    f"  Step {global_step:>6} | Loss: {loss.item():.4f} "
                    f"| LR: {current_lr:.2e} | Tok/s: {tokens_per_sec:.0f}"
                )
        
        # End of epoch
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"\n📊 Epoch {epoch+1} completed:")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "step": global_step,
            "loss": avg_loss,
            "best_loss": best_loss,
            "config": {
                "dim": model.config.dim,
                "num_layers": model.config.num_layers,
                "num_heads": model.config.num_heads,
                "n_kv_heads": model.config.n_kv_heads,
                "vocab_size": model.config.vocab_size,
                "ffn_dim": model.config.ffn_dim,
                "weight_share_iterations": model.config.weight_share_iterations,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            torch.save({"model_state": model.state_dict(), "config": checkpoint["config"]}, best_path)
            print(f"🏆 New best model! Saved to: {best_path}")
        
        # Save latest model (easy to resume)
        latest_path = save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Validation (optional)
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validating"):
                    val_ids = val_batch["input_ids"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    val_logits, _ = model(val_ids)
                    val_loss += F.cross_entropy(
                        val_logits.view(-1, val_logits.size(-1)),
                        val_labels.view(-1),
                    ).item()
            val_loss /= len(val_loader)
            print(f"   Validation Loss: {val_loss:.4f}")
            model.train()
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")
    print(f"{'='*60}")
    
    return model


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train NovaLM-ULTRA on HuggingFace Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (fast config, TinyStories)
  python train_hf_dataset.py --dataset roneneldan/TinyStories --config fast --epochs 1
  
  # Full training
  python train_hf_dataset.py --dataset HuggingFaceFW/fineweb --config balanced --epochs 3
  
  # Resume training
  python train_hf_dataset.py --resume checkpoints/latest.pt --epochs 5
  
  # Custom model config
  python train_hf_dataset.py --dataset my_data --dim 256 --layers 4 --heads 8 --vocab 5000
        """,
    )
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset name (default: TinyStories)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config (e.g., sample-10BT)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--text-column", type=str, default="text",
                        help="Text column in dataset (default: text)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to use (for testing)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    
    # Model config args
    parser.add_argument("--config", type=str, default="fast",
                        choices=["fast", "balanced", "powerful", "ultra"],
                        help="Predefined config (default: fast)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Model dimension (overrides config)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Number of layers (overrides config)")
    parser.add_argument("--heads", type=int, default=None,
                        help="Number of attention heads (overrides config)")
    parser.add_argument("--kv-heads", type=int, default=None,
                        help="Number of KV heads for GQA (overrides config)")
    parser.add_argument("--vocab", type=int, default=None,
                        help="Vocabulary size (overrides config)")
    parser.add_argument("--weight-share", type=int, default=None,
                        help="Weight sharing iterations (overrides config)")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Sequence length / block size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--min-lr", type=float, default=3e-5,
                        help="Minimum learning rate (default: 3e-5)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps (default: 100)")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay (default: 0.1)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--no-mixed-precision", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use (default: auto)")
    
    # Checkpoint args
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # HF tokenizer args
    parser.add_argument("--use-hf-tokenizer", action="store_true",
                        help="Use HuggingFace tokenizer instead of simple char tokenizer")
    parser.add_argument("--hf-tokenizer-name", type=str, default="bert-base-uncased",
                        help="HF tokenizer name (default: bert-base-uncased)")
    
    args = parser.parse_args()
    
    # =========================================================================
    # 1. LOAD DATASET FROM HUGGINGFACE
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"📚 LOADING DATASET: {args.dataset}")
    print(f"{'='*60}")
    
    try:
        from datasets import load_dataset
        
        dataset_kwargs = {}
        if args.subset:
            dataset_kwargs["name"] = args.subset
        if args.streaming:
            dataset_kwargs["streaming"] = True
        if args.max_samples:
            dataset_kwargs["split"] = f"{args.split}[:{args.max_samples}]"
        else:
            dataset_kwargs["split"] = args.split
        
        hf_dataset = load_dataset(args.dataset, **dataset_kwargs)
        print(f"  ✅ Loaded dataset: {args.dataset}")
        print(f"  📊 Num examples: {len(hf_dataset):,}")
        print(f"  📝 Text column: {args.text_column}")
        
        # Show a sample
        sample = hf_dataset[0][args.text_column][:200]
        print(f"  Sample: {sample}...")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("   Make sure 'datasets' is installed: pip install datasets")
        sys.exit(1)
    
    # =========================================================================
    # 2. CREATE TOKENIZER
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🔤 CREATING TOKENIZER")
    print(f"{'='*60}")
    
    tokenizer = create_tokenizer(
        use_hf_tokenizer=args.use_hf_tokenizer,
        hf_tokenizer_name=args.hf_tokenizer_name,
    )
    actual_vocab_size = tokenizer.vocab_size
    
    # Config from args
    config_map = {
        "fast": UltraConfig.fast_config(),
        "balanced": UltraConfig.balanced_config(),
        "powerful": UltraConfig.powerful_config(),
        "ultra": UltraConfig.ultra_config(),
    }
    config = config_map[args.config]
    
    # Override with custom args
    if args.dim is not None:
        config.dim = args.dim
    if args.layers is not None:
        config.num_layers = args.layers
    if args.heads is not None:
        config.num_heads = args.heads
    if args.kv_heads is not None:
        config.n_kv_heads = args.kv_heads
    if args.vocab is not None:
        config.vocab_size = args.vocab
        actual_vocab_size = args.vocab
    if args.weight_share is not None:
        config.weight_share_iterations = args.weight_share
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.block_size is not None:
        config.max_seq_len = args.block_size
    
    # Always override vocab to match tokenizer
    config.vocab_size = actual_vocab_size + 1  # +1 for padding
    
    # =========================================================================
    # 3. PREPARE DATASET
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🔨 PREPARING TRAINING DATA")
    print(f"{'='*60}")
    
    train_dataset = HFDatasetWrapper(
        hf_dataset,
        tokenizer,
        block_size=args.block_size,
        text_column=args.text_column,
    )
    
    if len(train_dataset) == 0:
        print("❌ No training data created! Check your dataset.")
        sys.exit(1)
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for Windows compatibility
        pin_memory=torch.cuda.is_available(),
    )
    
    # =========================================================================
    # 4. CREATE MODEL
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🤖 CREATING NOVALM-ULTRA MODEL")
    print(f"{'='*60}")
    
    model = create_ultra_model(config)
    print(model.summary())
    
    # =========================================================================
    # 5. START TRAINING
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🎯 STARTING TRAINING")
    print(f"{'='*60}")
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
        device=args.device,
        use_mixed_precision=not args.no_mixed_precision,
        resume_path=args.resume,
    )
    
    print(f"\n✅ Training complete!")
    print(f"   To generate text with your trained model:")
    print(f"     python scripts/example_usage.py --checkpoint {args.save_dir}/best_model.pt")


if __name__ == "__main__":
    main()