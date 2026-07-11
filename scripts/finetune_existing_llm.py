"""
============================================================================
NovaLM-ULTRA v1.0 - Fine-tune ANY Existing LLM
============================================================================
Fine-tune existing models from HuggingFace (GPT-2, Llama, Mistral, etc.)
on ANY dataset. Supports both CPU and GPU training.

EXAMPLES:
  # === Fine-tune GPT-2 on TinyStories (CPU/GPU) ===
  python finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --epochs 3
  
  # === Fine-tune Llama-2 (GPU only) ===
  python finetune_existing_llm.py --model meta-llama/Llama-2-7b-hf --dataset HuggingFaceFW/fineweb --epochs 1 --lora --quantize 4bit
  
  # === Fine-tune Mistral ===
  python finetune_existing_llm.py --model mistralai/Mistral-7B-v0.1 --dataset HuggingFaceFW/fineweb-edu --epochs 3 --lora --quantize 4bit
  
  # === Fine-tune on CPU ===
  python finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --device cpu --epochs 2 --batch-size 2
  
  # === Custom dataset ===
  python finetune_existing_llm.py --model gpt2 --dataset my_dataset --text-column content --epochs 5
"""

import sys
import os
import argparse
import math
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# ============================================================================
# SECTION 1: SUPPORTED MODELS
# ============================================================================

SUPPORTED_MODELS = {
    # Small models (CPU-friendly)
    "gpt2": {"type": "causal", "params": "124M", "min_gpu": "2 GB", "min_cpu": "4 GB"},
    "gpt2-medium": {"type": "causal", "params": "355M", "min_gpu": "4 GB", "min_cpu": "8 GB"},
    "gpt2-large": {"type": "causal", "params": "774M", "min_gpu": "6 GB", "min_cpu": "16 GB"},
    "gpt2-xl": {"type": "causal", "params": "1.5B", "min_gpu": "8 GB", "min_cpu": "24 GB"},
    "distilgpt2": {"type": "causal", "params": "82M", "min_gpu": "1 GB", "min_cpu": "2 GB"},
    "bert-base-uncased": {"type": "mlm", "params": "110M", "min_gpu": "2 GB", "min_cpu": "4 GB"},
    "bert-large-uncased": {"type": "mlm", "params": "340M", "min_gpu": "4 GB", "min_cpu": "8 GB"},
    "roberta-base": {"type": "mlm", "params": "125M", "min_gpu": "2 GB", "min_cpu": "4 GB"},
    "albert-base-v2": {"type": "mlm", "params": "12M", "min_gpu": "1 GB", "min_cpu": "2 GB"},
    "t5-small": {"type": "seq2seq", "params": "60M", "min_gpu": "2 GB", "min_cpu": "4 GB"},
    "t5-base": {"type": "seq2seq", "params": "220M", "min_gpu": "4 GB", "min_cpu": "8 GB"},
    
    # Medium models (GPU recommended)
    "microsoft/phi-2": {"type": "causal", "params": "2.7B", "min_gpu": "8 GB", "min_cpu": "32 GB"},
    "google/gemma-2b": {"type": "causal", "params": "2B", "min_gpu": "6 GB", "min_cpu": "16 GB"},
    "google/gemma-7b": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "Qwen/Qwen-1.5B": {"type": "causal", "params": "1.5B", "min_gpu": "6 GB", "min_cpu": "16 GB"},
    "Qwen/Qwen-7B": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    
    # Large models (GPU required, use LoRA + quantization)
    "meta-llama/Llama-2-7b-hf": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "meta-llama/Llama-2-13b-hf": {"type": "causal", "params": "13B", "min_gpu": "24 GB", "min_cpu": "128 GB"},
    "meta-llama/Meta-Llama-3-8B": {"type": "causal", "params": "8B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "mistralai/Mistral-7B-v0.1": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "mistralai/Mixtral-8x7B-v0.1": {"type": "causal", "params": "47B", "min_gpu": "48 GB", "min_cpu": "256 GB"},
    "codellama/CodeLlama-7b-hf": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "codellama/CodeLlama-34b-hf": {"type": "causal", "params": "34B", "min_gpu": "48 GB", "min_cpu": "256 GB"},
    "tiiuae/falcon-7b": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "tiiuae/falcon-40b": {"type": "causal", "params": "40B", "min_gpu": "80 GB", "min_cpu": "512 GB"},
    "stabilityai/stablelm-2-12b": {"type": "causal", "params": "12B", "min_gpu": "24 GB", "min_cpu": "96 GB"},
    
    # Hindi/Indian language models
    "sarvamai/OpenHathi-7B": {"type": "causal", "params": "7B", "min_gpu": "16 GB", "min_cpu": "64 GB"},
    "ai4bharat/indic-bert": {"type": "mlm", "params": "110M", "min_gpu": "2 GB", "min_cpu": "4 GB"},
}

# ============================================================================
# SECTION 2: LoRA (Low-Rank Adaptation) - For large model fine-tuning
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation for efficient fine-tuning.
    Instead of updating ALL weights, we only train small rank matrices.
    This saves 99% memory for large models!
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Register forward hook to apply LoRA
        self.register_forward_hook(self._lora_forward)
    
    @staticmethod
    def _lora_forward(module, input, output):
        x = input[0]
        lora_update = (x @ module.lora_A) @ module.lora_B
        return output + lora_update * module.scaling
    
    def forward(self, x):
        return self.original(x)


def apply_lora(model, rank=8, alpha=16, target_modules=None):
    """
    Apply LoRA to linear layers in the model.
    Only trains LoRA parameters, original weights stay frozen.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                         "gate_proj", "up_proj", "down_proj",
                         "query", "value", "key", "dense"]
    
    lora_params = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                
                lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                
                lora_params += 2 * module.in_features * rank
                total_params += module.in_features * module.out_features
    
    print(f"   LoRA applied! Trainable: {lora_params:,} / {total_params:,} params")
    print(f"   Memory savings: {total_params / max(lora_params, 1):.0f}x")
    
    return model


# ============================================================================
# SECTION 3: QUANTIZATION (For large models on limited GPU)
# ============================================================================

def get_quantization_config(quantize: str = "none"):
    """
    Get quantization config for loading large models.
    - "none": Full precision (best quality, most memory)
    - "8bit": 8-bit quantization (half memory, minimal quality loss)
    - "4bit": 4-bit quantization (quarter memory, slight quality loss)
    """
    if quantize == "none":
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        
        if quantize == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        elif quantize == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    except ImportError:
        print("⚠️  bitsandbytes not installed. Install: pip install bitsandbytes")
        return None
    
    return None


# ============================================================================
# SECTION 4: DATASET HANDLING
# ============================================================================

class TextDataset(Dataset):
    """Dataset for fine-tuning language models."""
    def __init__(self, texts, tokenizer, block_size=512, text_column="text"):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize all texts
        self.examples = []
        for text in texts:
            if isinstance(text, dict):
                text = text.get(text_column, "")
            if not isinstance(text, str):
                text = str(text)
            
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size, block_size // 2):
                chunk = tokens[i:i + block_size]
                if len(chunk) == block_size:
                    self.examples.append(chunk)
        
        if not self.examples:
            # Create a dummy example
            self.examples = [[tokenizer.eos_token_id] * block_size]
        
        self.data = torch.tensor(self.examples, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chunk = self.data[idx]
        return {"input_ids": chunk, "labels": chunk}


# ============================================================================
# SECTION 5: TRAINING FUNCTION
# ============================================================================

def finetune(
    model,
    tokenizer,
    train_loader,
    val_loader,
    config: dict,
):
    """
    Fine-tune any HuggingFace model on custom dataset.
    Works for both CPU and GPU.
    """
    # === DEVICE ===
    device_str = config.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"
    
    device = torch.device(device_str)
    print(f"📱 Device: {device_str}")
    
    if device_str == "cpu":
        if config.get("cpu_threads"):
            torch.set_num_threads(config["cpu_threads"])
            print(f"   CPU threads: {config['cpu_threads']}")
        print("   ⚠️  CPU training is SLOW for large models!")
        print("   Recommended: use GPT-2 or smaller (gpt2, distilgpt2)")
    
    # Move model to device
    model = model.to(device)
    
    # === MIXED PRECISION ===
    scaler = None
    amp_dtype = None
    mp = config.get("mixed_precision", "fp16")
    
    if mp == "fp16" and device_str == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        amp_dtype = torch.float16
    elif mp == "bf16" and device_str == "cuda":
        amp_dtype = torch.bfloat16
    
    # === OPTIMIZER ===
    # Only train parameters that require gradients (LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in trainable_params)
    
    print(f"   Frozen params: {frozen_params:,}")
    print(f"   Trainable params: {trainable_count:,} ({trainable_count * 4 / 1024 / 1024:.1f} MB)")
    
    optimizer = AdamW(
        trainable_params,
        lr=config.get("lr", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    # === LR SCHEDULER ===
    total_steps = len(train_loader) * config.get("epochs", 3)
    warmup_steps = config.get("warmup_steps", min(100, total_steps // 10))
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, 
                            eta_min=config.get("min_lr", config.get("lr", 5e-5) * 0.1)),
        ],
        milestones=[warmup_steps],
    )
    
    # === TRAINING LOOP ===
    print(f"\n{'='*60}")
    print(f"🚀 FINE-TUNING STARTED")
    print(f"   Model: {config.get('model_name', 'unknown')}")
    print(f"   Dataset: {config.get('dataset', 'unknown')}")
    print(f"   Batch: {config.get('batch_size', 4)}, LR: {config.get('lr', 5e-5):.2e}")
    print(f"   Epochs: {config.get('epochs', 3)}, Total steps: {total_steps}")
    print(f"{'='*60}\n")
    
    global_step = 0
    best_loss = float("inf")
    total_start = time.time()
    
    for epoch in range(config.get("epochs", 3)):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.get('epochs', 3)}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if config.get("grad_clip", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss.backward()
                if config.get("grad_clip", 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                optimizer.step()
            
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            if global_step % config.get("log_interval", 10) == 0:
                elapsed = time.time() - total_start
                tok_per_sec = (global_step * config.get("batch_size", 4) * input_ids.size(1)) / elapsed
                tqdm.write(f"  Step {global_step:>6} | Loss: {loss.item():.4f} | Tok/s: {tok_per_sec:.0f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"\n📊 Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        save_dir = Path(config.get("save_dir", "checkpoints")) / config.get("model_name", "finetuned")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state": model.state_dict() if not config.get("lora") else None,
            "lora_state": {k: v for k, v in model.state_dict().items() if "lora_" in k} if config.get("lora") else None,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step,
            "loss": avg_loss,
            "config": config,
        }
        
        # Save LoRA weights only (small file)
        if config.get("lora"):
            lora_path = save_dir / f"lora_epoch_{epoch+1}.pt"
            torch.save(checkpoint["lora_state"], lora_path)
            print(f"💾 LoRA saved: {lora_path}")
        
        # Save full checkpoint
        ckpt_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_dir / "best_model.pt"
            if config.get("lora"):
                torch.save(checkpoint["lora_state"], best_path)
            else:
                torch.save({"model_state": model.state_dict()}, best_path)
            print(f"🏆 Best model saved!")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ FINE-TUNING COMPLETE!")
    print(f"   Time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Saved to: {save_dir}")
    print(f"{'='*60}")
    
    return model


# ============================================================================
# SECTION 6: MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ANY existing LLM from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # CPU: Fine-tune GPT-2 on TinyStories
  python finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --device cpu --epochs 2 --batch-size 2
  
  # GPU: Fine-tune GPT-2
  python finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --epochs 3 --batch-size 8
  
  # GPU: Fine-tune Llama-2 with LoRA (saves memory)
  python finetune_existing_llm.py --model meta-llama/Llama-2-7b-hf --dataset HuggingFaceFW/fineweb --epochs 3 --lora --rank 16 --quantize 4bit
  
  # GPU: Fine-tune Mistral with LoRA
  python finetune_existing_llm.py --model mistralai/Mistral-7B-v0.1 --dataset HuggingFaceFW/fineweb-edu --epochs 5 --lora --rank 8
  
  # Full fine-tune (no LoRA) on smaller model
  python finetune_existing_llm.py --model google/gemma-2b --dataset roneneldan/TinyStories --epochs 3 --batch-size 4
  
  # List supported models
  python finetune_existing_llm.py --list-models
        """,
    )
    
    # Model args
    parser.add_argument("--model", type=str, default="gpt2",
                       help="HF model name or path (default: gpt2)")
    parser.add_argument("--list-models", action="store_true",
                       help="List all supported models")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                       help="HF dataset name (default: TinyStories)")
    parser.add_argument("--dataset-subset", type=str, default=None,
                       help="Dataset subset/config")
    parser.add_argument("--dataset-split", type=str, default="train",
                       help="Dataset split (default: train)")
    parser.add_argument("--text-column", type=str, default="text",
                       help="Text column name (default: text)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples to use")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--block-size", type=int, default=512,
                       help="Sequence length (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--min-lr", type=float, default=5e-6,
                       help="Min learning rate (default: 5e-6)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Warmup steps (default: 100)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay (default: 0.01)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                       help="Gradient clipping (default: 1.0)")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Log every N steps (default: 10)")
    
    # Fine-tuning method
    parser.add_argument("--lora", action="store_true",
                       help="Use LoRA (Low-Rank Adaptation) - 99% memory savings")
    parser.add_argument("--rank", type=int, default=8,
                       help="LoRA rank (default: 8, higher = more capacity)")
    parser.add_argument("--alpha", type=int, default=16,
                       help="LoRA alpha scaling (default: 16)")
    
    # Quantization
    parser.add_argument("--quantize", type=str, default="none",
                       choices=["none", "8bit", "4bit"],
                       help="Quantization (4bit=smallest memory, 8bit=balanced, none=best quality)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device (auto=detect, cpu, cuda=gpu, mps=apple)")
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                       choices=["fp16", "bf16", "no"],
                       help="Mixed precision (default: fp16)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                       help="CPU threads (for CPU training)")
    
    # Save
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Save directory (default: checkpoints)")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name for saving (default: auto)")
    
    # Inference after training
    parser.add_argument("--generate", action="store_true",
                       help="Generate text after fine-tuning")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt for generation")
    parser.add_argument("--gen-tokens", type=int, default=100,
                       help="Tokens to generate (default: 100)")
    parser.add_argument("--gen-temperature", type=float, default=0.7,
                       help="Temperature (default: 0.7)")
    
    args = parser.parse_args()
    
    # === LIST MODELS ===
    if args.list_models:
        print("\n" + "="*60)
        print("📋 SUPPORTED MODELS")
        print("="*60)
        print(f"{'Model Name':<40} {'Params':<10} {'GPU':<10} {'CPU':<10}")
        print("-"*70)
        for name, info in sorted(SUPPORTED_MODELS.items()):
            print(f"{name:<40} {info['params']:<10} {info['min_gpu']:<10} {info['min_cpu']:<10}")
        print(f"\nTip: For CPU training, use models with < 1B params (gpt2, distilgpt2, etc.)")
        print(f"Tip: For GPU training with large models, use --lora --quantize 4bit")
        return
    
    # === MODEL NAME ===
    model_name = args.model_name or args.model.replace("/", "-")
    
    # === CHECK MODEL SIZE VS HARDWARE ===
    model_info = None
    for key, info in SUPPORTED_MODELS.items():
        if key in args.model:
            model_info = info
            break
    
    if model_info:
        params = model_info["params"]
        if params.endswith("B"):
            param_count = float(params[:-1])
            print(f"\n📊 Model size: {params} parameters")
            if param_count >= 7 and args.device == "cpu" and not args.quantize != "none":
                print(f"   ⚠️  This model is LARGE! It may not fit on CPU.")
                print(f"   💡 Use --lora --quantize 4bit for memory efficiency")
                print(f"   💡 Or use --device cuda if you have a GPU")
            if param_count >= 7 and not args.lora:
                print(f"   💡 Recommended: add --lora to save 99% memory")
    
    # =========================================================================
    # 1. LOAD DATASET
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"📚 LOADING DATASET: {args.dataset}")
    print(f"{'='*60}")
    
    from datasets import load_dataset
    dataset_kwargs = {"split": args.dataset_split}
    if args.dataset_subset:
        dataset_kwargs["name"] = args.dataset_subset
    if args.streaming:
        dataset_kwargs["streaming"] = True
    if args.max_samples:
        dataset_kwargs["split"] = f"{args.dataset_split}[:{args.max_samples}]"
    
    dataset = load_dataset(args.dataset, **dataset_kwargs)
    print(f"  ✅ Loaded: {len(dataset):,} samples")
    
    # =========================================================================
    # 2. LOAD MODEL & TOKENIZER
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🤖 LOADING MODEL: {args.model}")
    print(f"{'='*60}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    
    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    
    # Load model with optional quantization
    print("  Loading model (this may take a while)...")
    quantization_config = get_quantization_config(args.quantize)
    
    load_kwargs = {
        "device_map": "auto" if torch.cuda.is_available() else None,
        "torch_dtype": torch.float32,
    }
    
    if quantization_config:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["torch_dtype"] = torch.float16
    
    # Determine model type
    model_type = "causal"
    if model_info:
        model_type = model_info["type"]
    
    if model_type == "causal":
        # Try AutoModelForCausalLM first
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        except:
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **load_kwargs)
    else:
        from transformers import AutoModelForMaskedLM
        model = AutoModelForMaskedLM.from_pretrained(args.model, **load_kwargs)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✅ Model loaded!")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # =========================================================================
    # 3. APPLY LoRA (if requested)
    # =========================================================================
    if args.lora:
        print(f"\n{'='*60}")
        print(f"🧩 APPLYING LoRA (rank={args.rank}, alpha={args.alpha})")
        print(f"{'='*60}")
        model = apply_lora(model, rank=args.rank, alpha=args.alpha)
    
    # =========================================================================
    # 4. PREPARE DATASET
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"🔨 PREPARING DATA")
    print(f"{'='*60}")
    
    texts = [example for example in dataset]
    train_data = TextDataset(texts, tokenizer, block_size=args.block_size, text_column=args.text_column)
    
    # Split into train/val
    val_len = int(len(train_data) * 0.05)
    train_len = len(train_data) - val_len
    
    from torch.utils.data import random_split
    train_subset, val_subset = random_split(train_data, [train_len, val_len])
    
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )
    
    print(f"  Train: {train_len:,} samples")
    print(f"  Val: {val_len:,} samples")
    
    # =========================================================================
    # 5. TRAIN
    # =========================================================================
    config = {
        "model_name": model_name,
        "dataset": args.dataset,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "log_interval": args.log_interval,
        "save_dir": args.save_dir,
        "mixed_precision": args.mixed_precision,
        "cpu_threads": args.cpu_threads,
        "lora": args.lora,
    }
    
    model = finetune(model, tokenizer, train_loader, val_loader, config)
    
    # =========================================================================
    # 6. GENERATE (optional)
    # =========================================================================
    if args.generate:
        print(f"\n{'='*60}")
        print(f"✍️ GENERATING TEXT")
        print(f"{'='*60}")
        
        model.eval()
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
        
        # Move to correct device
        model_device = next(model.parameters()).device
        input_ids = input_ids.to(model_device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.gen_tokens,
                temperature=args.gen_temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n   Prompt: {args.prompt}")
        print(f"   {'─'*60}")
        print(f"   {output_text}")
        print(f"   {'─'*60}")
    
    print(f"\n✅ ALL DONE!")


if __name__ == "__main__":
    main()