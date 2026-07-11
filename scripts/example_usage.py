"""
NovaLM-ULTRA v1.0 - Example Usage Script
=========================================
Shows how to:
1. Import and create the model
2. Generate text
3. Train on your own data
4. Save/load checkpoints

Usage:
    # Run with balanced config
    python example_usage.py
    
    # Run with trained checkpoint
    python example_usage.py --checkpoint ../checkpoints/best_model.pt
    
    # Run fast test
    python example_usage.py --config fast
"""
import sys
import os
import argparse
import math

import torch
import torch.nn.functional as F

# Add project root to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)


# ============================================================================
# EXAMPLE 1: Basic Import and Model Creation
# ============================================================================
def example_basic_creation():
    """How to create and inspect NovaLM-ULTRA model."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 1: Creating NovaLM-ULTRA")
    print("="*60)
    
    # Import the model
    from NovaLM_ULTRA import create_ultra_model, UltraConfig
    from NovaLM_ULTRA import NovaUltraModel
    
    # Create with fast config (for testing)
    config = UltraConfig.fast_config()
    model = create_ultra_model(config)
    
    # Print model summary
    print(model.summary())
    
    # Or create with balanced config (recommended for real use)
    config_balanced = UltraConfig.balanced_config()
    model_balanced = create_ultra_model(config_balanced)
    print(f"\nBalanced model params: {model_balanced.total_params:,}")
    print(f"Effective params: {model_balanced.get_effective_params():,}")
    
    return model


# ============================================================================
# EXAMPLE 2: Forward Pass
# ============================================================================
def example_forward_pass(model):
    """How to run a forward pass through the model."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 2: Forward Pass")
    print("="*60)
    
    # Create random input tokens
    batch_size = 1
    seq_len = 32
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass (training mode)
    model.train()
    logits, _ = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Output vocab dim: {logits.size(-1)}")
    
    # Calculate loss (next-token prediction)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        input_ids.view(-1),
    )
    print(f"Cross-entropy loss: {loss.item():.4f}")
    
    # Backward pass (gradient computation)
    loss.backward()
    print(f"✅ Backward pass successful! Gradients computed.")
    
    # Check gradients exist
    total_grad = sum(
        p.grad.norm().item() for p in model.parameters()
        if p.grad is not None
    )
    print(f"Total gradient norm: {total_grad:.4f}")


# ============================================================================
# EXAMPLE 3: Text Generation
# ============================================================================
def example_generation(model):
    """How to generate text with NovaLM-ULTRA."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 3: Text Generation")
    print("="*60)
    
    # Create a prompt
    prompt_text = "Once upon a time,"
    
    # We need a tokenizer. For character-level, we create one:
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}\"' \n\t-_=/\\@#$%^&*~`|<>+"
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    char_to_idx["<PAD>"] = 0
    char_to_idx["<UNK>"] = 0
    
    def encode(text):
        return [char_to_idx.get(c, 0) for c in text]
    
    def decode(ids):
        rev_map = {v: k for k, v in char_to_idx.items()}
        return "".join(rev_map.get(i, "") for i in ids)
    
    # Encode prompt
    prompt_ids = torch.tensor([encode(prompt_text)], dtype=torch.long)
    
    # Generate text
    print(f"Prompt: '{prompt_text}'")
    print("Generating...")
    
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    output_text = decode(output_ids[0].tolist())
    print(f"Generated: '{output_text}'")


# ============================================================================
# EXAMPLE 4: Training on Custom Data
# ============================================================================
def example_training(model):
    """How to train NovaLM-ULTRA on custom text data."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 4: Training on Custom Data")
    print("="*60)
    
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    
    # Create a tiny dataset from scratch
    class TinyTextDataset(Dataset):
        def __init__(self, text, block_size=32):
            # Convert text to token IDs (simple char-level)
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}\"' \n\t-_=/\\@#$%^&*~`|<>+"
            char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
            self.data = torch.tensor(
                [char_to_idx.get(c, 0) for c in text],
                dtype=torch.long,
            )
            self.block_size = block_size
        
        def __len__(self):
            return max(0, len(self.data) - self.block_size)
        
        def __getitem__(self, idx):
            chunk = self.data[idx:idx + self.block_size]
            return {"input_ids": chunk, "labels": chunk}
    
    # Sample training text
    sample_text = "The quick brown fox jumps over the lazy dog. " * 50
    dataset = TinyTextDataset(sample_text, block_size=model.config.max_seq_len)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    # Training loop (1 epoch)
    model.train()
    print("Training for 1 epoch on sample text...")
    
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        optimizer.zero_grad()
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"✅ Training complete! Average loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 5: Save and Load Model
# ============================================================================
def example_save_load(model):
    """How to save and load model checkpoints."""
    print("\n" + "="*60)
    print("📦 EXAMPLE 5: Save/Load Model")
    print("="*60)
    
    import tempfile
    
    # Save model state dict
    save_path = os.path.join(tempfile.gettempdir(), "novaultra_test.pt")
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "dim": model.config.dim,
            "num_layers": model.config.num_layers,
            "num_heads": model.config.num_heads,
            "n_kv_heads": model.config.n_kv_heads,
            "vocab_size": model.config.vocab_size,
            "ffn_dim": model.config.ffn_dim,
            "weight_share_iterations": model.config.weight_share_iterations,
        },
    }, save_path)
    print(f"✅ Model saved to: {save_path}")
    
    # Load model
    from NovaLM_ULTRA import UltraConfig, NovaUltraModel
    
    checkpoint = torch.load(save_path, map_location="cpu")
    loaded_config = UltraConfig(**checkpoint["config"])
    loaded_model = NovaUltraModel(loaded_config)
    loaded_model.load_state_dict(checkpoint["model_state"])
    print(f"✅ Model loaded successfully!")
    print(f"   Loaded model params: {loaded_model.total_params:,}")
    
    # Cleanup
    os.remove(save_path)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="NovaLM-ULTRA Example Script")
    parser.add_argument("--config", type=str, default="balanced",
                        choices=["fast", "balanced"],
                        help="Model config to use")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load")
    args = parser.parse_args()
    
    print("\n" + "█"*60)
    print("  ██████╗  ██████╗ ██╗   ██╗ █████╗ ██╗     ███╗   ███╗")
    print("  ██╔══██╗██╔═══██╗██║   ██║██╔══██╗██║     ████╗ ████║")
    print("  ██████╔╝██║   ██║██║   ██║███████║██║     ██╔████╔██║")
    print("  ██╔══██╗██║   ██║╚██╗ ██╔╝██╔══██║██║     ██║╚██╔╝██║")
    print("  ██║  ██║╚██████╔╝ ╚████╔╝ ██║  ██║███████╗██║ ╚═╝ ██║")
    print("  ╚═╝  ╚═╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝")
    print("  ULTRA - MASTER ARCHITECTURE v1.0")
    print("█"*60)
    
    # Step 1: Create model
    from NovaLM_ULTRA import UltraConfig
    config_map = {
        "fast": UltraConfig.fast_config(),
        "balanced": UltraConfig.balanced_config(),
    }
    config = config_map[args.config]
    
    if args.checkpoint:
        # Load from checkpoint
        from NovaLM_ULTRA import NovaUltraModel
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        loaded_config = UltraConfig(**checkpoint["config"])
        model = NovaUltraModel(loaded_config)
        model.load_state_dict(checkpoint["model_state"])
        print(f"\n✅ Loaded model from: {args.checkpoint}")
    else:
        # Create fresh model
        from NovaLM_ULTRA import create_ultra_model
        model = create_ultra_model(config)
    
    # Run all examples
    example_basic_creation()
    example_forward_pass(model)
    example_generation(model)
    example_training(model)
    example_save_load(model)
    
    print("\n" + "="*60)
    print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train on a real dataset:")
    print("     python scripts/train_hf_dataset.py --dataset roneneldan/TinyStories --config balanced")
    print("  2. Use a HuggingFace tokenizer:")
    print("     python scripts/train_hf_dataset.py --dataset HuggingFaceFW/fineweb --config balanced --use-hf-tokenizer")
    print("  3. Generate text with your trained model:")
    print("     python scripts/example_usage.py --checkpoint checkpoints/best_model.pt")
    print()


if __name__ == "__main__":
    main()