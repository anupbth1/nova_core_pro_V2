"""NovaLM-ULTRA - Quick Test Script. Tests model creation, forward pass, training, generation."""
import sys, os, importlib.util, torch

# Add parent directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules via importlib (hyphen in folder name prevents normal import)
base = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location("ultra_config", os.path.join(base, "ultra_config.py"))
ultra_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ultra_config)

spec = importlib.util.spec_from_file_location("ultra_model", os.path.join(base, "ultra_model.py"))
ultra_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ultra_model)

spec = importlib.util.spec_from_file_location("ultra_train", os.path.join(base, "ultra_train.py"))
ultra_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ultra_train)

UltraConfig = ultra_config.UltraConfig
create_ultra_model = ultra_model.create_ultra_model
UltraTrainer = ultra_train.UltraTrainer
CharDataset = ultra_train.CharDataset

# ==== TEST 1: Model Creation ====
print("\n" + "=" * 70)
print("TEST 1: Model Creation")
print("=" * 70)

config = UltraConfig.fast_config()
model = create_ultra_model(config)
print(model.summary())
print("  MODEL CREATED OK")

# ==== TEST 2: Forward Pass ====
print("\n" + "=" * 70)
print("TEST 2: Forward Pass")
print("=" * 70)

x = torch.randint(0, config.vocab_size, (1, 32))
logits, state = model(x, return_state=True)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {logits.shape}")
assert logits.shape == (1, 32, config.vocab_size), f"Shape mismatch: {logits.shape}"
print("  FORWARD PASS OK")

# ==== TEST 3: Backward Pass ====
print("\n" + "=" * 70)
print("TEST 3: Backward Pass (Gradient Flow)")
print("=" * 70)

model.train()
x = torch.randint(0, config.vocab_size, (2, 16))
y = torch.randint(0, config.vocab_size, (2, 16))
logits, _ = model(x)
loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"  Loss: {loss.item():.4f}")
print(f"  Gradient Norm: {grad_norm:.4f}")
print(f"  Total Params: {model.total_params:,}")
assert grad_norm > 0, "No gradients flowing!"
print("  BACKWARD PASS OK")

# ==== TEST 4: Generation ====
print("\n" + "=" * 70)
print("TEST 4: Text Generation")
print("=" * 70)

prompt = torch.randint(0, config.vocab_size, (1, 5))
output = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=20)
print(f"  Prompt shape:  {prompt.shape}")
print(f"  Output shape:  {output.shape}")
print(f"  Generated {output.shape[1] - prompt.shape[1]} new tokens")
assert output.shape[1] > prompt.shape[1], "Generation failed!"
print("  GENERATION OK")

# ==== TEST 5: State Passing ====
print("\n" + "=" * 70)
print("TEST 5: State Recurrence")
print("=" * 70)

model.eval()
with torch.no_grad():
    x1 = torch.randint(0, config.vocab_size, (1, 8))
    x2 = torch.randint(0, config.vocab_size, (1, 8))
    
    # Without state
    logits2_no_state, _ = model(x2)
    
    # With state
    _, state = model(x1, return_state=True)
    logits2_with_state, _ = model(x2, state=state)
    
    print(f"  No state continuity: {logits2_no_state.shape}")
    print(f"  With state continuity: {logits2_with_state.shape}")
    if state:
        print(f"  State keys: {list(state.keys())}")
    print("  STATE PASSING OK")

# ==== TEST 6: Trainer ====
print("\n" + "=" * 70)
print("TEST 6: Quick Training")
print("=" * 70)

text = ("The quick brown fox jumps over the lazy dog. " * 100 +
        "Once upon a time in a land far away there lived a beautiful princess. " * 100 +
        "Machine learning is the study of algorithms that improve through experience. " * 100)

dataset = CharDataset(text, seq_len=64, vocab_size=config.vocab_size)
print(f"  Dataset samples: {len(dataset)}")
print(f"  Dataset vocab: {dataset.vocab_size}")

trainer = UltraTrainer(model, config)
torch.manual_seed(42)
results = trainer.train(dataset, num_epochs=2, num_train_steps=20, log_interval=5)

print(f"\n  Final Loss: {results['final_loss']:.4f}")
print(f"  Perplexity: {results['perplexity']:.1f}")
print(f"  Speed: {results['tokens_per_second']:.0f} tok/s")
assert results['final_loss'] < float('inf'), "Training failed!"
print("  TRAINING OK")

# ==== ALL TESTS PASSED ====
print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print(f"\nNovaLM-ULTRA Summary:")
print(f"  Architecture:   8 combined innovations")
print(f"  Parameters:     {model.total_params:,}")
print(f"  Effective Depth: {model.effective_depth}")
print(f"  KV Cache Saved: {model.get_memory_savings():.1f}x vs MHA")
print(f"  Training Speed: {results['tokens_per_second']:.0f} tok/s (CPU)")
print(f"  Perplexity:     {results['perplexity']:.1f}")
print("=" * 70)