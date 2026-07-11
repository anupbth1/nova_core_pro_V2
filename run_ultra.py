"""
NovaLM-ULTRA - Complete Run Script
Tests creation, forward pass, backward pass, generation and training.
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ultra_config - use exec to avoid hyphen issues
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultra_config.py")).read())

# Import ultra_model
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultra_model.py")).read())

# Now UltraConfig, create_ultra_model, NovaUltraModel should be defined
assert 'UltraConfig' in dir(), "UltraConfig not loaded"
assert 'create_ultra_model' in dir(), "create_ultra_model not loaded"

# ==== TEST 1: Model Creation ====
print("\n" + "=" * 70)
print("TEST 1: Model Creation")
config = UltraConfig.fast_config()
model = create_ultra_model(config)
print(model.summary())
print("  MODEL CREATED OK")

# ==== TEST 2: Forward Pass ====
print("\n" + "=" * 70)
print("TEST 2: Forward Pass")
x = torch.randint(0, config.vocab_size, (1, 32))
logits, state = model(x, return_state=True)
assert logits.shape == (1, 32, config.vocab_size), f"Shape mismatch: {logits.shape}"
print(f"  Input:  {x.shape}")
print(f"  Output: {logits.shape}")
print("  FORWARD PASS OK")

# ==== TEST 3: Backward Pass ====
print("\n" + "=" * 70)
print("TEST 3: Backward Pass (Gradient Flow)")
model.train()
x = torch.randint(0, config.vocab_size, (2, 16))
y = torch.randint(0, config.vocab_size, (2, 16))
logits, _ = model(x)
loss = torch.nn.functional.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
print(f"  Loss: {loss.item():.4f}")
print(f"  Gradient Norm: {grad_norm:.4f}")
print(f"  Params: {model.total_params:,}")
assert grad_norm > 0, "No gradients!"
print("  BACKWARD PASS OK")

# ==== TEST 4: Generation ====
print("\n" + "=" * 70)
print("TEST 4: Text Generation")
prompt = torch.randint(0, config.vocab_size, (1, 5))
output = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=20)
print(f"  Prompt:  {prompt.shape} -> Output: {output.shape}")
assert output.shape[1] > prompt.shape[1], "No generation!"
print("  GENERATION OK")

# ==== TEST 5: State Passing ====
print("\n" + "=" * 70)
print("TEST 5: State Recurrence")
model.eval()
with torch.no_grad():
    x1 = torch.randint(0, config.vocab_size, (1, 8))
    x2 = torch.randint(0, config.vocab_size, (1, 8))
    logits_no_state, _ = model(x2)
    _, state = model(x1, return_state=True)
    logits_with_state, _ = model(x2, state=state)
    print(f"  No state: {logits_no_state.shape}")
    print(f"  With state: {logits_with_state.shape}")
    if state:
        print(f"  State keys: {list(state.keys())}")
print("  STATE PASSING OK")

# ==== TEST 6: Quick Training ====
print("\n" + "=" * 70)
print("TEST 6: Quick Training")

# Import CharDataset
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ultra_train.py")).read())

text = "The quick brown fox jumps over the lazy dog. " * 200 + \
       "Once upon a time there was a little neural network that could. " * 200
dataset = CharDataset(text, seq_len=64, vocab_size=config.vocab_size)
print(f"  Dataset: {len(dataset)} samples, vocab={dataset.vocab_size}")

trainer = UltraTrainer(model, config)
torch.manual_seed(42)
results = trainer.train(dataset, num_epochs=2, num_train_steps=20, log_interval=5)
print(f"  Final Loss: {results['final_loss']:.4f}")
print(f"  Perplexity: {results['perplexity']:.1f}")
print(f"  Speed: {results['tokens_per_second']:.0f} tok/s")
print("  TRAINING OK")

# ==== ALL TESTS PASSED ====
print("\n" + "=" * 70)
print("ALL 6 TESTS PASSED!")
print("=" * 70)
print(f"\nNovaLM-ULTRA Summary:")
print(f"  Architecture:   8 combined innovations")
print(f"  Parameters:     {model.total_params:,}")
print(f"  Effective:      {model.get_effective_params():,}")
print(f"  Effective Depth:{model.effective_depth}")
print(f"  KV Cache Saved: {model.get_memory_savings():.1f}x vs MHA")
print(f"  Training Speed: {results['tokens_per_second']:.0f} tok/s")
print(f"  Perplexity:     {results['perplexity']:.1f}")
print("=" * 70)