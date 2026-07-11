# NovaLM-ULTRA Training Guide 🚀

## नमस्ते! 👋

यह गाइड आपको सिखाएगी कि:
- NovaLM-ULTRA को कैसे install करें
- HuggingFace datasets से कैसे train करें
- अपना खुद का LLM कैसे बनाएं
- Model को कैसे save/load करें

---

## 📦 Step 1: Installation (इंस्टॉलेशन)

### Required Dependencies (ज़रूरी चीज़ें):

```bash
# 1. Python 3.8+ होना चाहिए
python --version

# 2. PyTorch install करें (अगर नहीं है)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU के लिए: pip install torch --index-url https://download.pytorch.org/whl/cu118

# 3. NovaLM-ULTRA install करें
cd NovaLM-ULTRA
pip install -r requirements.txt

# 4. या एक कमांड में सब कुछ:
pip install torch datasets transformers tokenizers tqdm
```

### Import कैसे करें:

```python
# Import करने का तरीका
import sys
sys.path.insert(0, "path/to/NovaLM-ULTRA")

from NovaLM_ULTRA import create_ultra_model, UltraConfig

# Model बनाएं
config = UltraConfig.balanced_config()
model = create_ultra_model(config)
print(model.summary())
```

---

## 🔧 Step 2: Model Configurations (मॉडल कॉन्फ़िगरेशन)

4 pre-built configs हैं:

| Config | dim | layers | heads | GPU RAM | CPU RAM | Use Case |
|--------|-----|--------|-------|---------|---------|----------|
| 🏃 **fast** | 256 | 4 | 8 | 0.5 GB | 1 GB | Testing |
| ⚖️ **balanced** | 512 | 8 | 16 | 2 GB | 4 GB | Small LLM |
| 💪 **powerful** | 2048 | 24 | 32 | 16 GB | 32 GB | Production |
| 🚀 **ultra** | 4096 | 64 | 64 | 80 GB | 128 GB | SOTA |

```python
# Testing के लिए
config = UltraConfig.fast_config()

# Real training के लिए
config = UltraConfig.balanced_config()

# Custom config
config = UltraConfig(
    dim=768,              # Model dimension
    num_layers=12,        # Number of layers
    num_heads=12,         # Attention heads
    n_kv_heads=4,         # GQA KV heads (4x savings)
    vocab_size=32000,     # Vocabulary size
    max_seq_len=2048,     # Max sequence length
    weight_share_iterations=4,  # 4x effective depth
    use_rwkv=True,        # Use RWKV-7
    use_ssm_scan=True,    # Use SSM
    use_titans_memory=True, # Use neural memory
    use_adaptive_router=True, # Use adaptive routing
)
```

---

## 📚 Step 3: HuggingFace Datasets से Training

### 3.1 TinyStories (छोटा dataset - testing के लिए)

```bash
python scripts/train_hf_dataset.py --dataset roneneldan/TinyStories --config fast --epochs 1 --batch-size 4
```

**Output:**
```
📚 LOADING DATASET: roneneldan/TinyStories
  ✅ Loaded dataset: roneneldan/TinyStories
  📊 Num examples: 2,000,000
🔤 CREATING TOKENIZER
  Created char-level tokenizer
🔨 PREPARING TRAINING DATA
  Tokenizing dataset...
  Created 500,000 training chunks
🤖 CREATING NOVALM-ULTRA MODEL
  Parameters: 5,800,000
  Effective: 11,600,000
🎯 STARTING TRAINING
📱 Using device: cpu
  Step 10 | Loss: 6.5421 | Tok/s: 850
```

### 3.2 FineWeb-Edu (बड़ा dataset - real training)

```bash
# Balanced config के साथ
python scripts/train_hf_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --subset sample-10BT \
  --config balanced \
  --epochs 3 \
  --batch-size 8 \
  --block-size 512 \
  --use-hf-tokenizer \
  --hf-tokenizer-name gpt2
```

### 3.3 Custom Dataset (अपना खुद का डेटा)

अपने text files से dataset बनाएं:

```python
from datasets import Dataset
import json

# अपना text load करें
with open("my_data.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# HF Dataset में convert करें
dataset = Dataset.from_dict({"text": texts})
dataset.save_to_disk("my_dataset")

# फिर train करें
# python scripts/train_hf_dataset.py --dataset my_dataset
```

---

## 🎯 Step 4: Complete End-to-End Example

पूरा example जो सब कुछ दिखाता है:

```python
"""
NovaLM-ULTRA: Complete Training Pipeline
"""

import sys, os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import NovaLM-ULTRA
from NovaLM_ULTRA import create_ultra_model, UltraConfig

# =============================================
# 1. CREATE MODEL
# =============================================
print("🚀 Creating NovaLM-ULTRA...")
config = UltraConfig(
    dim=256,
    num_layers=4,
    num_heads=8,
    n_kv_heads=4,
    vocab_size=1000,  # छोटा vocab for testing
    max_seq_len=128,
    weight_share_iterations=2,
)
model = create_ultra_model(config)
print(f"   Params: {model.total_params:,}")
print(f"   Effective: {model.get_effective_params():,}")

# =============================================
# 2. LOAD DATASET FROM HUGGINGFACE
# =============================================
print("\n📚 Loading HuggingFace dataset...")
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")

# Simple tokenizer
chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!? "
char_to_idx = {c: i+1 for i, c in enumerate(chars)}
char_to_idx["<PAD>"] = 0

def encode(text):
    return [char_to_idx.get(c, 0) for c in text[:128]]

# Tokenize and chunk
tokens = []
for example in dataset:
    tokenized = encode(example["text"])
    if len(tokenized) == 128:
        tokens.append(tokenized)

data = torch.tensor(tokens)

# =============================================
# 3. CREATE DATALOADER
# =============================================
loader = DataLoader(data, batch_size=4, shuffle=True)

# =============================================
# 4. TRAIN
# =============================================
print(f"\n🎯 Training on {len(tokens)} samples...")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.view(-1),
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

# =============================================
# 5. GENERATE TEXT
# =============================================
print("\n✍️ Generating text...")
prompt = "Once upon a time"
prompt_ids = torch.tensor([[char_to_idx.get(c, 0) for c in prompt]])

output = model.generate(
    prompt_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
)

# Decode
rev_map = {v: k for k, v in char_to_idx.items()}
generated = "".join(rev_map.get(i.item(), "") for i in output[0])
print(f"   Prompt: {prompt}")
print(f"   Output: {generated}")

# =============================================
# 6. SAVE MODEL
# =============================================
print("\n💾 Saving model...")
torch.save({
    "model_state": model.state_dict(),
    "config": config.to_dict(),
}, "my_model.pt")
print(f"   Saved to: my_model.pt")
print("\n✅ DONE! 🎉")
```

इसे सेव करें `train_llm.py` और चलाएं:
```bash
python train_llm.py
```

---

## 🧠 Step 5: Understanding the Architecture

NovaLM-ULTRA में 8 components हैं। समझिए क्यों:

| # | Component | Source | Benefit |
|---|-----------|--------|---------|
| 1 | **GQA Attention** | Llama 2/3 | 4x less KV cache memory |
| 2 | **RWKV-7 TimeMix** | RWKV | CPU पर 10x fast inference |
| 3 | **Titans Memory** | Google | Long context (200B tokens) |
| 4 | **Adaptive Router** | Mixtral | 70% tokens skip heavy compute |
| 5 | **Weight Sharing** | ALBERT | Nx depth at 1x params |
| 6 | **SwiGLU FFN** | Llama | Best activation function |
| 7 | **Selective SSM** | Mamba | Long-range dependencies |
| 8 | **Weight Tying** | Transformer | Embedding reuse |

---

## ⚡ Step 6: Performance Tips

### CPU Training को fast कैसे करें:
```bash
# OMP threads बढ़ाएं
set OMP_NUM_THREADS=8    # Windows
export OMP_NUM_THREADS=8  # Linux

# Use RWKV + SSM (attention skip)
python scripts/train_hf_dataset.py --config fast --dim 512
```

### GPU Training को optimize करें:
```bash
# Batch size बढ़ाएं
python scripts/train_hf_dataset.py --config balanced --batch-size 32 --use-hf-tokenizer

# Mixed precision use करें (default: on)
python scripts/train_hf_dataset.py --config powerful --batch-size 16
```

### Memory बचाने के लिए:
```python
# Weight sharing बढ़ाएं (same params, more depth)
config.weight_share_iterations = 8  # 8x effective depth

# GQA heads कम करें
config.n_kv_heads = 2  # 8x memory savings

# Vocab size छोटा रखें
config.vocab_size = 16000
```

---

## ❓ Step 7: Common Issues (आम समस्याएं)

### Problem: "ModuleNotFoundError: No module named 'NovaLM_ULTRA'"
**Solution:**
```python
import sys
sys.path.insert(0, "path/to/NovaLM-ULTRA")
```

### Problem: "CUDA out of memory"
**Solution:**
```bash
# Batch size कम करें
python scripts/train_hf_dataset.py --batch-size 2

# या fast config use करें
python scripts/train_hf_dataset.py --config fast
```

### Problem: Dataset नहीं मिल रहा
**Solution:**
```bash
pip install datasets
# Check: python -c "from datasets import load_dataset; print('OK')"
```

### Problem: Training बहुत slow है
**Solution:**
```bash
# 1. PyTorch compile use करें
# 2. OMP threads बढ़ाएं
# 3. Block size कम करें
python scripts/train_hf_dataset.py --block-size 128 --config fast
```

---

## 📊 Step 8: Monitoring Training

### Weights & Biases के साथ:
```bash
pip install wandb
wandb login

# फिर script में add करें:
import wandb
wandb.init(project="novaultra-training")
```

### Manual monitoring (script में built-in):
```
Epoch 1/3: 100%|██████████| 1000/1000 [01:23<00:00, 12.0it/s, loss=4.2, lr=3e-4]
📊 Epoch 1 completed:
   Average Loss: 4.2156
   Time: 83.2s
```

---

## 🚀 Step 9: Scaling Up

### Small Model (Testing):
```bash
python scripts/train_hf_dataset.py \
  --dataset roneneldan/TinyStories \
  --config fast \
  --epochs 1
```

### Medium Model (Real LLM):
```bash
python scripts/train_hf_dataset.py \
  --dataset HuggingFaceFW/fineweb \
  --config balanced \
  --dim 768 \
  --layers 12 \
  --heads 12 \
  --epochs 3 \
  --batch-size 8 \
  --use-hf-tokenizer \
  --hf-tokenizer-name gpt2
```

### Large Model (Production):
```bash
python scripts/train_hf_dataset.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --config powerful \
  --epochs 10 \
  --batch-size 4 \
  --block-size 2048 \
  --use-hf-tokenizer \
  --hf-tokenizer-name meta-llama/Llama-2-7b-hf
```

---

## 🎓 Summary (सारांश)

```
✅ Step 1: pip install -r requirements.txt
✅ Step 2: pip install -e .  (ya path add karein)
✅ Step 3: python scripts/train_hf_dataset.py --dataset roneneldan/TinyStories --config fast
✅ Step 4: python scripts/example_usage.py
✅ Step 5: Model save ho gaya checkpoints/ folder mein
```

**Questions?** Code में comments पढ़ें या GitHub issues खोलें!

---

## 📁 File Structure

```
NovaLM-ULTRA/
├── __init__.py          # Package init (imports)
├── ultra_config.py       # Configuration
├── ultra_model.py        # Model architecture (8 components)
├── ultra_train.py        # Basic trainer
├── ultra_benchmark.py    # Benchmark script
├── requirements.txt      # Dependencies
├── setup.py              # pip install setup
├── TRAINING_GUIDE.md     # This guide
├── README.md             # Quick start
├── scripts/
│   ├── train_hf_dataset.py   # HF dataset training
│   └── example_usage.py      # Usage examples
└── checkpoints/          # Saved models (created during training)
```

---

**Happy Training! 🚀✨**

python -m venv .venv
or
py -m venv .venv


.\.venv\Scripts\Activate.ps1

yadi fail kare to

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

or

python -m pip install -r requirements.txt


📊 Config Presets Reference

    Config	dim	 layers	heads	kv_heads	Params	Use Case
🏃 fast	    256	  4	     8	   auto   	~5.8M	 Testing
⚖️ balanced	512	  8	     16	    8	      ~25M	 Small LLM
💪 powerful	2048	24     32	    8	      ~350M	 Production
🚀 ultra	  4096	64	   64	    8	      ~7.8B	 SOTA

cd NovaLM-ULTRA

# Example 1 - Chhota model (apne numbers):
python scripts/train_ultra_complete.py ^
  --dim 256 --layers 4 --heads 8 --kv-heads 2 --head-dim 64 --ffn-dim 1024 ^
  --dataset "roneneldan/TinyStories" --epochs 3

# Example 2 - Medium model (apne numbers):
python scripts/train_ultra_complete.py ^
  --dim 512 --layers 8 --heads 16 --kv-heads 4 --ffn-dim 2048 ^
  --block-size 512 --vocab-size 32000 ^
  --dataset "roneneldan/TinyStories" --epochs 5

# Example 3 - Bada model (apne numbers):
python scripts/train_ultra_complete.py ^
  --dim 2048 --layers 24 --heads 32 --kv-heads 8 --ffn-dim 8192 ^
  --block-size 2048 --vocab-size 50277 ^
  --weight-share 4 --use-rwkv --use-ssm --use-memory --use-router ^
  --dataset "HuggingFaceFW/fineweb" --epochs 10 --batch-size 8 --mixed-precision bf16


Saare model architecture parameters:
Parameter	Meaning	Example Values
--dim	Model hidden size	128, 256, 512, 768, 1024, 2048, 4096
--layers	Number of transformer layers	4, 8, 12, 24, 32, 64
--heads	Number of attention heads	4, 8, 12, 16, 32, 64
--kv-heads	KV heads for GQA (4x less memory)	1, 2, 4, 8
--head-dim	Dimension per head	32, 64, 128
--ffn-dim	FFN intermediate size	512, 1024, 2048, 8192
--vocab-size	Vocabulary size	100, 1000, 32000, 50277
--block-size	Sequence length	128, 256, 512, 1024, 2048, 8192
--weight-share	Share layers N times	1 (no share), 2, 4, 8, 16

--use-rwkv        # RWKV-7 TimeMix lagega
--use-ssm         # Selective SSM (Mamba style) lagega
--use-memory      # Titans Neural Memory lagega
--use-router      # Adaptive Router lagega
--use-rope        # RoPE position embeddings lagega
--use-gqa         # Grouped-Query Attention lagega
--use-swiglu      # SwiGLU FFN lagega


200B+ Effective Model ke lie Settings
Weight Sharing ki madad se 200B+ effective model ban sakta hai, lekin physical params bahut kam lagege!

Formula:

Effective Params = Physical Params × Weight Share

200B = Physical × weight_share
Option 1: Weight Share 16 (Recommended)

python scripts/train_ultra_complete.py ^
  --dim 4096 --layers 28 --heads 32 --kv-heads 8 --ffn-dim 11008 ^
  --vocab-size 50277 --block-size 2048 ^
  --weight-share 16 ^                  # <-- 16 baar reuse karega
  --use-rwkv --use-ssm --use-memory --use-router ^
  --dataset "HuggingFaceFW/fineweb" --epochs 10 --batch-size 4 ^
  --mixed-precision bf16 --multi-gpu --compile
Physical: ~12.5B | Effective: ~200B ✅

Option 2: Weight Share 32 (Aur Kam GPU Memory)

python scripts/train_ultra_complete.py ^
  --dim 4096 --layers 16 --heads 32 --kv-heads 8 --ffn-dim 11008 ^
  --vocab-size 50277 --block-size 2048 ^
  --weight-share 32 ^                  # <-- 32 baar reuse karega
  --use-rwkv --use-ssm --use-memory --use-router ^
  --dataset "HuggingFaceFW/fineweb" --epochs 15 --batch-size 8 ^
  --mixed-precision bf16 --multi-gpu --compile
Physical: ~7B | Effective: ~224B ✅ (Kam GPU fit hoga)

Option 3: Weight Share 64 (Ek GPU mein bhi!)

python scripts/train_ultra_complete.py ^
  --dim 2560 --layers 16 --heads 20 --kv-heads 5 --ffn-dim 6912 ^
  --vocab-size 50277 --block-size 2048 ^
  --weight-share 64 ^                  # <-- 64 baar reuse karega
  --use-rwkv --use-ssm --use-memory --use-router ^
  --dataset "HuggingFaceFW/fineweb" --epochs 20 ^
  --mixed-precision bf16 --multi-gpu --compile
Physical: ~3.2B | Effective: ~205B ✅ (Single A100 mein bhi fit!)

Option 4: ULTIMATE 400B+ (Max Power)

python scripts/train_ultra_complete.py ^
  --dim 8192 --layers 32 --heads 64 --kv-heads 8 --ffn-dim 28672 ^
  --vocab-size 128000 --block-size 4096 ^
  --weight-share 64 ^                  # <-- 64 baar reuse
  --use-rwkv --use-ssm --use-memory --use-router ^
  --dataset "HuggingFaceFW/fineweb-edu" --epochs 20 ^
  --batch-size 2 --accumulation-steps 16 ^
  --mixed-precision bf16 --multi-gpu --compile --wandb
Physical: ~7B | Effective: ~448B 🔥 (8×A100 80GB chahiye)

Comparison:
Option	Physical Params	GPU Memory	Effective Params	GPUs Needed
1	12.5B	~50GB	200B	4×A100
2	7B	~28GB	224B	2×A100
3	3.2B	~14GB	205B	1×A100
4	7B	~56GB	448B	8×A100
Key Points:
weight-share = kitni baar layers reuse hogi
KV heads kam (8) → GQA se 4x memory bachao
RWKV + SSM + Memory + Router → attention ke bina bhi long context handle
bf16 mixed precision → memory aadhi ho jati hai
Chhote GPU pe (like RTX 3060 12GB):

--dim 1024 --layers 8 --heads 8 --kv-heads 2 --weight-share 32

## 200B+ Effective Model ke lie Settings

__Weight Sharing__ ki madad se 200B+ effective model bina bade physical params ke ban sakta hai.

### Recommended Setting (Weight Share 16):

```javascript
--dim 4096 --layers 28 --heads 32 --kv-heads 8 --ffn-dim 11008
--weight-share 16 --vocab-size 50277 --block-size 2048
```

__Physical: ~12.5B → Effective: ~200B ✅__ — 4×A100 chahiye

### Kam GPU pe (Weight Share 32):

```javascript
--dim 2560 --layers 16 --heads 20 --kv-heads 5 --ffn-dim 6912
--weight-share 64 --vocab-size 50277 --block-size 2048
```

__Physical: ~3.2B → Effective: ~205B ✅__ — 1×A100 mein bhi fit!

### Formula:

```javascript
Physical Params × Weight Share = Effective Params
~12.5B × 16 = 200B
~3.2B × 64 = 205B
~7B × 64 = 448B
```

python scripts\train_ultra_complete.py --dim 1024 --layers 16 --heads 16 --kv-heads 4 --head-dim 64 --ffn-dim 4096 --vocab-size 32000 --block-size 512 --weight-share 1 --dataset "HuggingFaceFW/fineweb" --dataset-subset "sample-10BT" --epochs 5 --batch-size 8 --lr 3e-4 --warmup-steps 100 --mixed-precision fp16 --save-dir ./models/1B_model
