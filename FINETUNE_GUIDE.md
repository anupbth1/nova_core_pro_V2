# NovaLM-ULTRA Fine-tuning & Inference Guide 🚀

## किसी भी Existing LLM को Fine-tune और Run कैसे करें

---

## 📋 TABLE OF CONTENTS

1. [CPU vs GPU Training](#1-cpu-vs-gpu-training)
2. [Fine-tune Existing Models](#2-fine-tune-existing-models)
3. [Run/Inference LLM](#3-runinference-llm)
4. [Can We Build >8B Models?](#4-can-we-build-8b-models)
5. [Memory Requirements](#5-memory-requirements)

---

## 1. CPU vs GPU Training

### 🖥️ CPU Training (धीमा लेकिन सबके पास है)
```
✅ Pros: किसी भी computer पर चलेगा, कोई extra hardware नहीं चाहिए
❌ Cons: बहुत slow, बड़े models नहीं चलेंगे
📊 Speed: ~50-500 tokens/second (GPT-2 size)
```

### 🎮 GPU Training (तेज़ लेकिन महंगा)
```
✅ Pros: 10-50x faster, बड़े models चला सकते हैं
❌ Cons: GPU चाहिए (NVIDIA RTX/GTX या Apple MPS)
📊 Speed: ~5000-50000 tokens/second (GPT-2 on RTX 3060)
```

### कब क्या use करें:

| Scenario | Recommended | Command |
|----------|------------|---------|
| **No GPU** (CPU only) | NovaLM-ULTRA (fast config) or GPT-2 fine-tune | `--device cpu --batch-size 2 --epochs 2` |
| **4-6 GB GPU** (GTX 1060, RTX 3050) | NovaLM-ULTRA (balanced) or GPT-2 fine-tune | `--device cuda --mixed-precision fp16` |
| **8-12 GB GPU** (RTX 3060, 3070) | Llama-2-7B with LoRA + 4bit | `--lora --quantize 4bit --batch-size 4` |
| **16-24 GB GPU** (RTX 3090, 4080) | Full fine-tune Llama-2-7B or Mistral-7B | `--batch-size 8 --mixed-precision bf16` |
| **48-80 GB GPU** (A100, 4090) | Full fine-tune 13B-40B models | `--batch-size 16 --multi-gpu` |

### CPU Training को Fast करने के Tips:

```bash
# 1. CPU threads बढ़ाएं
set OMP_NUM_THREADS=8  # Windows
export OMP_NUM_THREADS=8  # Linux/Mac

# 2. Batch size कम रखें
--batch-size 2

# 3. Block size कम करें
--block-size 128

# 4. FP32 use करें (mixed precision CPU पर काम नहीं करता)
--mixed-precision no

# 5. छोटा model use करें (GPT-2, distilgpt2)
--model gpt2

# Example:
python scripts/finetune_existing_llm.py --model distilgpt2 --dataset roneneldan/TinyStories --device cpu --batch-size 2 --block-size 128 --epochs 2 --cpu-threads 8
```

---

## 2. Fine-tune Existing Models

### 2.1 GPT-2 (CPU पर चलेगा, सबसे आसान)

```bash
# CPU पर
python scripts/finetune_existing_llm.py \
  --model gpt2 \
  --dataset roneneldan/TinyStories \
  --device cpu \
  --epochs 2 \
  --batch-size 2 \
  --block-size 256 \
  --generate --prompt "Once upon a time"

# GPU पर (10x faster)
python scripts/finetune_existing_llm.py \
  --model gpt2 \
  --dataset roneneldan/TinyStories \
  --epochs 5 \
  --batch-size 8 \
  --block-size 512 \
  --generate --prompt "The little girl"
```

### 2.2 Llama-2-7B (GPU चाहिए, LoRA से memory बचाएं)

```bash
# 4-bit quantization + LoRA (सिर्फ 6 GB GPU RAM)
python scripts/finetune_existing_llm.py \
  --model meta-llama/Llama-2-7b-hf \
  --dataset HuggingFaceFW/fineweb \
  --epochs 3 \
  --lora --rank 8 --alpha 16 \
  --quantize 4bit \
  --batch-size 4 \
  --mixed-precision fp16
```

### 2.3 Mistral-7B (Best quality for fine-tuning)

```bash
# LoRA with 8-bit quantization
python scripts/finetune_existing_llm.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset-subset sample-10BT \
  --epochs 5 \
  --lora --rank 16 \
  --quantize 8bit \
  --batch-size 4
```

### 2.4 Custom Dataset (अपना डेटा)

```bash
# अपने text files से
python scripts/finetune_existing_llm.py \
  --model gpt2 \
  --dataset my_dataset \  # HF dataset ya local folder
  --text-column text \
  --epochs 5
```

### 2.5 NovaLM-ULTRA को Train करें (from scratch)

```bash
# CPU पर (fast config)
python scripts/train_ultra_complete.py \
  --dataset roneneldan/TinyStories \
  --model-name "MyNovaLM" \
  --dim 256 --layers 4 --heads 8 \
  --epochs 1 --batch-size 4 --device cpu

# GPU पर (balanced config)
python scripts/train_ultra_complete.py \
  --dataset roneneldan/TinyStories \
  --model-name "MyNovaLM" \
  --dim 512 --layers 8 --heads 16 \
  --epochs 3 --batch-size 16 \
  --mixed-precision fp16
```

### 2.6 List All Supported Models

```bash
python scripts/finetune_existing_llm.py --list-models
```

---

## 3. Run/Inference LLM

### 3.1 Trained Model को Run करें (NovaLM-ULTRA)

```bash
# Text generate करें
python scripts/train_ultra_complete.py \
  --resume checkpoints/MyNovaLM/best_model.pt \
  --generate \
  --prompt "Ek baar ki baat hai" \
  --gen-tokens 200 \
  --gen-temperature 0.8

# या
python scripts/example_usage.py \
  --checkpoint checkpoints/MyNovaLM/best_model.pt
```

### 3.2 Fine-tuned GPT-2 को Run करें

```python
# simple_run.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model load करें
model_name = "gpt2"  # या अपना fine-tuned model path
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Generate
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.3 Any Llama/Mistral Model को Run करें

```python
# run_llama.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model load
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",  # या fine-tuned path
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Generate
prompt = "What is the meaning of life?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3.4 Web Interface (Gradio)

```python
# web_ui.py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # अपना model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate(prompt, temp=0.7, tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=tokens,
        temperature=temp,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(0.1, 2.0, 0.7, label="Temperature"),
        gr.Slider(10, 500, 100, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="My LLM - Text Generation",
).launch(share=True)  # share=True से public link मिलेगा
```

```bash
# Install gradio
pip install gradio

# Run
python web_ui.py
# खुलेगा: http://localhost:7860
```

### 3.5 API Server (FastAPI)

```python
# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
# Then: curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Hello", "max_tokens": 50}'
```

---

## 4. Can We Build >8B Models?

**हाँ! बिल्कुल बना सकते हैं!** 8B से ऊपर के models भी बन सकते हैं।

### How to Build Models Larger than 8B:

#### 🚀 NovaLM-ULTRA Architecture (Weight Sharing से बड़ा model)

NovaLM-ULTRA में **weight sharing** है - इसका मतलब:
- **Physical params**: 7.8B (जितने weights store होते हैं)
- **Effective params**: 125B+ (जितने weights effectively use होते हैं)

```
Physical Params = dim² × layers × components
Effective Params = Physical × weight_share_iterations

Example:
dim=4096, layers=64, weight_share=16
Physical: ~7.8B params
Effective: ~125B params (16x depth!)
```

#### 🏗️ Ultra Config (4096 dim, 64 layers, 16x weight sharing)

```bash
python scripts/train_ultra_complete.py \
  --dataset HuggingFaceFW/fineweb-edu \
  --model-name "NovaUltra-125B" \
  --dim 4096 --layers 64 --heads 64 --kv-heads 8 \
  --vocab-size 50277 \
  --weight-share 16 \
  --block-size 8192 \
  --epochs 10 \
  --multi-gpu \
  --mixed-precision bf16 \
  --compile \
  --wandb
```

#### 💪 Powerful Config (2048 dim, 24 layers, 8x weight sharing)

```bash
python scripts/train_ultra_complete.py \
  --dataset HuggingFaceFW/fineweb \
  --model-name "NovaPowerful" \
  --dim 2048 --layers 24 --heads 32 --kv-heads 8 \
  --vocab-size 50277 \
  --weight-share 8 \
  --block-size 4096 \
  --epochs 10 \
  --multi-gpu \
  --mixed-precision bf16
```

#### 🔧 Custom Big Model (whatever you want)

```python
# Custom config - आप अपनी मर्ज़ी का model बना सकते हैं
config = UltraConfig(
    dim=8192,           # 8K dimension (GPT-3 size)
    num_layers=128,      # 128 layers
    num_heads=128,       # 128 heads
    n_kv_heads=8,        # GQA: 16x savings
    ffn_dim=32768,       # 32K FFN
    vocab_size=128000,   # Large vocab
    max_seq_len=131072,  # 128K context
    weight_share_iterations=32,  # 32x effective depth
)
# Effective: 8192² × 128 × 32 ≈ 274B parameters!
```

### Memory Requirements for Different Sizes:

| Model Size | GPU RAM Needed | CPU RAM Needed | Training Time (1 epoch on 100B tokens) |
|-----------|---------------|---------------|--------------------------------------|
| **1B** | 4 GB | 8 GB | 2 days (1 GPU) |
| **7B** | 16 GB | 64 GB | 2 weeks (8 GPUs) |
| **13B** | 24 GB | 128 GB | 1 month (8 GPUs) |
| **70B** | 80 GB | 512 GB | 3 months (32 GPUs) |
| **125B** | 160 GB | 1 TB | 6 months (64 GPUs) |
| **1 Trillion** | 2 TB | 10 TB | 1 year (512 GPUs) |

### Distributed Training Commands:

```bash
# 2 GPUs
python scripts/train_ultra_complete.py --dim 2048 --layers 24 --multi-gpu --gpu-ids 0,1

# 4 GPUs
python scripts/train_ultra_complete.py --dim 4096 --layers 32 --multi-gpu --gpu-ids 0,1,2,3

# 8 GPUs
python scripts/train_ultra_complete.py --dim 8192 --layers 64 --multi-gpu --gpu-ids 0,1,2,3,4,5,6,7
```

---

## 5. Memory Requirements

### Model Size vs Memory Table

| Model | Params | FP32 | FP16 | 8-bit | 4-bit |
|-------|--------|------|------|-------|-------|
| **NovaLM-ULTRA (fast)** | 5.8M | 23 MB | 12 MB | 6 MB | 3 MB |
| **NovaLM-ULTRA (balanced)** | 25M | 100 MB | 50 MB | 25 MB | 13 MB |
| **NovaLM-ULTRA (powerful)** | 350M | 1.4 GB | 700 MB | 350 MB | 175 MB |
| **NovaLM-ULTRA (ultra)** | 7.8B | 31 GB | 16 GB | 8 GB | 4 GB |
| **GPT-2** | 124M | 500 MB | 250 MB | 125 MB | 63 MB |
| **Llama-2-7B** | 7B | 28 GB | 14 GB | 7 GB | 3.5 GB |
| **Llama-2-13B** | 13B | 52 GB | 26 GB | 13 GB | 6.5 GB |
| **Llama-3-70B** | 70B | 280 GB | 140 GB | 70 GB | 35 GB |
| **Mixtral-8x7B** | 47B | 188 GB | 94 GB | 47 GB | 24 GB |

### Memory Calculation Formula:

```
FP32:  params × 4 bytes
FP16:  params × 2 bytes
8-bit: params × 1 byte
4-bit: params × 0.5 bytes

Training needs ~4x more memory (optimizer states + gradients)
```

---

## 🎯 Quick Summary

```
CPU पर Train करना है?
→ python scripts/finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --device cpu --batch-size 2

GPU पर Train करना है?
→ python scripts/finetune_existing_llm.py --model gpt2 --dataset roneneldan/TinyStories --batch-size 8

Llama-2 fine-tune करना है?
→ python scripts/finetune_existing_llm.py --model meta-llama/Llama-2-7b-hf --lora --quantize 4bit --dataset HuggingFaceFW/fineweb

8B से बड़ा model बनाना है?
→ python scripts/train_ultra_complete.py --dim 4096 --layers 64 --weight-share 16 --multi-gpu --wandb

Model को run/inference करना है?
→ python scripts/train_ultra_complete.py --resume checkpoints/model/best_model.pt --generate --prompt "Your prompt"
```

**अपना LLM बनाने में कोई limit नहीं है!** 🚀