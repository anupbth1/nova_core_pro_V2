"""
NovaLM-ULTRA v1.0 - Master Architecture
The ULTIMATE LLM combining BEST of ALL architectures:
  ✓ GQA Attention (Llama-proven) 
  ✓ RWKV-7 TimeMix (linear attention)
  ✓ Titans Neural Memory (long context)
  ✓ Adaptive Routing (smart compute allocation)
  ✓ Weight Sharing (200B capacity in 8B params)
  ✓ CPU+GPU Optimized (pure PyTorch)
  ✓ Ultra-Fast Training (0 CUDA dependency)

Usage:
    from NovaLM_ULTRA import create_ultra_model, UltraConfig
    config = UltraConfig.balanced_config()
    model = create_ultra_model(config)
"""

import sys, os

# Add this directory to path (handles hyphen in folder name)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from ultra_config import UltraConfig
from ultra_model import (
    NovaUltraModel,
    create_ultra_model,
    GroupedQueryAttention,
    SwiGLUFFN,
    RWKVTimeMix,
    SelectiveSSM,
    TitansNeuralMemory,
    AdaptiveRouter,
    RotaryEmbedding,
    UltraBlock,
)

# Trainer - optional import
try:
    from ultra_train import UltraTrainer, CharDataset, TinyStoriesDataset
    _has_trainer = True
except ImportError:
    _has_trainer = False

__all__ = [
    # Model
    "NovaUltraModel",
    "create_ultra_model",
    "UltraConfig",
    
    # Components
    "GroupedQueryAttention",
    "SwiGLUFFN", 
    "RWKVTimeMix",
    "SelectiveSSM",
    "TitansNeuralMemory",
    "AdaptiveRouter",
    "RotaryEmbedding",
    "UltraBlock",
    
    # Trainer
    "UltraTrainer",
    "CharDataset",
    "TinyStoriesDataset",
]

__version__ = "1.0.0"