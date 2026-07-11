"""
NovaLM-ULTRA v1.0 - Setup Script
Install with: pip install -e .
"""
from setuptools import setup, find_packages
import os

# Read requirements
with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="NovaLM-ULTRA",
    version="1.0.0",
    description="NovaLM-ULTRA - Master Architecture combining 8 best LLM architectures",
    long_description="""
    NovaLM-ULTRA combines 8 proven architectures into ONE optimized LLM:
    1. Grouped-Query Attention (GQA) - 4x KV cache reduction
    2. RWKV-7 TimeMix - Linear attention for fast CPU inference
    3. Titans Neural Memory - Surprise-based memory storage
    4. Adaptive Conditional Routing - Per-token compute allocation
    5. Weight Sharing - Nx depth at 1x parameter cost
    6. SwiGLU FFN - Best activation function
    7. Multi-Head Latent Attention - DeepSeek-style KV compression
    8. Selective SSM Scan - Mamba-style long-range dependencies
    
    Performance targets:
    - CPU: 5-10x faster than equivalent Transformer
    - GPU: 2-3x faster training, 4x less memory
    - Quality: Matches or exceeds pure Transformer at same params
    """,
    author="NovaLM Team",
    packages=find_packages(include=["NovaLM-ULTRA", "NovaLM-ULTRA.*"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)