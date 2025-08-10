# micro-LLAMA
single-GPU-compatible implementation

# Educational Minimal Implementation

## 1. Introduction

`MICRO-LLaMA 3.1` is a **didactic, single-GPU-compatible reimplementation** of Meta’s LLaMA 3.1 architecture.  
It removes distributed tensor parallelism primitives such as `ColumnParallelLinear`, `RowParallelLinear`, and `VocabParallelEmbedding` — replacing them with their dense equivalents (`nn.Linear` and `nn.Embedding`).  

This adaptation is aimed at **educators, researchers, and practitioners** who want to:
- Understand the transformer architecture in depth.
- Experiment with LLaMA-like models **without multi-node orchestration**.
- Profile and modify model internals in a **readable, hackable codebase**.

---

## 2. Educational Motivation

While Meta’s original LLaMA codebase is highly optimized for **throughput and memory scaling**, it comes with:
- **Heavy distributed training dependencies** (`torch.distributed`, FSDP/TP).
- **Complex tensor parallel layers** that obscure the core algorithmic flow.
- Checkpoint loading mechanisms optimized for multi-GPU sharding.

In contrast, `nano-LLaMA 3.1`:
- Runs end-to-end **on a single GPU** (≥ 16 GB VRAM recommended for the 8B model).
- Uses **plain PyTorch modules** so each computation step can be traced without indirection.
- Focuses on **readability over maximum performance**.

---

## 3. Architectural Overview

### 3.1 Core Changes from LLaMA 3.1
| Original Layer                         | nano-LLaMA Equivalent |
|----------------------------------------|------------------------|
| `ColumnParallelLinear`                 | `torch.nn.Linear`      |
| `RowParallelLinear`                    | `torch.nn.Linear`      |
| `VocabParallelEmbedding`               | `torch.nn.Embedding`   |

**Implications**:
- No model-parallel sharding — **entire weight matrices reside on a single device**.
- Simpler debugging, at the cost of **higher memory footprint** per GPU.
- No collective communication ops; fully local compute graph.

---

### 3.2 Transformer Stack
- **Embedding Layer**: Standard learned token embeddings.
- **Rotary Positional Encoding (RoPE)**: Implemented with **scalable frequency adjustment** (`apply_scaling`) to support longer context windows.
- **Multi-Head Attention**:
  - Flash Attention optional (`--flash`) for reduced memory bandwidth usage.
  - **KV caching** implemented for autoregressive decoding efficiency.
- **Feed-Forward Network (FFN)**: SwiGLU activation, size scaling by `multiple_of` for hardware-friendly alignment.
- **RMSNorm**: Normalization before attention/FFN, matching modern transformer best practices.
- **Final Projection**: Dense output mapping to vocabulary logits.

---

## 4. Rope Scaling and Long Context Handling

The original LLaMA 3.1 supports **up to 8,192 tokens** via RoPE.  
This implementation optionally rescales RoPE frequencies to extend context without retraining, following [long context scaling strategies](https://arxiv.org/abs/2306.15595).

- **Low-frequency components** remain unchanged (preserve global coherence).
- **High-frequency components** are compressed (avoid over-rotation).

---

## 5. KV Cache Implementation

Autoregressive inference efficiency is achieved by caching the computed **Key** and **Value** tensors per attention layer:

- **Without cache**: Each decoding step recomputes attention over the entire sequence → O(n²) scaling.
- **With cache**: Only new tokens are processed, previous states are reused → O(n) scaling per step.

Implemented in `KVCache`:
```python
self.register_buffer("cache_k", torch.zeros(...))
self.register_buffer("cache_v", torch.zeros(...))
