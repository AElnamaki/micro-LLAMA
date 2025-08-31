# micro-LLAMA
A Single-GPU-Compatible, Educational Implementation of LLaMA 3.1

## 1. Introduction
**micro-LLAMA** is a minimal, didactic reimplementation of Meta AI’s LLaMA 3.1 architecture, designed to run on a single GPU. It replaces distributed tensor parallelism primitives (e.g., `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`) with their dense equivalents (`nn.Linear`, `nn.Embedding`) for simplicity and accessibility.

This project is tailored for:
- **Educators** teaching transformer architectures.
- **Researchers** experimenting with LLaMA-like models without multi-node setups.
- **Practitioners** seeking a readable, hackable codebase for profiling and modification.

**Note**: This is a **work-in-progress (WIP)** project, actively developed and not yet ready for production use. The current focus is on the LLaMA 3.1 8B base model.

## 2. Educational Motivation
Meta AI’s official LLaMA codebase is optimized for high-throughput, multi-GPU environments, but this comes with:
- Heavy dependencies on distributed training (`torch.distributed`, FSDP/TP).
- Complex tensor-parallel layers that obscure the core transformer logic.
- Checkpoint loading optimized for multi-GPU sharding.

In contrast, **micro-LLAMA**:
- Runs end-to-end on a single GPU (≥16 GB VRAM recommended for the 8B model).
- Uses plain PyTorch modules for traceable, transparent computation.
- Prioritizes code readability over maximum performance.

This repository is to LLaMA 3.1 what **nanoGPT** is to GPT-2: a minimal, dependency-light implementation that supports training, fine-tuning, and inference with a simple codebase.

## 3. Architectural Overview
### 3.1 Core Changes from LLaMA 3.1
| Original Layer            | micro-LLAMA Equivalent |
|---------------------------|------------------------|
| `ColumnParallelLinear`    | `torch.nn.Linear`      |
| `RowParallelLinear`       | `torch.nn.Linear`      |
| `VocabParallelEmbedding`  | `torch.nn.Embedding`   |

**Implications**:
- No model-parallel sharding; all weights reside on a single device.
- Simplified debugging at the cost of a higher memory footprint per GPU.
- No collective communication; fully local compute graph.

### 3.2 Transformer Stack
- **Embedding Layer**: Standard learned token embeddings.
- **Rotary Positional Encoding (RoPE)**: Scalable frequency adjustment (`apply_scaling`) for extended context windows.
- **Multi-Head Attention**:
  - Optional Flash Attention (`--flash`) for reduced memory bandwidth.
  - KV caching for efficient autoregressive decoding.
- **Feed-Forward Network (FFN)**: SwiGLU activation, with size scaling by `multiple_of` for hardware-friendly alignment.
- **RMSNorm**: Applied before attention and FFN layers, following modern transformer best practices.
- **Final Projection**: Dense output mapping to vocabulary logits.

## 4. RoPE Scaling and Long Context Handling
LLaMA 3.1 supports up to 8,192 tokens via RoPE. micro-LLAMA optionally rescales RoPE frequencies to extend context length without retraining:
- Low-frequency components remain unchanged to preserve global coherence.
- High-frequency components are compressed to avoid over-rotation.

## 5. KV Cache Implementation
For efficient autoregressive inference, micro-LLAMA caches Key and Value tensors per attention layer:
- **Without cache**: Each decoding step recomputes attention over the entire sequence (O(n²) scaling).
- **With cache**: Only new tokens are processed, reusing previous states (O(n) scaling per step).

## 6. Setup and Usage
### 6.1 Reference Implementation
The official LLaMA 3.1 code from Meta AI serves as the reference. However, the official repository lacks clear documentation for model usage. Below are the steps to set up and run the reference implementation for comparison.

#### Step 1: Clone the Official Repository
```
git clone https://github.com/meta-llama/llama-models.git
```

#### Step 2: Download the LLaMA 3.1 8B Model
Request access to LLaMA 3.1 at llama.meta.com/llama-downloads. Select meta-llama-3.1-8b (base model). Then:
```
cd llama-models/models/llama3_1
chmod u+x download.sh
./download.sh
```
Enter the URL provided in the email from Meta. This downloads ~16 GB of data (8B parameters in bfloat16) to `./Meta-Llama-3.1-8B`.

#### Step 3: Set Up the Environment
Create and activate a Conda environment:
```
conda create -n llama31 python=3.10
conda activate llama31
```
Note: Avoid Python versions ≥3.12 due to incomplete PyTorch support.

#### Step 4: Install Dependencies
From the `llama-models` directory:
```
pip install -r requirements.txt
pip install -e .
```

#### Step 5: Run the Reference Script
From the root of your project directory:
```
pip install fire
torchrun --nnodes 1 --nproc_per_node 1 reference.py \
    --ckpt_dir llama-models/models/llama3_1/Meta-Llama-3.1-8B \
    --tokenizer_path llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model


**Note**: The reference script is adapted from Meta’s `example_text_completion.py`, which is marked as deprecated for LLaMA 3.0. Despite this, it produces valid completions for LLaMA 3.1, such as:
Clearly, the meaning of life is to be found in the joy of living, in the joy of love, in the joy of work...

Simply put, the theory of relativity states that the laws of physics are the same for all non-accelerating observers...

Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
```
*Fix*: The original Meta script had a trailing whitespace bug in prompts, which affected tokenization. This is fixed in micro-LLAMA’s implementation.

### 6.2 Running micro-LLAMA
The micro-LLAMA implementation is in `llama31.py` (~700 lines) and tested via `test_llama31.py`, which reproduces the reference output for validation.

Run the Test Script:
```
python test_llama31.py
```
This runs without `torchrun` or distributed dependencies, producing identical outputs to the reference, confirming correctness.

### 6.3 Fine-Tuning
A preliminary fine-tuning script is included, using the Tiny Stories dataset. It currently requires significant VRAM (e.g., training only RMSNorm layers consumes a large portion of an 80 GB GPU). Run fine-tuning via the main entry point in `llama31.py`.

## 7. Status and Limitations
- **WIP**: The project is under active development and not yet production-ready.
- **Single-GPU Focus**: Optimized for simplicity, not performance. Multi-GPU setups are not supported.
- **Memory Requirements**: The 8B model requires ≥16 GB VRAM for inference and significantly more for fine-tuning.
- **Dependencies**: Minimal, relying on PyTorch and lightweight libraries (e.g., fire for CLI).

## 8. Next Steps
- Improve fine-tuning efficiency to reduce VRAM usage.
- Expand documentation and add tutorials for educational use.
- Support additional LLaMA 3.1 model sizes (e.g., 70B, if feasible on single GPUs).
- Enhance long-context handling with optimized RoPE scaling.

## 9. Contributing
Contributions are welcome! Please submit issues or pull requests to the repository. Focus areas include:
- Bug fixes for inference or fine-tuning.
- Optimizations for memory usage.
- Educational examples or documentation.

## 10. License
This project is licensed under the same terms as the original LLaMA 3.1 model. Ensure you have access to LLaMA 3.1 weights via Meta AI’s official process before using this code.

