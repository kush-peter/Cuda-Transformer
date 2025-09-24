# Transformer in CUDA 🚀

A from-scratch implementation of the Transformer architecture (**Attention Is All You Need**) in **C++ + CUDA** - no PyTorch/TensorFlow, just raw GPU kernels.

---

## What this repo contains
- `src/transformer_attention.cu` - core implementation of full Transformer forward pass: Multi-Head Attention, Feed-Forward, LayerNorm, positional encodings.
- `src/perf_benchmarks.cu` - CPU vs GPU matrix multiplication of 1024x1024 size matrices
- `README.md` - this file.

---

## Quick features
- Full **Transformer forward pass** implemented in CUDA.
- **Multi-Head Self-Attention**, Feed-Forward layers, and basic LayerNorm.
- **GPU vs CPU benchmarks** for large matrix multiplications (1024×1024 example included).
- Modular CUDA backend: kernel launch wrappers, memory helpers.
- Clean and easy to extend for training or inference hooks.

--- 

## Next Steps / Future Work:
- **Optimize kernels** using shared memory tiling and loop unrolling.
- Extend to **full multi-layer Transformer** with batched inputs.
- Add training and backprop hooks for end-to-end learning.
- Integrate **positional encoding and attention masking** for NLP datasets.

---

## Compile & Run (example)
**Requires:** NVIDIA GPU, CUDA toolkit (nvcc)
**Compile & Run**
```bash
cd Cuda-Transformer
## Build Transformer 
nvcc -o transformer src/transformer_attention.cu
./transformer_attention

# Build and run benchmarks
nvcc -o perf_benchmarks src/perf_benchmarks.cu
./perf_benchmarks

```
## Sample Output for perf_benchmarks
```bash
Generated random 1024x1024 matrices A and B.
CPU MatMul Time: 12073.1 ms
GPU MatMul Time: 373.833 ms
Speedup: 32.2954x
C_cpu[0][0]: 262.589, C_gpu[0][0]: 262.589