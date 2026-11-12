# A Transformer from scratch in C++ and CUDA

The goal is to understand the attention mechanism in depth without a lot of abstraction. I may not remember much after a long time, but this project should give me an idea of how everything works inside a Transformer model. I am trying to make it as simple as possible, even though managing all these buffers gets complicated pretty quickly.

I am studying how to build it. This is a learning project.

## What it does

This is a character-level next-token model. It reads text from `src/data/chunk.txt`, encodes characters as integers, and learns to predict the next character. Generation appends a predicted character and uses the latest context to predict another one.

The C++/CUDA implementation includes token embeddings, positional encoding, multi-head causal attention, LayerNorm, residual connections, an LM head, cross-entropy loss, backward passes, and AdamW. The point is to follow the actual operations and gradients rather than hide them behind a framework.

## Folder structure

```text
.
├── CMakeLists.txt           # C++ and CUDA build
├── BUILD.md                # More build options
├── src/
│   ├── main.cpp            # Model settings and entry point
│   ├── include/            # Shared helpers, data loader and parameter structs
│   ├── forward/            # Forward pass and its CUDA kernels
│   ├── backpropagation/    # Backward pass and its CUDA kernels
│   ├── optimizer/          # AdamW implementation
│   ├── kernel/             # Shared CUDA operations
│   ├── data/               # Training text
│   ├── cache/              # Tensor dumps and saved debug configuration
│   ├── bin/                # Compiled executable
│   └── unit_note/          # Derivations and working notes
└── debugger/
    ├── kernel_debugger.py  # Build, run and compare the dumps
    ├── debug/              # PyTorch checks for forward and backward operations
    ├── binary_reader/      # Load C++ dumps and model dimensions
    ├── ml_components/      # Reference operations and tensor loading
    ├── rnd/                # Embedding initialization helpers
    └── Analysis/           # Weight inspection and plotting
```

## Derivations and notes

These are my working notes in [src/unit_note](src/unit_note). I wrote them while trying to understand the math.

- [Attention backward derivation](src/unit_note/flashback.md) · [PDF](src/unit_note/flashback.pdf)
- [Softmax and cross-entropy gradient derivation](src/unit_note/interfaceback.md) · [PDF](src/unit_note/interfaceback.pdf)
- [Softmax and parallel reduction notes](src/unit_note/softmax_parallel_reduction.org)
- [Softmax backward kernel example](src/unit_note/backgrad_softmax_kernel_example.org)
- [Early embedding notes](src/unit_note/note.org) — marked outdated

## Current parameters

These settings are in `src/main.cpp`.

| Parameter | Value | Meaning |
| --- | --- | --- |
| `d_model` | 128 | Channels per token |
| `num_heads` | 16 | Attention heads |
| `head_dim` | 8 | `d_model / num_heads` |
| `batch_size` | 16 | Sequences per batch |
| `seq_len` | 16 | Characters per training context |
| `vocab_size` | From the corpus | Number of distinct encoded characters |
| `epoch` | 2250 | Configured training epochs; DEBUG uses 1 |
| `drop_last` | `true` | Drop an incomplete batch |

Currently, `main.cpp` calls training only when DEBUG is enabled. The generation call is commented out. For normal training, call `model->train(epoch)` outside the debug condition; enable the prompt and `model->forward(prompt, charPool)` call to generate text afterward.

## Build and run

You need CMake 3.24+, the CUDA toolkit (`nvcc`), a compatible C++ compiler, and an NVIDIA GPU to run the model. From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./src/bin/attention
```

After editing code, repeat the build and run commands. CMake reuses unchanged object files, so it does not compile every CUDA file each time. See [BUILD.md](BUILD.md) for more options.

## Python debugger

The debugger uses PyTorch and NumPy as references for the custom CUDA operations. With those installed in your Python environment, run:

```bash
python3 debugger/kernel_debugger.py
```

It builds the same CMake target in `build-debugger` with DEBUG enabled, then runs one training epoch. C++ writes the model dimensions to `src/cache/config.json` and tensor dumps to `src/cache/cpp_out`. Python reads those tensors, reconstructs their shapes, and compares the CUDA results with the corresponding PyTorch operations.

Checks print colored `[PASS]` or `[FAIL]` results for operations such as head reshaping, attention matrix multiplication, masking, softmax, LayerNorm, and backward gradients. Separate embedding and AdamW checks are also present, but their calls in the runner are currently commented out.

These are checks of specific operations using the dumped tensors. Passing them helps, but it does not prove that the whole training and generation pipeline is correct. The dimensions and dumps need to come from the same run.
