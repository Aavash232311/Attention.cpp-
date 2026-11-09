# Building

Requires CMake 3.24 or newer, the CUDA toolkit (nvcc), and a compatible host C++ compiler.

From the repository root, configure once and build in parallel:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
./src/bin/attention
```

After source edits, repeat only the build and run commands. CMake recompiles
affected source files and their header dependents; unchanged CUDA objects are
reused. Increase `--parallel 4` if you have CPU and RAM available. Header changes
can still rebuild several objects until the implementations move into source files.

The `run` target builds and runs from the correct working directory:

```bash
cmake --build build --target run --parallel 4
```

Release optimization is enabled by default. CUDA fast math and device-link-time
optimization are not enabled. The CUDA compiler's default architecture is used
unless you specify one, for example `-DCMAKE_CUDA_ARCHITECTURES=86` for compute
capability 8.6. On a machine with a detectable GPU you can configure with
`-DCMAKE_CUDA_ARCHITECTURES=native` to target that machine's GPU.

If Ninja is installed, use `-G Ninja` on the first configure with a fresh build
directory. The build commands remain the same.

Existing compile-time switches are available as CMake options:

```bash
cmake -S . -B build -DATTENTION_DEBUG_DUMPS=ON
cmake -S . -B build -DATTENTION_DEBUG_DUMPS=OFF
```

`ATTENTION_DEBUG_DUMPS` defines `DEBUG`, which also changes the existing main
program to run only one epoch. `ATTENTION_ENGINE_RUN=ON` defines `ENGINE_RUN`.

GitHub Actions builds this same CMake target using Ninja and parallel jobs,
with debug dumps enabled to match the previous CI compile. It caches compiler
outputs using ccache across runs. The first run populates the cache; later runs
can reuse unchanged compilations. CI compiles only and does not run GPU training.
