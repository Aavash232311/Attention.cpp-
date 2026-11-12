import os
import subprocess
import torch
from pathlib import Path


from rnd.release_embeddings import release_token_embeddings
from debug.debug_autograd import debug_autograd
from debug.debug_embeddings import verify_embeddings
from debug.debug_flash_attention import DebugFlashAttention
from debug.debug_optimizer import DebugOptimizer

from binary_reader.hyperparameters import read_hyperparameters

from debug.quick_binary_check_forward import quick_attention_check

''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification.
    Never trust the debugger too much, this is for basic mathematical verification (for specific operations) only.
    Use local print and this combined :)

    THIS IS JUST MY ROUGH WORK NOTHING ELSE A SMALL SANITY CHECK

    Wont notice things like infinitely large values in a kernel sometimes,
    just focuses on mathematical operations.
 '''

# Make sure we are able to read C++ project directory from here
os.chdir(Path(__file__).resolve().parents[1])


# After hyperparameters are loaded, we need to load those binary that the C++ releases
# CMake caches each object file and rebuilds only changed sources in parallel.
subprocess.run([
    "cmake", "-S", ".", "-B", "build-debugger",
    "-DCMAKE_BUILD_TYPE=Release", "-DATTENTION_DEBUG_DUMPS=ON",
    "-DATTENTION_ENGINE_RUN=OFF",
], check=True)
subprocess.run([
    "cmake", "--build", "build-debugger", "--target", "attention",
    "--parallel", str(os.cpu_count() or 1),
], check=True)
subprocess.run(["./src/bin/attention"], check=True) # this releases the forward pass binary as well


print("\n")
print('*' * 60)
print("Forward pass ")
print('*' * 60)
quick_attention_check()

# Read release hyperparameters, compiles the C++ and releases the hyperparameters.
d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparameters()

print("\n")
print('*' * 60)

# token_emb = release_token_embeddings(d_model=d_model, seq_len=seq_len, vocab_size=vocab_size, batch_size=batch_size)
#
#
# # check if embedding component of the transformer is okay.
# verify_embeddings(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads, token_embeddings=token_emb)

torch.set_printoptions(precision=4)

input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
input_ids.to(torch.int32).numpy().tofile("./src/cache/pytorch_out/input_ids.bin")


G = debug_autograd(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads)


flash_attention_debugger = DebugFlashAttention(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads, head_dim=d_model // num_heads, dl_dw=G)
flash_attention_debugger.verify_backward()

# why would you do this in weird way? well this is a scratch work.
# if you pay me to write code it won't be this messy
# optimizer_debugger = DebugOptimizer(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads, head_dim=d_model // num_heads, dl_dw=G)
# optimizer_debugger.verify_optimizer()

