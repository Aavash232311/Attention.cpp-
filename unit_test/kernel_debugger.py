import os
import torch
from pathlib import Path

from debug.debug_autograd import debug_autograd
from debug.debug_embeddings import verify_embeddings
from binary_reader.hyperparamaters import read_hyperparamaters

''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''
# Make sure we are able to read C++ project directory from here
os.chdir(Path.cwd().parent)
# Read release hyperparameters, compiles the C++ and releases the hyperparamaters.
d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparamaters()

print("\n")
print('*' * 60)
print("Forward pass ")
print('*' * 60)

# check if embedding component of the transformer is okay.
verify_embeddings(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads)

torch.set_printoptions(precision=4)

input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
input_ids.to(torch.int32).numpy().tofile("./src/cache/pytorch_out/input_ids.bin")


debug_autograd(d_model=d_model, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, num_heads=num_heads)


