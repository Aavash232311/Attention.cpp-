
import os
from pathlib import Path


''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''

# First C++ execution to retrive the hyperparamaters with -PARAMS flag
os.system(
    "nvcc -DDEBUG -DPARAMS src/attention.cpp src/kernel/math.cu -o src/bin/attention"
)
os.system("./src/bin/attention")


import json
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")

import sys

try:
    with open('./src/cache/config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Critical Error: {e}")
    print("Exiting program.")
    sys.exit(1)  

# with open("./src/cache/token_embeddings.bin", "rb") as f:
#     buf = f.read()


'''
Randomness lies in the init of weight and bias.
So we are going to generate that in python and load in C++
and then run the debugger.

'''

class Predictability:

    def __init__(self, d_model, vocab_size, seq_len, batch_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size


    def he_init(self, fan_in, fan_out):
        std = (2.0 / fan_in) ** 0.5
        return torch.randn(fan_out, fan_in) * std  # For initillization of token embeddings


    def token_embeddings(self, fan_in, fan_out):
        p = self.he_init(fan_in, fan_out)
        p.detach().cpu().numpy().tofile("./src/cache/token_embeddings.bin")

d_model = data['d_model']
vocab_size = data['vocab_size']
batch_size = data['batch_size']
seq_len = data['seq_len']
num_heads = data['num_heads']

predictability = Predictability(
    d_model=d_model,
    vocab_size=vocab_size,
    seq_len=seq_len,
    batch_size=batch_size
)


predictability.token_embeddings(
    vocab_size,
    d_model
)