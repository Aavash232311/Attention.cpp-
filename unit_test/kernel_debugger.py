
import os
from pathlib import Path


''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''

# Execute and compile with the debugger flag
os.system("nvcc -DDEBUG src/attention.cpp src/kernel/math.cu -o src/bin/attention")
os.system("./src/bin/attention")



import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")

with open("./src/cache/token_embeddings.bin", "rb") as f:
    buf = f.read()

token_embeddings = torch.frombuffer(buf, dtype=torch.float32).reshape(32, 892)

# print(token_embeddings)