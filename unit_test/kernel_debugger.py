
import os
from pathlib import Path


''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''

# Execute and compile with the debugger flag
os.system("nvcc -DDEBUG src/attention.cpp src/kernel/math.cu -o src/bin/attention")
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

with open("./src/cache/token_embeddings.bin", "rb") as f:
    buf = f.read()

token_embeddings = torch.frombuffer(buf, dtype=torch.float32).reshape(data['d_model'], data['vocab_size'])


print(token_embeddings)
