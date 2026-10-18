
import math
import torch
import numpy as np
import torch.nn.functional as F

import os
import sys
import json
from pathlib import Path

import numpy as np


def read_hyperparamaters(path="./src/cache/config.json"):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Critical Error: {e}")
        print("Exiting program.")
        sys.exit(1)

    print("Autograd engine C++ kenrel out")



    d_model = data['d_model']
    vocab_size = data['vocab_size']
    batch_size = data['batch_size']
    seq_len = data['seq_len']
    num_heads = data['num_heads']

    return d_model, vocab_size, batch_size, seq_len, num_heads

def load_tensor(path, shape, dtype=np.float32, offset=0):
    arr = np.fromfile(path, dtype=dtype, count=int(np.prod(shape)), offset=offset)
    return torch.from_numpy(arr.reshape(shape))


def inspect_weights():
    d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparamaters()
    weight = load_tensor("./src/cache/cpp_out/attn_weight_q.bin",
                               shape=(batch_size, batch_size),
                               dtype=np.float32)

    embeddings = load_tensor("./src/cache/cpp_out/attn_embed_q.bin",
                             shape=(batch_size, vocab_size),
                             dtype=np.float32)

    import matplotlib.pyplot as plt

    weight_np = weight.detach().cpu().numpy().flatten()


    plt.figure(figsize=(8, 5))
    plt.plt = plt.hist(weight_np, bins=50, color='skyblue', edgecolor='black', alpha=0.7)

    plt.title('Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    os.chdir(Path.cwd().parent.parent)

    inspect_weights()

    sys.exit()
