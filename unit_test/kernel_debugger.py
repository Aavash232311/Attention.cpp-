
import os
from pathlib import Path
from ml_components.positional_encoding import sinusoidal_positional_encoding


''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''



'''
Reaseases hyperparamaters from C++ in a json file
'''
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
# Output of component that have randomness
class Predictability:

    def __init__(self, d_model, vocab_size, seq_len, batch_size):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size


    def he_init(self, fan_in, fan_out):
        std = (2.0 / fan_in) ** 0.5
        dist = torch.randn(fan_out, fan_in) * std  # For initillization of token embeddings
        return dist


    def token_embeddings(self, fan_in, fan_out):
        p = self.he_init(fan_in, fan_out)
        p.detach().cpu().numpy().tofile("./src/cache/pytorch_out/token_embeddings.bin")
        return p



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


token_embeddings = predictability.token_embeddings(
    vocab_size,
    d_model
)

'''
    Loads C++ with python generated paramaters to avoid randomness
'''
os.system(
    "nvcc -DDEBUG -DDRUN src/attention.cpp src/kernel/math.cu -o src/bin/attention"
)
os.system("./src/bin/attention")

'''
    DEBUG EMBEDDINGS

    HE-INIT is done in python and loaded in C++
    C++ kernels outputs added positional embeddings
'''


class Debugger:

    def __init__(self, d_model, vocab_size, batch_size, seq_len, num_heads ,folder):
        self.folder = folder
        self.d_model = d_model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads

    def readEmbeddings(self, file_name):
        num_elements = batch_size * seq_len * d_model
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(path, size=num_elements, dtype=torch.float32)
        return flat.reshape(batch_size, seq_len, d_model)
    


debugger = Debugger(
    d_model=d_model,
    vocab_size=vocab_size,
    batch_size=batch_size,
    seq_len=seq_len,
    num_heads=num_heads,
    folder='./src/cache/cpp_out'
)


def total_emebddings(te, pe):
    # positional encoding [seq_len, d_model]
    # token embedding [vocab_size, d_model]
    pass


pe = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)

# total embeddings from pytorch
p_embeddings = total_emebddings(token_embeddings, pe)

# total emebddings output from the kernel
k_embeddings = debugger.readEmbeddings('embedding.bin')


