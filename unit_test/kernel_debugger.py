import os
from pathlib import Path
from ml_components.transformer import Transformer
from ml_components.positional_encoding import sinusoidal_positional_encoding
from ml_components.grad import Grad
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
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")

import sys
print("Kernel debugger script")

try:
    with open('./src/cache/config.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Critical Error: {e}")
    print("Exiting program.")
    sys.exit(1)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda") 


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
        dist = torch.randn(fan_out, fan_in) * std
        return dist
    
    # releases token embeddings
    def token_embeddings(self, fan_in, fan_out):
        p = self.he_init(fan_in, fan_out)
        p.detach().cpu().numpy().tofile("./src/cache/pytorch_out/token_embeddings.bin")
        return p


d_model = data['d_model']
vocab_size = data['vocab_size']
batch_size = data['batch_size']
seq_len = data['seq_len']
num_heads = data['num_heads']

torch.set_printoptions(precision=4)

model = Transformer(
    batch_size,
    seq_len,
    d_model,
    vocab_size,
    num_heads
)

predictability = Predictability(
    d_model=d_model,
    vocab_size=vocab_size,
    seq_len=seq_len,
    batch_size=batch_size
)


token_embeddings = predictability.token_embeddings(
    d_model,
    vocab_size
)

input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
input_ids.to(torch.int32).numpy().tofile("./src/cache/pytorch_out/input_ids.bin")

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

    def __init__(self, d_model, vocab_size, batch_size, seq_len, num_heads, folder):
        self.folder = folder
        self.d_model = d_model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads

    def readEmbeddings(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(path, size=N, dtype=torch.float32)
        return flat.reshape(self.batch_size, self.seq_len, self.d_model)
    
    def readX(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(path, size=N, dtype=torch.int32) 
        return flat.reshape(self.seq_len, self.batch_size)
    


debugger = Debugger(
    d_model=d_model,
    vocab_size=vocab_size,
    batch_size=batch_size,
    seq_len=seq_len,
    num_heads=num_heads,
    folder='./src/cache/cpp_out'
)

print(f"d_model={d_model} batch_size={batch_size} seq_len={seq_len} vocab_size={vocab_size}")


pe = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)
# same embedding that c++ uses after being released from python.
k_total_emebddings = debugger.readEmbeddings('embedding.bin', batch_size * seq_len * d_model)
x = debugger.readX("x.bin", seq_len * batch_size)

py_total_emebdding = model.total_embeddings(x=x, token_embedding_table=token_embeddings, positional_embedding_table=pe)

print(f"Token embedding kernel status: {torch.allclose(k_total_emebddings, py_total_emebdding) if "Ok" else "Not ok"}")


# DEBUGGING FOR BACKPROPAGATION ITS SOMETING HARD FOR ME TO BACKTRACK
# WE ARE DOING THIS HERE BECAUSE THIS IS LITTLE DIFFICULT FOR MY EYES TO ESTIMATE
# COMPATED TO FORWARD PASS


grad = Grad(d_model=d_model, seq_len=seq_len, batch_size=batch_size, num_heads=num_heads, vocab_size=vocab_size)



