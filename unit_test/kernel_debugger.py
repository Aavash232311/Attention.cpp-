import os
from pathlib import Path
from ml_components.grad import Autograd, load_tensor
from ml_components.transformer import Transformer
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
import numpy as np
import torch.nn as nn


from binary_reader.autograd_binary_reader import Reader

warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")

import sys

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

print("Autograd engine C++ kenrel out")

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

print("\n")
print('*' * 60)
print("Forward pass ")
print('*' * 60)

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
        self.vocab_size = vocab_size


    def readEmbeddings(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(path, size=N, dtype=torch.float32)
        return flat.reshape(self.batch_size, self.seq_len, self.d_model)
    
    def readX(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(path, size=N, dtype=torch.int32) 
        return flat.reshape(self.seq_len, self.batch_size)
    

    def read_predictions(self, file_name, np_dtype=np.float32):
        path = os.path.join(self.folder, file_name)
        data = np.fromfile(path, dtype=np_dtype)
        tensor = torch.from_numpy(data)
        return tensor.reshape(self.batch_size, self.seq_len, self.vocab_size)

debugger = Debugger(
    d_model=d_model,
    vocab_size=vocab_size,
    batch_size=batch_size,
    seq_len=seq_len,
    num_heads=num_heads,
    folder='./src/cache/cpp_out'
)

delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w = Reader(
    batch_size=batch_size,
    seq_len=seq_len,
    vocab_size=vocab_size,
    d_model=d_model
)


pe = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)
# same embedding that c++ uses after being released from python.
k_total_emebddings = debugger.readEmbeddings('embedding.bin', batch_size * seq_len * d_model)
x = debugger.readX("x.bin", seq_len * batch_size)

py_total_emebdding = model.total_embeddings(x=x, token_embedding_table=token_embeddings, positional_embedding_table=pe)


# DEBUGGING FOR BACKPROPAGATION ITS SOMETING HARD FOR ME TO BACKTRACK
# WE ARE DOING THIS HERE BECAUSE THIS IS LITTLE DIFFICULT FOR MY EYES TO ESTIMATE
# COMPATED TO FORWARD PASS

print(f"Checking net embedding C++ kernel, status: {torch.allclose(py_total_emebdding, k_total_emebddings)}")


print("\n")
print('*' * 60)
print("Section AUTO GRAD (The Chain Rule of Derivative) ")
print('*' * 60)
# transpose h
tranposing_h = h.transpose(1, 2)
print(f"h^t transpose kernel: {torch.allclose(tranposing_h, h_t)}")
# we need to verify delta h^t
dl_dw_torch = tranposing_h @ delta

# boradcasting for last-dimension
delta_torch = y_predicted - y_actual


# Note-: GIGO sometimes you might just be transposing the garbage who knows,
# if you think that is the case then manually print and see from the C++ script.
print(f"Checking delta across kernels {torch.allclose(delta_torch, delta)}")
print(f"Checking wt transpose kernel: {torch.allclose(w.T, wt)}")
