import os
import torch
from pathlib import Path

from ml_components.transformer import Transformer
from ml_components.normal_dis import  Predictability
from ml_components.positional_encoding import sinusoidal_positional_encoding
from binary_reader.embedding_reader import Debugger
from binary_reader.autograd_binary_reader import Reader
from binary_reader.hyperparamaters import read_hyperparamaters

''' Automated debugging script for CUDA kernel in attention.cpp
    File based dump verification
 '''
os.chdir(Path.cwd().parent)
# Read release hyperparameters, compiles the C++ and releases the hyperparamaters.
d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparamaters()

print("\n")
print('*' * 60)
print("Forward pass ")
print('*' * 60)


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
torch.set_printoptions(precision=4)

input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
input_ids.to(torch.int32).numpy().tofile("./src/cache/pytorch_out/input_ids.bin")

# Loads C++ with python generated parameters to avoid randomness
os.system("nvcc -DDEBUG -DDRUN src/main.cpp src/kernel/utils.cu src/forward/Kernel/layer_norm.cu src/forward/Kernel/embedding.cu src/forward/Kernel/linear.cu src/forward/Kernel/attention_head.cu src/forward/Kernel/interface.cu src/backpropagation/Kernel/interface_back.cu -o src/bin/attention")
os.system("./src/bin/attention")


debugger = Debugger(
    C=d_model,
    V=vocab_size,
    B=batch_size,
    T=seq_len,
    num_head=num_heads,
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
k_total_embeddings = debugger.readEmbeddings('embedding.bin', batch_size * seq_len * d_model)
x = debugger.readX("x.bin", seq_len * batch_size)

py_total_embedding = model.total_embeddings(x=x, token_embedding_table=token_embeddings, positional_embedding_table=pe)


# DEBUGGING FOR BACKPROPAGATION ITS SOMETHING HARD FOR ME TO BACKTRACK
# WE ARE DOING THIS HERE BECAUSE THIS IS LITTLE DIFFICULT FOR MY EYES TO ESTIMATE
# COMPATED TO FORWARD PASS

print(f"Checking net embedding C++ kernel, status: {torch.allclose(py_total_embedding, k_total_embeddings)}")


print("\n")
print('*' * 60)
print("Section AUTO GRAD (The Chain Rule of Derivative) ")
print('*' * 60)
# transpose h
transposing_h = h.transpose(1, 2)
print(f"h^t transpose kernel: {torch.allclose(transposing_h, h_t)}")
# we need to verify delta h^t
dl_dw_torch = transposing_h @ delta

# broadcasting for last-dimension
delta_torch = y_predicted - y_actual


# Note-: GIGO sometimes you might just be transposing the garbage who knows
# if you think that is the case then manually print and see from the C++ script.
print(f"Checking delta across kernels {torch.allclose(delta_torch, delta)}")
print(f"Checking wt transpose kernel: {torch.allclose(w.T, wt)}")
