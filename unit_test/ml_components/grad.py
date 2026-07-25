import os
import numpy as np
import torch

def load_tensor(path, shape, dtype=np.float32, offset=0):
    arr = np.fromfile(path, dtype=dtype, count=int(np.prod(shape)), offset=offset)
    return torch.from_numpy(arr.reshape(shape))

class Autograd:
    

    def __init__(self, d_model, vocab_size, batch_size, seq_len, num_heads, folder):
        self.folder = folder
        self.d_model = d_model
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.vocab_size = vocab_size





    def propagate_lmhead(self):
        pass