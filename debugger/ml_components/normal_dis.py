import torch

'''
uncertainty lies in the init of weight and bias.
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
