import torch
import torch.nn as nn


'''
    Transformer for sanity check of the cuda kernels.
'''


def total_embeddings(x, token_embedding_table, positional_embedding_table):
    T, B = x.shape

    tok_emb = token_embedding_table[x]               # (T, B, C)
    pos_emb = positional_embedding_table[:T]          # (T, C)

    out = tok_emb + pos_emb.unsqueeze(1)               # (T, B, C)
    return out.transpose(0, 1)                          # (B, T, C)


class Transformer(nn.Module):

    def __init__(self, batch_size, seq_len, d_model, vocab_size, num_heads):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")


