'''
    Transformer for sanity check of the cuda kernels.
'''


def total_embeddings(x, token_embedding_table, positional_embedding_table):
    T, B = x.shape

    tok_emb = token_embedding_table[x]               # (T, B, C)
    pos_emb = positional_embedding_table[:T]          # (T, C)

    out = tok_emb + pos_emb.unsqueeze(1)               # (T, B, C)
    return out.transpose(0, 1)                          # (B, T, C)
