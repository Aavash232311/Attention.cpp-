import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = torch.randn(seq_len, d_model)    
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # [seq_len, 1]
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                          (-torch.log(torch.tensor(10000.0)) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe  # [seq_len, d_model]