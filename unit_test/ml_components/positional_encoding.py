import math
import torch

def get_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len).unsqueeze(1)  

    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(seq_len, d_model)
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


seq_len = 128
d_model = 512

pe = get_positional_encoding(seq_len, d_model)
print(pe.shape) 