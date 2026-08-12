import torch
from binary_reader.autograd_binary_reader import ReaderFlashAttention


''' This will debug the mathematical operation in flash attention class '''

class DebugFlashAttention(torch.nn.Module):
    def __init__(self, batch_size, seq_len, vocab_size, d_model, num_heads, head_dim):
        super(DebugFlashAttention, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # looks like ideal gas equation, but it's not
        self.P, self.V, self.PT, self.VT = ReaderFlashAttention(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim)

    # dV = P^T G
    # dP = GV^T
    def victor_tango(self):
        print('P:', self.P)


    def debug_pv(self):
        pass