import torch






''' This will debug the mathematical operation in flash attention class '''

class DebugFlashAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, seq_len, vocab_size):
        super(DebugFlashAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size


    # dV = P^T G
    # dP = GV^T
    def victor_tango(self):
        pass

    def debug_pv(self):
        pass