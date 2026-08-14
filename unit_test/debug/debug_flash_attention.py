import torch
from debug.static import RESET, RED, GREEN
from binary_reader.autograd_binary_reader import ReaderFlashAttention
''' This will debug the mathematical operation in flash attention class '''

class DebugFlashAttention(torch.nn.Module):
    def __init__(self, batch_size, seq_len, vocab_size, d_model, num_heads, head_dim, dl_dw: torch.Tensor):
        super(DebugFlashAttention, self).__init__()
        self.batch_size = batch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.dl_dw = dl_dw # Upstream gradient G

        # looks like ideal gas equation, but it's not
        self.P, self.V, self.PT, self.VT, self.G_unc = ReaderFlashAttention(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim)

    # dV = P^T G
    # dP = GV^T
    def victor_tango(self):
        torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'))
        check_transpose_p = torch.allclose(self.P.transpose(2, 3), self.PT)
        check_transpose_v = torch.allclose(self.V.transpose(2, 3), self.VT)

        if not check_transpose_p:
            print(f"Checking PT C++ kernel, status:{RED} {check_transpose_p} {RESET}")
        else:
            print(f"Checking PT C++ kernel, status:{GREEN} {check_transpose_p} {RESET}")

        if not check_transpose_v:
            print(f"Checking VT C++ kernel, status:{RED} {check_transpose_v} {RESET}")
        else:
            print(f"Checking VT C++ kernel, status:{GREEN} {check_transpose_v} {RESET}")

    def debug_pv(self):
        pass