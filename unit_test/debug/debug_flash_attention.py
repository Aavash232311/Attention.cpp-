import math
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
        self.P, self.V, self.PT, self.VT, self.G_unc, self.G, self.dp, self.dv, self.softmax_upstream, self.dQ, self.K, self.Q, self.d_score_t, self.dK = ReaderFlashAttention(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim)

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

        # check the un-contact logic here for upstream gradient G
        dis_g_torch = self.G.view(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim
        )

        # G_unc = Shape (B, n_head, seq_len, head_dim)
        g_unc_t = self.G_unc.transpose(1, 2)
        check_un_contact_G = torch.allclose(
            dis_g_torch,
            g_unc_t, # just to check, confusion glued together here.
            atol=1e-4,
            rtol=1e-4
        )

        # check for dp and dv, backmost layer of flash attention logic
        dv_torch = self.PT @ self.G_unc
        dp_torch = self.G_unc @ self.VT

        check_dv = torch.allclose(dv_torch, self.dv, atol=1e-4, rtol=1e-4)
        check_dp = torch.allclose(dp_torch, self.dp, atol=1e-4, rtol=1e-4)

        # check softmax back pass
        '''
            First component of the rwo_sum 
            (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, seq_len, head_dim)
            = (batch_size, num_heads, seq_len, head_dim) 
            
            p and dp in kernel and torch matches not sure why it wont but I checked it again.
            Kernel looks healthy to me, every row cannot be zero because one row of P sums to 1.
            There can be combination of numbers that sums to one but not sure how all rows can be zero.
        '''
        row_sum = (self.P * self.dp).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        d_score_torch = self.P * (self.dp - row_sum)


        check_softmax_Grad = torch.allclose(d_score_torch, self.softmax_upstream, atol=1e-4, rtol=1e-4)

        # Now for the dQ and dK tensor for S = QK^T part
        scaling_factor = 1.0 / math.sqrt(self.head_dim)
        # now the new upstream gradient will be d_score_torch

        dQ = scaling_factor * (d_score_torch @ self.K)
        check_dq = torch.allclose(dQ, self.dQ, atol=1e-4, rtol=1e-4)

        check_d_score_t = torch.allclose(d_score_torch.transpose(2, 3), self.d_score_t, atol=1e-4, rtol=1e-4)


        if not check_dp:
            print(f"Checking dp kernel, status:{RED} {check_dp} {RESET}")
        else:
            print(f"Checking dp kernel, status:{GREEN} {check_dp} {RESET}")

        if not check_dv:
            print(f"Checking dv kernel, status:{RED} {check_dv} {RESET}")
        else:
            print(f"Checking dv kernel, status:{GREEN} {check_dv} {RESET}")

        if not check_un_contact_G:
            print(f"Checking contact/uncontact dl_dh kernel, status:{RED} {check_un_contact_G} {RESET}")
        else:
            print(f"Checking contact/uncontact dl_dh kernel, status:{GREEN} {check_un_contact_G} {RESET}")

        if not check_softmax_Grad:
            print(f"Checking softmax back pass kernel, status:{RED} {check_softmax_Grad} {RESET}")
        else:
            print(f"Checking softmax back pass kernel, status:{GREEN} {check_softmax_Grad} {RESET}")

        if not check_dq:
            print(f"Checking dQ kernel, status:{RED} {check_dq} {RESET}")
        else:
            print(f"Checking dQ kernel, status:{GREEN} {check_dq} {RESET}")

        if not check_d_score_t:
            print(f"Checking d score transpose kernel, status:{RED} {check_d_score_t} {RESET}")
        else:
            print(f"Checking d score transpose kernel, status:{GREEN} {check_d_score_t} {RESET}")

    def debug_pv(self):
        pass