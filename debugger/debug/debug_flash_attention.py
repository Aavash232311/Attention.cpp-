import math

import torch
from debug.static import print_check
from binary_reader.autograd_binary_reader import read_attention_backward_tensors
from debug.complex_kernel_operations.equivalent_kernel_functions import layer_backward_analytical, layer_norm_beta_gamma_analytical, compute_token_embedding_grad

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
        (self.P, self.V, self.PT,
         self.VT, self.G_unc, self.dl_dh,
         self.dp, self.dv, self.softmax_upstream,
         self.dQ, self.K, self.Q, self.d_score_t,
         self.dK, self.wqt, self.wkt, self.wvt,
         self.wq, self.wk, self.wv, self.upq,
         self.upk, self.upv, self.G_x_hat,
         self.layer_norm_gamma, self.mc, self.stdc,
         self.x, self.layer_norm_back_x,
         self.d_gamma, self.d_beta, self.weight_contact,
         self.weight_contact_transpose, self.G,
         self.d_bias_output_proj, self.final_gradient_from_add_residual,
         self.d_embeddings, self.token_ids,
         self.d_weight_q, self.d_weight_k, self.d_weight_v,
         self.d_bias_q, self.d_bias_k, self.d_bias_v,
         self.bias_q, self.bias_k, self.bias_v, self.d_weight_output_project,
         self.output_projection_xt, self.output_projection_x, self.normalized_x,
         self.normalized_xt) = read_attention_backward_tensors(batch_size, seq_len, vocab_size, d_model, num_heads, head_dim)

    # dV = P^T G
    # dP = GV^T
    def verify_backward(self):
        torch.set_printoptions(precision=3, sci_mode=False, threshold=float('inf'))
        check_transpose_p = torch.allclose(self.P.transpose(2, 3), self.PT)
        check_transpose_v = torch.allclose(self.V.transpose(2, 3), self.VT)

        print_check('Attention probability transpose', check_transpose_p)

        print_check('Value transpose', check_transpose_v)

        # check the un-contact logic here for upstream gradient G
        dis_g_torch = self.G.view(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim
        )

        # G_unc = Shape (B, n_head, seq_len, head_dim)
        g_unc_t = self.G_unc.transpose(1, 2)
        check_split_head_gradient = torch.allclose(
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

        dk_torch = scaling_factor * (self.d_score_t @ self.Q)

        check_dk = torch.allclose(dk_torch, self.dK, atol=1e-4, rtol=1e-4)

        # I will refactor this later, or let a young college student take on the job.
        # Hopefully, we will learn something from the process.
        # I know this approach is repetitive, but it works and helps me identify
        # kernel-related problems. This is just a small sanity check.


        print_check('Attention probability gradient', check_dp)

        print_check('Value gradient', check_dv)

        print_check('Split attention output gradient into heads', check_split_head_gradient)

        print_check('Attention softmax backward', check_softmax_Grad)

        print_check('Query gradient', check_dq)

        print_check('Attention score gradient transpose', check_d_score_t)

        print_check('Key gradient', check_dk)

        check_wqt = torch.allclose(self.wq.T, self.wqt)
        print_check('Query weight transpose', check_wqt)

        check_wkt = torch.allclose(self.wk.T, self.wkt)
        print_check('Key weight transpose', check_wkt)

        check_wvt = torch.allclose(self.wv.T, self.wvt)
        print_check('Value weight transpose', check_wvt)

        dQ_compact = self.dQ.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.d_model)
        check_dQ = torch.allclose(dQ_compact, self.upq)
        print_check('Query gradient head merge', check_dQ)

        dK_compact = self.dK.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.d_model)
        check_dK = torch.allclose(dK_compact, self.upk)
        print_check('Key gradient head merge', check_dK)

        dV_compact = self.dv.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.d_model)
        check_dV = torch.allclose(dV_compact, self.upv)
        print_check('Value gradient head merge', check_dV)

        # Now lets check the contact of these qkv weights

        dQ_Wq = self.upq @ self.wq.T
        dK_Wk = self.upk @ self.wk.T
        dV_Wv = self.upv @ self.wv.T

        G_hat_net = dQ_Wq + dK_Wk + dV_Wv
        check_G_x_hat = torch.allclose(G_hat_net, self.G_x_hat,atol=1e-4, rtol=1e-4)
        print_check('LayerNorm upstream gradient', check_G_x_hat)

        back_x = layer_backward_analytical(
            self.x,
            self.G_x_hat,
            self.layer_norm_gamma,
            self.mc,
            self.stdc,
            self.batch_size,
            self.seq_len,
            self.d_model
        )

        # this layer_norm was little bit tricky many parallel reduction, and shapes to take care of
        check_layer_norm_back_grad = torch.allclose(back_x, self.layer_norm_back_x,atol=1e-4, rtol=1e-4)

        print_check('LayerNorm input gradient', check_layer_norm_back_grad)

        d_gamma, d_beta = layer_norm_beta_gamma_analytical(
            self.x,
            self.G_x_hat,
            self.mc,
            self.stdc,
            self.batch_size,
            self.seq_len
        )

        check_d_gamma = torch.allclose(d_gamma, self.d_gamma,atol=1e-4, rtol=1e-4)
        check_d_beta = torch.allclose(d_beta, self.d_beta,atol=1e-4, rtol=1e-4)

        print_check('LayerNorm scale gradient', check_d_gamma)

        print_check('LayerNorm bias gradient', check_d_beta)

        ''' Here we are checking the output projection linear background '''

        output_projection_weight_transpose = self.weight_contact.T
        check_output_projection_weight_transpose = torch.allclose(output_projection_weight_transpose, self.weight_contact_transpose,atol=1e-4, rtol=1e-4)


        print_check('Output projection weight transpose', check_output_projection_weight_transpose)

        ''' dl_dh = Gradient from the interface '''
        d_attention_torch = self.dl_dh @ self.weight_contact_transpose
        check_d_attention = torch.allclose(d_attention_torch, self.G, atol=1e-4, rtol=1e-4)

        print_check('Output projection input gradient', check_d_attention)

        check_d_bias_output_proj = torch.allclose(self.d_bias_output_proj,
                                                  self.dl_dh.sum(dim=(0, 1)),
                                                  atol=1e-4,
                                                  rtol=1e-4)
        # print(f"sum b, t: {self.dl_dh.sum(dim=(0, 1))}"
        #       f"{self.d_bias_output_proj}")

        # Here in the output project our upstream gradient is dl_dh

        print_check('Output projection bias gradient', check_d_bias_output_proj)

        # add the net gradient dl_dh and output from layer norm backward in torch here for the first part
        final_grad_torch = self.dl_dh + self.layer_norm_back_x

        check_final_grad = torch.allclose(
            final_grad_torch,
            self.final_gradient_from_add_residual,
            atol=1e-4,
            rtol=1e-4
        )

        print_check('Input gradient including residual', check_final_grad)


        d_token_embedding_torch = compute_token_embedding_grad(
            self.final_gradient_from_add_residual,
            self.token_ids,
            self.vocab_size
        )

        check_d_embedding = torch.allclose(d_token_embedding_torch, self.d_embeddings, atol=1e-4,
            rtol=1e-4)

        print_check('Token embedding gradient', check_d_embedding)

        # Here we will check the gradient calculation for attention heads Q, K and V.
        check_dw_q = torch.allclose(self.d_weight_q @ self.wqt, self.d_weight_q @ self.wqt, atol=1e-4, rtol=1e-4)
        check_dw_k = torch.allclose(self.d_weight_k @ self.wkt, self.d_weight_k @ self.wkt, atol=1e-4, rtol=1e-4)
        check_dw_v = torch.allclose(self.d_weight_v @ self.wvt, self.d_weight_v @ self.wvt, atol=1e-4, rtol=1e-4)

        print_check('Query weight gradient', check_dw_q)

        print_check('Key weight gradient', check_dw_k)

        print_check('Value weight gradient', check_dw_v)

        # sum along the b,t because that's the gradient for the bias term here.

        sum_dq = dQ_compact.sum(dim=(0, 1))

        check_d_bias_q = torch.allclose(self.upq.sum(dim=(0, 1)), self.d_bias_q, atol=1e-4, rtol=1e-4)
        check_d_bias_k = torch.allclose(self.upk.sum(dim=(0, 1)), self.d_bias_k, atol=1e-4, rtol=1e-4)
        check_d_bias_v = torch.allclose(self.upv.sum(dim=(0, 1)), self.d_bias_v, atol=1e-4, rtol=1e-4)

        print_check('Query bias gradient', check_d_bias_q)

        print_check('Key bias gradient', check_d_bias_k)

        print_check('Value bias gradient', check_d_bias_v)


        checking_x_output_project = torch.allclose(
            self.output_projection_x.transpose(1, 2),
            self.output_projection_xt,
            atol=1e-4,
            rtol=1e-4
        )

        print_check('Output projection input transpose', checking_x_output_project)


        local_weight_output_proj = (self.output_projection_xt @ self.dl_dh).sum(dim=(0))

        check_local_weight_lm_head = torch.allclose(local_weight_output_proj, self.d_weight_output_project, atol=1e-4, rtol=1e-4)

        print_check('Output projection weight gradient', check_local_weight_lm_head)

        # self.layer_norm_back_x is something that is fed into the layer norm.
        # meaning after net embedding.

        # Note: these are normalized so we need to set the comparison loose.
        check_transpose_norm_x = torch.allclose(
            self.normalized_x.transpose(1, 2),
            self.normalized_xt,
            atol=1e-3,
            rtol=1e-3
        )

        print_check('Normalized input transpose', check_transpose_norm_x)
