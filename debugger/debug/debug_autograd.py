import torch

from debug.static import print_check
from binary_reader.autograd_binary_reader import read_lm_head_backward_tensors

def debug_autograd(
        d_model: int,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        num_heads: int
):
    torch.set_printoptions(precision=8, sci_mode=False, threshold=float('inf'))
    (delta, y_predicted,
     y_actual, h, dl_dw_kernel,
     h_t, wt, w, dl_dh_kernel, d_bias_lm_head) = read_lm_head_backward_tensors(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        d_model=d_model
    )

    print("\n")
    print('*' * 60)
    print("Section AUTO GRAD (The Chain Rule of Derivative) ")
    print('*' * 60)
    # transpose h
    transposing_h = h.transpose(1, 2)
    check_ht_transpose = torch.allclose(transposing_h, h_t)


    print_check('LM head input transpose', check_ht_transpose)


    # broadcasting for last-dimension
    delta_torch = y_predicted - y_actual

    # Note-: GIGO sometimes you might just be transposing the garbage who knows
    # if you think that is the case then manually print and see from the C++ script.

    check_delta_across_kernels = torch.allclose(delta_torch, delta)
    print_check('Cross-entropy logits gradient', check_delta_across_kernels)

    check_wt = torch.allclose(w.T, wt)

    print_check('LM head weight transpose', check_wt)

    dl_dh_torch = delta @ wt



    # we need to account for small rounding errors
    check_dl_dh = torch.allclose(dl_dh_torch, dl_dh_kernel, atol=1e-4, rtol=1e-4)

    print_check('LM head input gradient', check_dl_dh)

    check_d_bias_lm_head = torch.allclose(delta_torch.sum(dim=(0, 1)), d_bias_lm_head)

    print_check('LM head bias gradient', check_d_bias_lm_head)

    local_weight_lm_head = (h_t @ delta).sum(dim=(0))

    check_local_weight_lm_head = torch.allclose(local_weight_lm_head, dl_dw_kernel, atol=1e-4, rtol=1e-4)

    print_check('LM head weight gradient', check_local_weight_lm_head)
