import torch

from debug.static import RED, GREEN, RESET
from binary_reader.autograd_binary_reader import Reader

def debug_autograd(
        d_model: int,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        num_heads: int
):
    torch.set_printoptions(precision=8, sci_mode=False, threshold=float('inf'))

    delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w, G_cnt = Reader(
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

    if check_ht_transpose:
        print(f"h^t transpose kernel: {GREEN} {torch.allclose(transposing_h, h_t)} {RESET}")
    else:
        print(f"h^t transpose kernel: {RED} {check_ht_transpose} {RESET}")
    # we need to verify delta h^t
    dl_dw_torch = transposing_h @ delta

    # broadcasting for last-dimension
    delta_torch = y_predicted - y_actual

    # Note-: GIGO sometimes you might just be transposing the garbage who knows
    # if you think that is the case then manually print and see from the C++ script.

    check_delta_across_kernels = torch.allclose(delta_torch, delta)
    if not check_delta_across_kernels:
        print(f"Checking delta across kernels: {RED} {check_delta_across_kernels} {RESET}")
    else:
        print(f"Checking delta across kernels: {GREEN} {check_delta_across_kernels} {RESET}")

    check_wt = torch.allclose(w.T, wt)

    if not check_wt:
        print(f"Checking wt transpose kernel: {RED} {torch.allclose(w.T, wt)} {RESET}")
    else:
        print(f"Checking wt transpose kernel: {GREEN} {check_wt} {RESET}")

    return dl_dw_kernel