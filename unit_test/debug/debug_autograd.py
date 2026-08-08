import os
import torch

from binary_reader.autograd_binary_reader import Reader

def debug_autograd(
        d_model: int,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        num_heads: int
):
    # Loads C++ with python generated parameters to avoid randomness
    os.system(
        "nvcc -DDEBUG -DDRUN src/main.cpp src/kernel/utils.cu src/forward/Kernel/layer_norm.cu src/forward/Kernel/embedding.cu src/forward/Kernel/linear.cu src/forward/Kernel/attention_head.cu src/forward/Kernel/interface.cu src/backpropagation/Kernel/interface_back.cu -o src/bin/attention")
    os.system("./src/bin/attention")

    delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w = Reader(
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
    print(f"h^t transpose kernel: {torch.allclose(transposing_h, h_t)}")
    # we need to verify delta h^t
    dl_dw_torch = transposing_h @ delta

    # broadcasting for last-dimension
    delta_torch = y_predicted - y_actual

    # Note-: GIGO sometimes you might just be transposing the garbage who knows
    # if you think that is the case then manually print and see from the C++ script.
    print(f"Checking delta across kernels {torch.allclose(delta_torch, delta)}")
    print(f"Checking wt transpose kernel: {torch.allclose(w.T, wt)}")
