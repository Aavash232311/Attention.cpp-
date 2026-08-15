import torch
import numpy as np

from ml_components.grad import load_tensor

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

def ReaderFlashAttention(
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        head_dim: int
):
    # batch_size * num_heads * seq_len * seq_len = P
    # batch_size * num_heads * seq_len * head_dim = V
    P = load_tensor('./src/cache/cpp_out/P.bin', shape=(batch_size, num_heads, seq_len, seq_len), dtype=np.float32).to(device)
    V = load_tensor('./src/cache/cpp_out/V.bin', shape=(batch_size, num_heads, seq_len, head_dim), dtype=np.float32).to(device)

    # Shape is same, value is different after the transpose operation takes place
    PT = load_tensor('./src/cache/cpp_out/pt.bin', shape=(batch_size, num_heads, seq_len, seq_len), dtype=np.float32).to(device)
    VT = load_tensor('./src/cache/cpp_out/vt.bin', shape=(batch_size, num_heads, head_dim, seq_len), dtype=np.float32).to(device)

    G_unc = load_tensor('./src/cache/cpp_out/G_uncontact.bin', shape=(batch_size, num_heads, seq_len, head_dim), dtype=np.float32).to(device)
    dl_dh = load_tensor('./src/cache/cpp_out/dl_dh.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(
        device)


    dp = load_tensor("./src/cache/cpp_out/dp.bin",
                     shape=(batch_size, num_heads, seq_len, seq_len),
                     dtype=np.float32).to(device)

    dV = load_tensor("./src/cache/cpp_out/dv.bin",
                     shape=(batch_size, num_heads, seq_len, head_dim),
                     dtype=np.float32).to(device)

    return P, V, PT, VT, G_unc, dl_dh, dp, dV

def Reader(
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int
):
    delta = load_tensor('./src/cache/cpp_out/delta.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    y_predicted = load_tensor('./src/cache/cpp_out/y_prediced.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    y_actual = load_tensor('./src/cache/cpp_out/y_actual.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32).to(device)
    h = load_tensor('./src/cache/cpp_out/h.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(device)
    # (B, C, vocab_size)
    dl_dw_kernel = load_tensor('./src/cache/cpp_out/dl_dw.bin', shape=(batch_size, d_model, vocab_size), dtype=np.float32).to(device)
    h_t = load_tensor('./src/cache/cpp_out/h_t.bin', shape=(batch_size, d_model, seq_len), dtype=np.float32).to(device)

    wt = load_tensor('./src/cache/cpp_out/wt.bin', shape=(vocab_size, d_model), dtype=np.float32).to(device)
    w = load_tensor('./src/cache/cpp_out/w.bin', shape=(d_model, vocab_size), dtype=np.float32).to(device)

    dl_dh = load_tensor('./src/cache/cpp_out/dl_dh.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32).to(device)

    return delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w, dl_dh

