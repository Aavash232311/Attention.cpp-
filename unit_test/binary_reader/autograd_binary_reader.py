import numpy as np
from ml_components.grad import Autograd, load_tensor




def Reader(
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        d_model: int
):
    # To avoid monolith here

    delta = load_tensor('./src/cache/cpp_out/delta.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32)
    y_predicted = load_tensor('./src/cache/cpp_out/y_prediced.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32)
    y_actual = load_tensor('./src/cache/cpp_out/y_actual.bin', shape=(batch_size, seq_len, vocab_size), dtype=np.float32)
    h = load_tensor('./src/cache/cpp_out/h.bin', shape=(batch_size, seq_len, d_model), dtype=np.float32)
    # (B, C, vocab_size)
    dl_dw_kernel = load_tensor('./src/cache/cpp_out/dl_dw.bin', shape=(batch_size, d_model, vocab_size), dtype=np.float32)
    h_t = load_tensor('./src/cache/cpp_out/h_t.bin', shape=(batch_size, d_model, seq_len), dtype=np.float32)

    wt = load_tensor('./src/cache/cpp_out/wt.bin', shape=(vocab_size, d_model), dtype=np.float32)
    w = load_tensor('./src/cache/cpp_out/w.bin', shape=(d_model, vocab_size), dtype=np.float32)

    return delta, y_predicted, y_actual, h, dl_dw_kernel, h_t, wt, w

