import torch
import numpy as np
from ml_components.grad import load_tensor
from binary_reader.hyperparamaters import read_hyperparamaters


def quick_attention_check():

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")


    d_model, vocab_size, batch_size, seq_len, num_heads = read_hyperparamaters()

    head_dim = d_model // num_heads

    q = load_tensor("./src/cache/cpp_out/q.bin",
                    shape=(batch_size, num_heads, seq_len, head_dim),
                    dtype=np.float32).to(device)

    kt = load_tensor("./src/cache/cpp_out/tranposed_k.bin",
                     shape=(batch_size, num_heads, head_dim, seq_len),
                     dtype=np.float32).to(device)

    qkt = load_tensor("./src/cache/cpp_out/qkt.bin",
                      shape=(batch_size, num_heads, seq_len, seq_len),
                      dtype=np.float32).to(device)

    reference_qkt = q @ kt

    q_org = load_tensor("./src/cache/cpp_out/q_org.bin",
                        shape=(batch_size, seq_len, num_heads, head_dim),
                        dtype=np.float32).to(device)

    k_org = load_tensor("./src/cache/cpp_out/k_org.bin",
                        shape=(batch_size, seq_len, num_heads, head_dim),
                        dtype=np.float32).to(device)

    v_org = load_tensor("./src/cache/cpp_out/v_org.bin",
                        shape=(batch_size, seq_len, num_heads, head_dim),
                        dtype=np.float32).to(device)

    q_reshaped = load_tensor("./src/cache/cpp_out/q_reshaped.bin",
                        shape=(batch_size, num_heads, seq_len, head_dim),
                        dtype=np.float32).to(device)

    k_reshaped = load_tensor("./src/cache/cpp_out/k_reshaped.bin",
                             shape=(batch_size, num_heads, seq_len, head_dim),
                             dtype=np.float32).to(device)
    v_reshaped = load_tensor("./src/cache/cpp_out/v_reshaped.bin",
                             shape=(batch_size, num_heads, seq_len, head_dim),
                             dtype=np.float32).to(device)

    print("q:", torch.allclose(q_reshaped, q_org.transpose(1, 2).contiguous(), atol=1e-6, rtol=1e-6))
    print("k:", torch.allclose(k_reshaped, k_org.transpose(1, 2).contiguous(), atol=1e-6, rtol=1e-6))
    print("v:", torch.allclose(v_reshaped, v_org.transpose(1, 2).contiguous(), atol=1e-6, rtol=1e-6))

    print("kt:", torch.allclose(kt, k_reshaped.transpose(-1, -2).contiguous(), atol=1e-6, rtol=1e-6))
    print("q vs k differ:", not torch.allclose(q_reshaped, k_reshaped))

    print(torch.allclose(q_reshaped, v_reshaped, atol=1e-6, rtol=1e-6))

    print(q_reshaped[0])