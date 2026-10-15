import math
import torch
import numpy as np
import torch.nn.functional as F
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

    check_qkt = torch.allclose(q_reshaped @ kt, qkt, atol=1e-6, rtol=1e-6)
    print("Check QKT matmul: ", check_qkt)

    sqrt_d_head = load_tensor("./src/cache/cpp_out/sqrt_d_head.bin",
                                    shape=(batch_size, num_heads, seq_len, seq_len),
                                    dtype=np.float32).to(device)

    casual_masking = load_tensor("./src/cache/cpp_out/casual_masking.bin",
                                 shape=(batch_size, num_heads, seq_len, seq_len),
                                 dtype=np.float32).to(device)

    check_division = torch.allclose(
        sqrt_d_head,
        qkt / math.sqrt(head_dim),
        atol=1e-6,
        rtol=1e-6
    )
    print(f"Check division by d_head: {check_division}")

    T = sqrt_d_head.size(-1)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=sqrt_d_head.device), diagonal=1)

    # Masked positions must be -inf
    check_masked = torch.isneginf(casual_masking[..., future]).all()

    # Unmasked positions must be unchanged from the sqrt_d_head scores
    check_kept = torch.allclose(
        casual_masking[..., ~future],
        sqrt_d_head[..., ~future],
        atol=1e-6,
        rtol=1e-6,
    )

    print(f"check casual masking: {check_masked}")


    softmax_attn = load_tensor("./src/cache/cpp_out/softmaxed_attn.bin",
                               shape=(batch_size, num_heads, seq_len, seq_len),
                               dtype=np.float32).to(device)

    ref_softmax = torch.softmax(casual_masking, dim=-1)

    check_softmax = torch.allclose(
        softmax_attn,
        ref_softmax,
        atol=1e-6,
        rtol=1e-5,
    )

    attn_out_cpp = load_tensor("./src/cache/cpp_out/attn_score.bin",
                               shape=(batch_size, num_heads, seq_len, head_dim),
                               dtype=np.float32).to(device)

    ref_out = softmax_attn @ v_reshaped  # (B, H, T, head_dim)

    check_attn_out = torch.allclose(attn_out_cpp, ref_out, atol=1e-5, rtol=1e-5)
    print(f"Final attention score: {check_attn_out}")

    re_shaped_multi_head = load_tensor("./src/cache/cpp_out/re_shaped_multi_head.bin",
                                       shape=(batch_size, seq_len, d_model),
                                       dtype=np.float32).to(device)
    check_multi_head = torch.allclose(
        re_shaped_multi_head, # (B, H, T, head_dim) -> (B, T, H, head_dim) -> BTC
        attn_out_cpp.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model),
        atol=1e-6,
        rtol=1e-5,
    )

    temp_x = load_tensor("./src/cache/cpp_out/temp_x.bin",
                         shape=(batch_size, seq_len, d_model),
                         dtype=np.float32).to(device)

    final_out_attn = load_tensor("./src/cache/cpp_out/final_attn_out.bin",
                                 shape=(batch_size, seq_len, d_model),
                                 dtype=np.float32).to(device)

    out_put_proj = load_tensor("./src/cache/cpp_out/out_put_proj.bin",
                               shape=(batch_size, seq_len, d_model),
                               dtype=np.float32).to(device)

    print(f"Reshaped attention score: {check_multi_head}")

    ref_residual = temp_x + out_put_proj

    check_residual_adding = torch.allclose(
        final_out_attn,
        ref_residual,
        atol=1e-6,
        rtol=1e-5,
    )


    normalized_x = load_tensor("./src/cache/cpp_out/layer_norm_forward.bin",
                               shape=(batch_size, seq_len, d_model),
                               dtype=np.float32).to(device)

    gamma = load_tensor("./src/cache/cpp_out/gamma_device.bin",
                        shape=(d_model),
                        dtype=np.float32).to(device)

    beta = load_tensor("./src/cache/cpp_out/beta_device.bin",
                        shape=(d_model),
                        dtype=np.float32).to(device)

    ref_ln = F.layer_norm(temp_x, (d_model,), weight=gamma, bias=beta, eps=1e-8)

    check_layer_norm = torch.allclose(normalized_x, ref_ln, atol=1e-5, rtol=1e-5)

    print(f"Check layer_norm: {check_layer_norm}")