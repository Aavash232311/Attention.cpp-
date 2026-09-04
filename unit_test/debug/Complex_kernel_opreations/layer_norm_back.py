import torch

def layer_norm_back(x: torch.Tensor,
                    g: torch.Tensor,
                    mean_cache: torch.Tensor,
                    std_dev_cache: torch.Tensor,
                    gamma: torch.Tensor,
                    B: int,
                    T: int,
                    C: int):
    # reshape mean/std so they broadcast against (B, T, C) along the C dimension

    # I am too old for these, but we are doing something like (B*T, C) in kernels
    mean = mean_cache.view(B, T, 1)  # (B, T, 1)
    std = std_dev_cache.view(B, T, 1)  # (B, T, 1)

    x_hat = (x - mean) / std  # (B, T, C), the normalized values

    # dL/dx_hat_i = g_i * gamma_i  (gamma broadcasts over B, T)
    dxhat = g * gamma  # (B, T, C)

    # the two row-wise sums (one scalar per (b, t) row, summed over C)
    sum1 = dxhat.sum(dim=-1, keepdim=True)  # (B, T, 1)
    sum2 = (dxhat * x_hat).sum(dim=-1, keepdim=True)  # (B, T, 1)

    # final closed-form gradient
    dx = (1.0 / (C * std)) * (C * dxhat - sum1 - x_hat * sum2)  # (B, T, C)

    return dx
