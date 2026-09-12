import torch


def layer_backward_analytical(x: torch.Tensor,
                              G: torch.Tensor,
                              gamma: torch.Tensor,
                              mean: torch.Tensor,
                              std: torch.Tensor,
                              B: int,
                              T: int,
                              D: int):
    """
    :param T:
    :param B:
    :param x: Shape(B,T,C)
    :param G: Shape(B,T,C)
    :param gamma: Shape(C,)   i.e. (D,)
    :param mean: Shape(B, T)
    :param std: Shape(B, T)
    :param D: int d_model
    :return: backward tensor shaped (B,T,C)
    """

    epsilon = 1e-8
    mean = mean.reshape(B, T, 1)
    std = std.reshape(B, T, 1)
    gamma = gamma.reshape(1, 1, D)

    D: float = float(D)
    first_comp = 1.0 / (D * std)

    x_hat = (x - mean) / std
    dl_dx_hat = G * gamma

    second_comp = D * dl_dx_hat

    sum_term_1 = dl_dx_hat.sum(dim=-1, keepdim=True)
    sum_term_2 = (dl_dx_hat * x_hat).sum(dim=-1, keepdim=True)

    dx = first_comp * (second_comp - sum_term_1 - x_hat * sum_term_2)
    return dx

def layer_norm_beta_gamma_analytical(x: torch.Tensor,
                                       G: torch.Tensor,
                                       mean: torch.Tensor,
                                       std: torch.Tensor,
                                       B: int, T: int):
    """
    :param x: input after net embedding
    :param G: G_x_hat net upstream gradient
    :param mean: mean cache from forward pass
    :param std: std cache from forward pass
    :param B: batch size
    :param T: seq length
    :return: d_gamma, d_beta
    """

    mean = mean.reshape(B, T, 1)
    std = std.reshape(B, T, 1)

    x_hat = (x - mean) / std
    d_beta = G.sum(dim=(0, 1))
    d_gamma = (G * x_hat).sum(dim=(0, 1))

    return d_gamma, d_beta


def compute_token_embedding_grad(G: torch.Tensor,
                                 token_ids: torch.Tensor,
                                 vocab_size: int) -> torch.Tensor:
    B, T, C = G.shape
    assert token_ids.shape == (B, T), "token_ids must match G's (B, T) shape"

    grad_E = torch.zeros(vocab_size, C, dtype=G.dtype, device=G.device)

    flat_ids  = token_ids.reshape(-1).long()   # forcing integer 64
    flat_grad = G.reshape(-1, C)

    grad_E.index_add_(0, flat_ids, flat_grad)

    return grad_E