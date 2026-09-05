import torch


def layer_backward_analytical(x:torch.Tensor,
                              G:torch.Tensor,
                              gamma:torch.Tensor,
                              mean:torch.Tensor,
                              std:torch.Tensor,
                              eps: float=1e-8):
    """
    :param x: Shape(B,T,C)
    :param G: Shape(B,T,C)
    :param gamma: Shape(B, 1, 1,)
    :param mean: Shape(B, T, C)
    :param std: Shape(B, T, C)
    :param eps: float
    :return: backward tensor shaped (B,T, C)
    """

    