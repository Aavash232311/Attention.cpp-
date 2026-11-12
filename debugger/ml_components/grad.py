import numpy as np
import torch

def load_tensor(path, shape, dtype=np.float32, offset=0):
    arr = np.fromfile(path, dtype=dtype, count=int(np.prod(shape)), offset=offset)
    return torch.from_numpy(arr.reshape(shape))
