
import math
import torch
import numpy as np
import torch.nn.functional as F
from ml_components.grad import load_tensor
from binary_reader.hyperparamaters import read_hyperparamaters

import numpy as np

class Analysis:
    def __init__(self):
        self.weight = load_tensor("./src/cache/cpp_out/attn_weight_q.bin",
                                  shape=(batch_size, batch_size),
                                  dtype=torch.float32)
        print(self.weight)