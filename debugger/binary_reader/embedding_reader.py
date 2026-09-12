import os
import torch
import numpy as np

class Debugger:

    def __init__(self, C, V, B, T, num_head, folder):
        self.folder = folder
        self.d_model = C
        self.batch_size = B
        self.seq_len = T
        self.num_heads = num_head
        self.vocab_size = V

    def readEmbeddings(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(str(path), size=N, dtype=torch.float32)
        return flat.reshape(self.batch_size, self.seq_len, self.d_model)

    def readX(self, file_name, N):
        path = os.path.join(self.folder, file_name)

        flat = torch.from_file(str(path), size=N, dtype=torch.int32)
        return flat.reshape(self.seq_len, self.batch_size)

    def read_predictions(self, file_name, np_dtype=np.float32):
        path = os.path.join(self.folder, file_name)
        tensor = torch.from_numpy(np.fromfile(path, dtype=np_dtype))
        return tensor.reshape(self.batch_size, self.seq_len, self.vocab_size)