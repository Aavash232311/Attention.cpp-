import os
import sys
import json
import torch
import warnings

# Read the released hyperparamaters .json file from C++ script.
warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")


def read_hyperparamaters(path="./src/cache/config.json"):
    # Releases hyperparamaters from C++ in a JSON file. Compile and execute, this approach is known to be slow, but it's okay.
    os.system("nvcc -DDEBUG -DPARAMS src/main.cpp src/kernel/utils.cu src/forward/Kernel/layer_norm.cu src/forward/Kernel/embedding.cu src/forward/Kernel/linear.cu src/forward/Kernel/attention_head.cu src/forward/Kernel/interface.cu src/backpropagation/Kernel/interface_back.cu src/backpropagation/Kernel/flash_attention.cu -o src/bin/attention")
    os.system("./src/bin/attention")


    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Critical Error: {e}")
        print("Exiting program.")
        sys.exit(1)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("Autograd engine C++ kenrel out")



    d_model = data['d_model']
    vocab_size = data['vocab_size']
    batch_size = data['batch_size']
    seq_len = data['seq_len']
    num_heads = data['num_heads']

    return d_model, vocab_size, batch_size, seq_len, num_heads




