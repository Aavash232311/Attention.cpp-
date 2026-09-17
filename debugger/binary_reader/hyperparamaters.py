import os
import sys
import json
import torch
import warnings

# Read the released hyperparamaters .json file from C++ script.
warnings.filterwarnings("ignore", category=UserWarning, message="The given buffer is not writable")


def read_hyperparamaters(path="./src/cache/config.json"):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Critical Error: {e}")
        print("Exiting program.")
        sys.exit(1)

    print("Autograd engine C++ kenrel out")



    d_model = data['d_model']
    vocab_size = data['vocab_size']
    batch_size = data['batch_size']
    seq_len = data['seq_len']
    num_heads = data['num_heads']

    return d_model, vocab_size, batch_size, seq_len, num_heads




