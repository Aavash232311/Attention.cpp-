import torch

from debug.static import RED, GREEN, RESET
from binary_reader.embedding_reader import Debugger
from ml_components.transformer import total_embeddings
from ml_components.positional_encoding import sinusoidal_positional_encoding

def verify_embeddings(
        d_model: int,
        seq_len: int,
        batch_size: int,
        vocab_size: int,
        num_heads: int,
        token_embeddings
):
    torch.set_printoptions(sci_mode=False, precision=4)

    debugger = Debugger(
        C=d_model,
        V=vocab_size,
        B=batch_size,
        T=seq_len,
        num_head=num_heads,
        folder='./src/cache/cpp_out'
    )

    pe = sinusoidal_positional_encoding(seq_len=seq_len, d_model=d_model)
    # same embedding that c++ uses after being released from python.
    k_total_embeddings = debugger.readEmbeddings('embedding.bin', batch_size * seq_len * d_model)
    x = debugger.readX("x.bin", seq_len * batch_size)

    py_total_embedding = total_embeddings(x=x, token_embedding_table=token_embeddings,
                                                positional_embedding_table=pe)

    check = torch.allclose(py_total_embedding, k_total_embeddings)
    if not check:
        print(f"Checking net embedding C++ kernel, status:{RED} {check} {RESET}")
        return
    print(f"Checking net embedding C++ kernel, status:{GREEN} {check} {RESET}")
