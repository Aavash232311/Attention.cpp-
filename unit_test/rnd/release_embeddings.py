
from ml_components.normal_dis import  Predictability


def release_token_embeddings(
        d_model: int,
        vocab_size: int,
        seq_len: int,
        batch_size: int,
):
    predictability = Predictability(
        d_model=d_model,
        vocab_size=vocab_size,
        seq_len=seq_len,
        batch_size=batch_size
    )

    # Releases token embedding for C++ and returns the same embedding
    token_embeddings = predictability.token_embeddings(
        d_model,
        vocab_size
    )

    return token_embeddings
