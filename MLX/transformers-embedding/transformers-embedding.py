import math
import mlx.core as mx


class Embedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = mx.random.normal(shape=(vocab_size, d_model)) * (1.0 / math.sqrt(d_model))

    def __call__(self, tokens: mx.array) -> mx.array:
        return mx.take(self.weight, tokens, axis=0)


def create_embedding_layer(vocab_size: int, d_model: int) -> Embedding:
    """
    Create an embedding layer.
    """
    return Embedding(vocab_size, d_model)


def embed_tokens(embedding: Embedding, tokens: mx.array, d_model: int) -> mx.array:
    """
    Convert token indices to scaled embeddings.
    """
    embedded = embedding(tokens)
    scaled_embeddings = embedded * math.sqrt(d_model)
    return scaled_embeddings
