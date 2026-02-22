import mlx.core as mx


def sigmoid(x: mx.array) -> mx.array:
    """
    Sigmoid activation function with numerical stability.
    """
    x = mx.clip(x, -500, 500)
    return 1 / (1 + mx.exp(-x))


def discriminator(x: mx.array) -> mx.array:
    """
    Classify inputs as real or fake.
    """
    _, input_dim = x.shape

    W1 = mx.random.normal(shape=(input_dim, 256)) * 0.02
    b1 = mx.zeros((256,))

    W2 = mx.random.normal(shape=(256, 128)) * 0.02
    b2 = mx.zeros((128,))

    W3 = mx.random.normal(shape=(128, 1)) * 0.02
    b3 = mx.zeros((1,))

    h1 = mx.matmul(x, W1) + b1
    h1 = mx.maximum(0.2 * h1, h1)

    h2 = mx.matmul(h1, W2) + b2
    h2 = mx.maximum(0.2 * h2, h2)

    logits = mx.matmul(h2, W3) + b3
    probs = sigmoid(logits)

    return probs
