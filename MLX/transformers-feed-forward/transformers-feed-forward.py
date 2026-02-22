import mlx.core as mx


def feed_forward(x: mx.array, W1: mx.array, b1: mx.array, W2: mx.array, b2: mx.array) -> mx.array:
    """
    Apply position-wise feed-forward network.
    """
    hidden = mx.matmul(x, W1) + b1
    relu_out = mx.maximum(0, hidden)
    output = mx.matmul(relu_out, W2) + b2
    return output
