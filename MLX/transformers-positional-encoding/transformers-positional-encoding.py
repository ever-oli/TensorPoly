import mlx.core as mx


def positional_encoding(seq_length: int, d_model: int) -> mx.array:
    """
    Generate sinusoidal positional encodings.
    """
    position = mx.arange(seq_length)[:, None]
    dims = mx.arange(d_model)[None, :]

    exponent = (2 * (dims // 2)) / d_model
    angle_rates = mx.exp(-mx.log(mx.array(10000.0)) * exponent)
    angles = position * angle_rates

    return mx.where((dims % 2) == 0, mx.sin(angles), mx.cos(angles))
