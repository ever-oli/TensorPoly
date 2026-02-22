import mlx.core as mx


def relu(x: mx.array) -> mx.array:
    return mx.maximum(0, x)


class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels
        self.out_ch = out_channels

        self.W1 = mx.random.normal(shape=(in_channels, bottleneck_channels)) * 0.01
        self.W2 = mx.random.normal(shape=(bottleneck_channels, bottleneck_channels)) * 0.01
        self.W3 = mx.random.normal(shape=(bottleneck_channels, out_channels)) * 0.01

        if in_channels != out_channels:
            self.Ws = mx.random.normal(shape=(in_channels, out_channels)) * 0.01
        else:
            self.Ws = None

    def forward(self, x: mx.array) -> mx.array:
        identity = x

        out = mx.matmul(x, self.W1)
        out = relu(out)

        out = mx.matmul(out, self.W2)
        out = relu(out)

        out = mx.matmul(out, self.W3)

        if self.Ws is not None:
            identity = mx.matmul(identity, self.Ws)

        out = out + identity
        out = relu(out)

        return out
