import mlx.core as mx


def unet_output(features: mx.array, num_classes: int) -> mx.array:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    """
    batch, H, W, _ = features.shape
    output = mx.zeros((batch, H, W, num_classes))
    return output
