import tensorflow as tf
import tensorflow.keras.layers as tfkl
import gin

from tensorflow.keras.initializers import VarianceScaling


@gin.configurable
def build_fpn(features, dim=256):
    """
    Over your backbone feature build a FPN (inspired from tensorpack)
    :param features: A list of tensors of shape [N, height, widht, channels]
    :returns: A list of tensors of shape [N + 1, height, width, channels]

    """

    lateral_connection_2345 = [
        tfkl.Conv2D(dim, (1, 1), padding='same', kernel_initializer=VarianceScaling(scale=1.))(c)
        for c in features
    ]

    lat_sum_5432 = []
    for idx, block in enumerate(lateral_connection_2345[::-1]):
        if idx > 0:
            up_shape = tf.shape(block)
            block = block + tf.image.resize(
                lat_sum_5432[-1], [up_shape[1], up_shape[2]],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        lat_sum_5432.append(block)

    # 3×3convolution on each merged map to generate the final feature map,
    # which is to reduce the aliasing effect of upsampling. We do not apply the conv on 5
    lateral_connection_2345 = [
        tfkl.Conv2D(dim, (3, 3), padding='same', kernel_initializer=VarianceScaling(scale=1.))(c)
        if i != len(lat_sum_5432) - 1 else c for i, c in enumerate(lat_sum_5432[::-1])
    ]

    p6 = tfkl.MaxPool2D()(lateral_connection_2345[-1])
    return lateral_connection_2345 + [p6]
