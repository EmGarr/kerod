import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras.initializers import VarianceScaling


class Pyramid(KL.Layer):
    """Over your backbone feature build a FPN (inspired from tensorpack)"""

    def __init__(self, dim=256, **kwargs):
        super().__init__(**kwargs)
        # DIRTY but hardcoded value for the number of layers on which the pyramid will be build
        # It is necessary to create the convolution
        num_outputs = 4
        self.lateral_connection_2345 = [
            KL.Conv2D(dim, (1, 1), padding='same', kernel_initializer=VarianceScaling(scale=1.))
            for _ in range(num_outputs)
        ]

        self.anti_aliasing_conv = [
            KL.Conv2D(dim, (3, 3), padding='same', kernel_initializer=VarianceScaling(scale=1.))
            for _ in range(num_outputs)
        ]

    def call(self, inputs):
        """Over your backbone feature build a FPN (inspired from tensorpack)

        Arguments:

        - *inputs*: A list of tensors of shape [N, height, widht, channels]
        
        Returns:

        A list of tensors of shape [N + 1, height, width, channels]
        """

        lateral_connection_2345 = [
            conv(tensor) for tensor, conv in zip(inputs, self.lateral_connection_2345)
        ]

        lat_sum_5432 = []
        for idx, block in enumerate(lateral_connection_2345[::-1]):
            if idx > 0:
                up_shape = tf.shape(block)
                block = block + tf.image.resize(lat_sum_5432[-1], [up_shape[1], up_shape[2]],
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            lat_sum_5432.append(block)

        # 3×3convolution on each merged map to generate the final feature map,
        # which is to reduce the aliasing effect of upsampling.
        lateral_connection_2345 = [
            conv(tensor) for conv, tensor in zip(self.anti_aliasing_conv, lat_sum_5432[::-1])
        ]

        p6 = KL.MaxPool2D()(lateral_connection_2345[-1])
        return lateral_connection_2345 + [p6]
