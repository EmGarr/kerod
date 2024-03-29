import tensorflow as tf
from kerod.utils.documentation import remove_unwanted_doc
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

__pdoc__ = {}


class FPN(layers.Layer):
    """Over your backbone feature build a FPN (inspired from tensorpack)"""

    def __init__(self, dim=256, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
        self._kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        num_level_pyramid = len(input_shape[0])
        self.lateral_connection_2345 = [
            layers.Conv2D(self._dim, (1, 1),
                          padding='same',
                          kernel_initializer=VarianceScaling(scale=1.),
                          kernel_regularizer=self._kernel_regularizer)
            for _ in range(num_level_pyramid)
        ]

        self.anti_aliasing_conv = [
            layers.Conv2D(self._dim, (3, 3),
                          padding='same',
                          kernel_initializer=VarianceScaling(scale=1.),
                          kernel_regularizer=self._kernel_regularizer)
            for _ in range(num_level_pyramid)
        ]

        super().build(input_shape)

    def call(self, inputs):
        """Over your backbone feature build a FPN (inspired from tensorpack)

        Arguments:
            inputs: A list of tensors of shape [N, height, widht, channels]

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

        p6 = layers.MaxPool2D()(lateral_connection_2345[-1])
        return lateral_connection_2345 + [p6]

    def get_config(self):
        base_config = super().get_config()
        base_config['dim'] = self._dim
        return base_config


remove_unwanted_doc(FPN, __pdoc__)
