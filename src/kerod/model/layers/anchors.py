import tensorflow as tf
from kerod.core.constants import MAX_IMAGE_DIMENSION
from kerod.utils.documentation import remove_unwanted_doc

__pdoc__ = {}


def generate_anchors(stride: int, scales: tf.Tensor, ratios: tf.Tensor, max_size: int):
    """Will generate a determistic grid based on the input images dimension.

    The anchors will be generated once and for all on the biggest possible image.
    At each forward according to the shape of the tensor we can extract the corresponding anchors.

    Arguments:
        stride: Downscaling ratio compared to the original image. At stride 16 your original
            image will be 16 times bigger than your actual tensor
        scales: The scale of the anchors e.g: 8, 16, 32
        ratios: The ratios are the different shapes that you want to apply on your anchors.
            e.g: (0.5, 1, 2)
        max_size: Maximum size of the input image. The anchors will computed once and for all.

    Returns:
        A tensor of shape [num_scales * num_ratios * height * width, 4].
        The anchors have the format [y_min, x_min, y_max, x_max].
    """
    ratios, scales = tf.meshgrid(ratios, scales)
    scales = tf.reshape(scales, [-1])
    ratios = tf.reshape(ratios, [-1])
    ratio_sqrts = tf.sqrt(ratios)
    widths = scales / ratio_sqrts
    heights = ratios * widths
    base_anchors = tf.stack([-widths, -heights, widths, heights], axis=-1) * 0.5

    max_stride = tf.math.ceil(max_size / stride)
    y_centers = tf.cast(tf.range(max_stride), dtype=scales.dtype) * stride
    x_centers = tf.cast(tf.range(max_stride), dtype=scales.dtype) * stride
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    shifts = tf.stack([x_centers, y_centers, x_centers, y_centers], axis=-1)

    anchors = tf.expand_dims(base_anchors, 0) + tf.expand_dims(shifts, 2)

    return tf.gather(anchors, [1, 0, 3, 2], axis=-1)


class Anchors(tf.keras.layers.Layer):
    """Will generate a determistic grid and store it in memory to avoid recompute it at each run.

     At each forward according to the shape of the tensor we can extract the corresponding anchors.

    Arguments:
        stride: Downscaling ratio compared to the original image. At stride 16 your original
            image will be 16 times bigger than your actual tensor
        scales: The scale of the anchors e.g: 8, 16, 32
        ratios: The ratios are the different shapes that you want to apply on your anchors.
            e.g: (0.5, 1, 2)
        max_size: Maximum size of the input image. The anchors will computed once and for all.

    Call arguments:
        inputs: A tensor of shape [batch_size, height, widht, channel]

    Call returns:
        A tensor of shape [num_scales * num_ratios * height * width, 4].
        The anchors have the format [y_min, x_min, y_max, x_max].
    """

    def __init__(self, stride, scales, ratios, **kwargs):
        super().__init__(**kwargs)

        self._stride = stride
        self._scales = scales
        self._ratios = ratios
        self._anchors = generate_anchors(stride,
                                         tf.constant([scales], self._compute_dtype),
                                         tf.constant(ratios, self._compute_dtype),
                                         max_size=MAX_IMAGE_DIMENSION)

    def call(self, inputs):
        """Return anchors based on the shape of the input tensors

        Arguments:
            inputs: A tensor of shape [batch_size, height, widht, channel]

        Returns:
            A tensor of shape [num_scales * num_ratios * height * width, 4].
            The anchors have the format [y_min, x_min, y_max, x_max].
        """
        shape = tf.shape(inputs)
        height, width = shape[1], shape[2]
        anchors = self._anchors[:height, :width]
        return tf.reshape(anchors, (-1, 4))

    def get_config(self):
        config = super().get_config()
        config['stride'] = self._stride
        config['scales'] = self._scales
        config['ratios'] = self._ratios
        return config


remove_unwanted_doc(Anchors, __pdoc__)
