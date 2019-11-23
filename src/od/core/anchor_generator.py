import tensorflow as tf
from tensorflow.keras import backend as K


def generate_anchors(stride: int, scales: tf.Tensor, ratios: tf.Tensor, tensor_shape: list):
    """Will generate a determistic grid based on the input images dimension.

    Arguments:

    - *stride*: Downscaling ratio compared to the original image. At stride 16 your original
        image will be 16 times bigger than your actual tensor
    - *scales*: The scale of the anchors e.g: 8, 16, 32
    - *ratios*: The ratios are the different shapes that you want to apply on your anchors.
            e.g: (0.5, 1, 2)
    - *tensor_shape*: An array of 4 dim [batch_size, height, width, num_channels]

    Returns:

    A tensor of shape [num_scales * num_ratios * height * width, 4].
    The anchors have the format [y_min, x_min, y_max, x_max].

    """
    ratios_grid, scales_grid = tf.meshgrid(ratios, scales)
    scales = tf.reshape(scales_grid, [-1])
    ratios = tf.reshape(ratios_grid, [-1])

    ratio_sqrts = tf.sqrt(ratios)
    widths = scales / ratio_sqrts
    heights = ratios * widths
    base_anchors = tf.stack([-widths, -heights, widths, heights], axis=1) * 0.5

    y_centers = tf.cast(range(tensor_shape[1]), dtype=scales.dtype) * stride
    x_centers = tf.cast(range(tensor_shape[2]), dtype=scales.dtype) * stride
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
    x_centers = tf.reshape(x_centers, [-1])
    y_centers = tf.reshape(y_centers, [-1])

    shifts = tf.stack([x_centers, y_centers, x_centers, y_centers], axis=1)

    anchors = tf.expand_dims(base_anchors, 0) + tf.expand_dims(shifts, 1)
    anchors = tf.reshape(anchors, shape=(-1, 4))
    return tf.gather(anchors, [1, 0, 3, 2], axis=-1)
