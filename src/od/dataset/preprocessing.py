import tensorflow as tf


def resize_to_min_dim(image, short_edge_length, max_dimension):
    """Resize an image given to the min size maintaining the aspect ratio.

    If one of the image dimensions is bigger than the max_dimension after resizing, it will scale
    the image such that its biggest dimension is equal to the max_dimension.

    Arguments :

    - *image*: A np.array of size [height, width, channels].
    - *short_edge_length*: minimum image dimension.
    - *max_dimension*: If the resized largest size is over max_dimension. Will use to max_dimension
    to compute the resizing ratio.

    Returns:
    - resized_image: The input image resized with the aspect_ratio preserved in float32
    """
    height, width = image.shape[:2]

    im_size_min = min(height, width)
    im_size_max = max(height, width)
    scale = short_edge_length / im_size_min
    # Prevent the biggest axis from being more than MAX_SIZE
    if tf.math.round(scale * im_size_max) > max_dimension:
        scale = max_dimension / im_size_max

    target_height = tf.cast(height * scale, dtype=tf.int32)
    target_width = tf.cast(width * scale, dtype=tf.int32)
    return tf.image.resize(tf.expand_dims(image, axis=0),
                           size=[target_height, target_width],
                           method=tf.image.ResizeMethod.BILINEAR)[0]
