import tensorflow as tf


def transform_fpcoor_for_tf(boxes: tf.Tensor, tensor_shape: tuple, crop_shape: tuple) -> tf.Tensor:
    """The way tf.image.crop_and_resize works (with normalized box):
    Initial point (the value of output[0]): x0_box * (W_img - 1)
    Spacing: w_box * (W_img - 1) / (W_crop - 1)
    Use the above grid to bilinear sample.

    However, what we want is (with fpcoor box):
    Spacing: w_box / W_crop
    Initial point: x0_box + spacing/2 - 0.5
    (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
        (0.0, 0.0) is the same as pixel value (0, 0))

    This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
    This function has been taken from tensorpack:
    (https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_box.py)

    Arguments:

    - *normalized_boxes*:  A Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]. These
    boxes have already been normalized in the feature space. The coordinates are not in
    the input image space.
    - *tensor_shape*:
    - *crop_shape*:

    Returns:

    A tensor of float32 and shape [N, ..., num_boxes, (y_min, x_min, y_max, x_max)]
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

    spacing_w = (x_max - x_min) / tf.cast(crop_shape[1], tf.float32)
    spacing_h = (y_max - y_min) / tf.cast(crop_shape[0], tf.float32)

    tensor_shape = [tf.cast(tensor_shape[0] - 1, tf.float32), tf.cast(tensor_shape[1] - 1, tf.float32)]
    ny0 = (y_min + spacing_h / 2 - 0.5) / tensor_shape[0]
    nx0 = (x_min + spacing_w / 2 - 0.5) / tensor_shape[1]

    nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / tensor_shape[1]
    nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / tensor_shape[0]

    return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)


def convert_to_center_coordinates(boxes: tf.Tensor) -> tf.Tensor:
    """Convert boxes to their center coordinates

    y_min, x_min, y_max, x_max -> y_cent, x_cent, h, w

    Arguments:

    - *boxes*: A Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]

    Returns:

    A tensor of float32 and shape [N, ..., num_boxes, (ycenter, xcenter, height, width)]
    """
    y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
    width = x_max - x_min
    height = y_max - y_min
    ycenter = y_min + height / 2.
    xcenter = x_min + width / 2.
    return tf.concat([ycenter, xcenter, height, width], axis=-1)


def compute_area(boxes: tf.Tensor) -> tf.Tensor:
    """Compute the area of boxes.

    Arguments:

    - *boxes*: Tensor of float32 and shape [N, ..., (y_min,x_min,y_max_,x_max)]

    Returns:

    A tensor of float32 and shape [N, ..., num_boxes]
    """
    with tf.name_scope('Area'):
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def compute_intersection(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """Compute pairwise intersection areas between boxes.

    Arguments:

    - *boxes1*: Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *boxes2*: Tensor of float32 and shape [N, ..., (y_max,x_max,y_max,x_max)]

    Returns:

    A tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope('Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxes1, num_or_size_splits=4, axis=1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxes2, num_or_size_splits=4, axis=1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
        intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
        intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths


def compute_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """Computes pairwise intersection-over-union between box collections.

    Arguments:

    - *boxes1*: Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *boxes2*: Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]

    Returns:

    A tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope('IOU'):
        intersections = compute_intersection(boxes1, boxes2)
        areas1 = compute_area(boxes1)
        areas2 = compute_area(boxes2)
        unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections),
                        tf.truediv(intersections, unions))


def normalize_box_coordinates(boxes, height: int, width: int):
    """ Normalize the boxes coordinates with image shape and transpose the coordinates

    Arguments:

    - *boxes*: Tensor of float32 and shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *height*: An integer
    - *width*: An integer
    """

    y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    y_min = y_min / height
    x_min = x_min / width
    y_max = y_max / height
    x_max = x_max / width

    # Won't be backpropagated to rois anyway, but to save time
    boxes = tf.stop_gradient(tf.concat([y_min, x_min, x_max, y_max], axis=-1))
    return boxes
