import tensorflow as tf

def transform_fpcoor_for_tf(boxes, image_shape, crop_shape) -> tf.Tensor:
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

    - *boxes*:
    - *image_shape*:
    - *crop_shape*:

    Returns:
        normalized_boxes: 
    """
    y0, x0, y1, x1 = tf.split(boxes, 4, axis=-1)

    spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
    spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

    image_shape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
    nx0 = (x0 + spacing_w / 2 - 0.5) / image_shape[1]
    ny0 = (y0 + spacing_h / 2 - 0.5) / image_shape[0]

    nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / image_shape[1]
    nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / image_shape[0]

    return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)


def compute_area(boxes) -> tf.Tensor:
    """Compute the area of boxes.

    Arugments:

    - *boxes*: Tensor of float32 and shape [N, ..., (y1,x1,y2,x2)]

    Returns: 
        A tensor of float32 and shape [N, ..., num_boxes]
    """
    with tf.name_scope('Area'):
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def compute_intersection(boxes1, boxes2) -> tf.Tensor:
    """Compute pairwise intersection areas between boxes.

    Arguments:

    - *boxes1*: Tensor of float32 and shape [N, ..., (y1,x1,y2,x2)]
    - *boxes2*: Tensor of float32 and shape [N, ..., (y1,x1,y2,x2)]

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


def compute_iou(boxes1, boxes2) -> tf.Tensor:
    """Computes pairwise intersection-over-union between box collections.

    Arguments:

    - *boxes1*: Tensor of float32 and shape [N, ..., (y1,x1,y2,x2)]
    - *boxes2*: Tensor of float32 and shape [N, ..., (y1,x1,y2,x2)]

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
