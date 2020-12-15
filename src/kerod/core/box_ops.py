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

    - *normalized_boxes*:  A Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]. These
    boxes have already been normalized in the feature space. The coordinates are not in
    the input image space.
    - *tensor_shape*: Height and width respectively
    - *crop_shape*:

    Returns:

    A tensor of shape [N, ..., num_boxes, (y_min, x_min, y_max, x_max)]
    """
    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=-1)

    spacing_h = (y_max - y_min) / tf.cast(crop_shape[0], boxes.dtype)
    spacing_w = (x_max - x_min) / tf.cast(crop_shape[1], boxes.dtype)

    tensor_shape = (tf.cast(tensor_shape[0] - 1,
                            boxes.dtype), tf.cast(tensor_shape[1] - 1, boxes.dtype))

    ny0 = (y_min + spacing_h / 2 - 0.5) / tensor_shape[0]
    nx0 = (x_min + spacing_w / 2 - 0.5) / tensor_shape[1]

    nh = spacing_h * tf.cast(crop_shape[0] - 1, boxes.dtype) / tensor_shape[0]
    nw = spacing_w * tf.cast(crop_shape[1] - 1, boxes.dtype) / tensor_shape[1]

    return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=-1)


def convert_to_center_coordinates(boxes: tf.Tensor) -> tf.Tensor:
    """Convert boxes to their center coordinates

    y_min, x_min, y_max, x_max -> y_cent, x_cent, h, w

    Arguments:

    - *boxes*: A Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]

    Returns:

    A tensor of shape [N, ..., num_boxes, (ycenter, xcenter, height, width)]
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

    - *boxes*: Tensor of shape [N, ..., (y_min,x_min,y_max_,x_max)]

    Returns:

    A tensor of shape [N, ..., num_boxes]
    """
    with tf.name_scope('Area'):
        y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
        return tf.squeeze((y_max - y_min) * (x_max - x_min), -1)


def compute_intersection(boxes1: tf.Tensor, boxes2: tf.Tensor, perm=None) -> tf.Tensor:
    """Compute pairwise intersection areas between boxes.

    Arguments:

    - *boxes1*: Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *boxes2*: Tensor of shape [N, ..., (y_max,x_max,y_max,x_max)]

    Returns:

    A tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope('Intersection'):
        y_min1, x_min1, y_max1, x_max1 = tf.split(value=boxes1, num_or_size_splits=4, axis=-1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(value=boxes2, num_or_size_splits=4, axis=-1)
        all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2, perm=perm))
        all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2, perm=perm))
        zero = tf.convert_to_tensor(0.0, boxes1.dtype)
        intersect_heights = tf.maximum(zero, all_pairs_min_ymax - all_pairs_max_ymin)
        all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2, perm=perm))
        all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2, perm=perm))
        intersect_widths = tf.maximum(zero, all_pairs_min_xmax - all_pairs_max_xmin)
        return intersect_heights * intersect_widths


def compute_iou(boxes1: tf.Tensor, boxes2: tf.Tensor) -> tf.Tensor:
    """Computes pairwise intersection-over-union between boxes.

    Example:

   The axis x correspond to boxes2 and y the boxes1:

    ```python
    from kerod.core.box_ops import compute_iou
    import numpy as np

    boxes1 = np.array([[548.26666, 364.57202, 706.1333 , 524.472  ],
           [473.6    , 547.924  , 565.3333 , 635.336  ],
           [477.86664, 688.63605, 580.26666, 786.70795],
           [497.06668, 750.464  , 576.     , 857.064  ]])

    boxes2 = np.array([[474.74518, 553.37256, 565.2548 , 598.62744],
                     [448., 736., 576., 864.],
                      [464., 672., 592., 800.],
                      [560., 368., 688., 496.]
                     ])
    compute_iou(boxes1, boxes2)
    ```

    output

    ```
    <tf.Tensor: shape=(4, 4), dtype=float64, numpy=
    array([[0.        , 0.        , 0.        , 0.6490545 ],
           [0.51081317, 0.        , 0.        , 0.        ],
           [0.        , 0.23198337, 0.61294949, 0.        ],
           [0.        , 0.51356762, 0.18718853, 0.        ]])>
    ```

    Arguments:

    - *boxes1*: A 2D or 3D Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *boxes2*: A 2D or 3D Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]

    Returns:

    A tensor with shape [N, M] representing pairwise iou scores.

    Raises:

    ValueError: If your tensor is different than 2D or 3D.
    """
    return compute_giou(boxes1, boxes2, mode='iou')


def compute_giou(boxes1: tf.Tensor, boxes2: tf.Tensor, mode: str = "giou") -> tf.Tensor:
    """Computes pairwise general intersection-over-union between boxes following:
    https://giou.stanford.edu/

    Arguments:

    - *boxes1*: A 2D or 3D Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *boxes2*: A 2D or 3D Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *mode*: You can select iou or giou.

    Returns:

    A tensor with shape [N, M] representing pairwise iou scores.

    Raises:

    ValueError: If your tensor is different than 2D or 3D.
    """
    with tf.name_scope(mode.upper()):
        if len(boxes1.shape) == 2:
            perm = None
            which_dim_expands = 0
        elif len(boxes1.shape) == 3:
            perm = (0, 2, 1)
            which_dim_expands = 1
        else:
            raise ValueError('Compute Iou is only suppoted for 2D and 3D Tensor')

        intersections = compute_intersection(boxes1, boxes2, perm=perm)
        areas1 = compute_area(boxes1)
        areas2 = compute_area(boxes2)
        unions = areas1[..., None] + tf.expand_dims(areas2, which_dim_expands) - intersections
        iou = tf.where(intersections == 0, tf.zeros_like(intersections),
                       tf.truediv(intersections, unions))

        if mode == "iou":
            return iou

        y_min1, x_min1, y_max1, x_max1 = tf.split(boxes1, 4, axis=-1)
        y_min2, x_min2, y_max2, x_max2 = tf.split(boxes2, 4, axis=-1)
        enclose_ymin = tf.minimum(y_min1, tf.transpose(y_min2, perm=perm))
        enclose_xmin = tf.minimum(x_min1, tf.transpose(x_min2, perm=perm))
        enclose_ymax = tf.maximum(y_max1, tf.transpose(y_max2, perm=perm))
        enclose_xmax = tf.maximum(x_max1, tf.transpose(x_max2, perm=perm))

        zero = tf.convert_to_tensor(0.0, boxes1.dtype)
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        giou = iou - tf.math.divide_no_nan((enclose_area - unions), enclose_area)
        return giou


def normalize_box_coordinates(boxes, height: int, width: int):
    """ Normalize the boxes coordinates with image shape

    Arguments:

    - *boxes*: Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)]
    - *height*: An integer
    - *width*: An integer
    """

    y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
    y_min = y_min / height
    x_min = x_min / width
    y_max = y_max / height
    x_max = x_max / width

    # Won't be backpropagated to rois anyway, but to save time
    boxes = tf.stop_gradient(tf.concat([y_min, x_min, y_max, x_max], axis=-1))
    return boxes


def clip_boxes(boxes: tf.Tensor, window: tf.Tensor) -> tf.Tensor:
    """Perform a clipping according to a window on the boxes.

    Arguments:

    - *boxes*: A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
    - *window*: A tensor of shape [batch_size, h, w]

    Returns:

    A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
    """
    boxes = tf.maximum(boxes, tf.cast(0, boxes.dtype))
    m = tf.tile(tf.expand_dims(window, axis=1), [1, 1, 2])
    boxes = tf.minimum(boxes, tf.cast(m, boxes.dtype))
    return boxes


def flip_left_right(boxes: tf.Tensor) -> tf.Tensor:
    """[Taken from tensorflow models] Left-right flip the boxes.

    Arguments:

    - *boxes*: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
        Boxes are in normalized form meaning their coordinates vary
        between [0, 1]. Each row is in the form of [ymin, xmin, ymax, xmax].

    Return:

    Flipped boxes.
    """
    ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], 1)
    return flipped_boxes
