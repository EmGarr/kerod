import tensorflow as tf
import numpy as np
from od.core.anchor_generator import generate_anchors


def get_all_anchors(stride, sizes, ratios, max_size):
    """This function has been taken from tensorpack. https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_rpn.py#L158

    Get all anchors in the largest possible image, shifted, floatbox

    Arguments:

    - stride (int): the stride of anchors.
    - sizes (tuple[int]): the sizes (sqrt area) of anchors
    - ratios (tuple[int]): the aspect ratios of anchors
    - max_size (int): maximum size of input image
    
    Returns:

    anchors: SxSxNUM_ANCHORx4, where S == ceil(MAX_SIZE/STRIDE), floatbox
        The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.

    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on 0, have sqrt areas equal to the specified sizes, and aspect ratios as given.
    anchors = []
    for sz in sizes:
        for ratio in ratios:
            w = np.sqrt(sz * sz / ratio)
            h = ratio * w
            anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5

    field_size = int(np.ceil(max_size / stride))
    shifts = (np.arange(0, field_size) * stride).astype("float32")
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()

    # Kx4, K = field_size * field_size
    K = shifts.shape[0]

    A = cell_anchors.shape[0]
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose(
        (1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype("float32")
    return field_of_anchors


def test_generate_anchors():
    """Ensure the anchor generation is the same than tensorpack"""
    stride = 16
    ratios = [0.5, 1, 2]
    scales = [8, 16, 32]
    feature_map_shape = 8
    anchors_tensorpack = get_all_anchors(stride, scales, ratios,
                                         stride * feature_map_shape).reshape((1, -1, 4))

    anchors_od = generate_anchors(stride, tf.constant(scales, tf.float32),
                                  tf.constant(ratios, tf.float32),
                                  (1, feature_map_shape, feature_map_shape, None))

    anchors_od = tf.gather(anchors_od, [1, 0, 3, 2], axis=-1)
    np.testing.assert_array_almost_equal(anchors_tensorpack, anchors_od, decimal=5)
