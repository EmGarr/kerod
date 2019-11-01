import pytest
import tensorflow as tf
import numpy as np
from od.model.detection.fast_rcnn import assign_pyramid_level_to_boxes, match_boxes_to_their_pyramid_level, multilevel_roi_align


@pytest.mark.parametrize("level_target,max_level,expected_target",
                         [[2, 4, [0, 0, 0, 3, 2, 3, 1]], [2, 7, [0, 0, 0, 6, 2, 3, 1]]])
def test_assign_pyramid_level_to_boxes(level_target, max_level, expected_target):
    boxes = tf.constant([[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 10000, 10000],
                         [0, 0, 250, 250], [0, 0, 500, 500], [0, 0, 112, 112]],
                        dtype=tf.float32)
    pyramid_level = assign_pyramid_level_to_boxes(boxes, max_level, level_target=level_target)
    assert np.array_equal(pyramid_level.numpy(), np.array(expected_target, dtype=np.int32))


def test_match_boxes_to_their_pyramid_level():
    boxes = tf.constant([[[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 1000, 1000],
                          [0, 0, 250, 250], [0, 0, 500, 500], [0, 0, 112, 112]],
                         [[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 10000, 10000],
                          [0, 0, 250, 250], [0, 0, 500, 500], [0, 0, 112, 112]]],
                        dtype=tf.float32)
    exp_boxes_per_level = [
        [[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 100, 100], [0, 0, 1, 1],
         [0, 0, 10, 10]],
        [[0, 0, 112, 112], [0, 0, 112, 112]],
        [[0, 0, 250, 250], [0, 0, 250, 250]],
        [[0, 0, 1000, 1000], [0, 0, 500, 500], [0, 0, 10000, 10000], [0, 0, 500, 500]],
    ]
    exp_indices_per_level = [[0, 0, 0, 1, 1, 1], [0, 1], [0, 1], [0, 0, 1, 1]]
    exp_original_pos_per_level = [[0, 1, 2, 7, 8, 9], [6, 13], [4, 11], [3, 5, 10, 12]]

    boxes_plvl, box_indices_plvl, original_pos_plvl = match_boxes_to_their_pyramid_level(boxes, 4)

    for exp, out in zip(exp_boxes_per_level, boxes_plvl):
        assert np.array_equal(np.array(exp, dtype=np.float32), out.numpy())

    for exp, out in zip(exp_indices_per_level, box_indices_plvl):
        assert np.array_equal(np.array(exp, dtype=np.float32), out.numpy())

    for exp, out in zip(exp_original_pos_per_level, original_pos_plvl):
        assert np.array_equal(np.array(exp, dtype=np.float32), out.numpy())


def test_multilevel_roi_align():
    boxes = tf.constant([[[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 1000, 1000],
                          [0, 0, 250, 250], [0, 0, 500, 500], [0, 0, 112, 112]],
                         [[0, 0, 100, 100], [0, 0, 1, 1], [0, 0, 10, 10], [0, 0, 1000, 1000],
                          [0, 0, 250, 250], [0, 0, 500, 500], [0, 0, 112, 112]]],
                        dtype=tf.float32)
    inputs_lvl1 = np.ones((2, 100, 100, 1))
    inputs_lvl1[1] *= -1
    inputs_lvl2 = np.ones((2, 50, 50, 1)) * 2
    inputs_lvl2[1] *= -1
    inputs_lvl3 = np.ones((2, 25, 25, 1)) * 3
    inputs_lvl3[1] *= -1
    inputs_lvl4 = np.ones((2, 50, 50, 1)) * 4
    inputs_lvl4[1] *= -1
    output = multilevel_roi_align([inputs_lvl1, inputs_lvl2, inputs_lvl3, inputs_lvl4], boxes,
                                  (1000, 1000), 7)

    # expected box position per level [[0, 1, 2, 7, 8, 9], [6, 13], [4, 11], [3, 5, 10, 12]]
    tensor = np.ones((14, 7, 7, 1), dtype=np.float32)
    tensor[7:] *= -1  # tensor from the pos 2 in the batch
    tensor[6] *= 2
    tensor[13] *= 2
    tensor[4] *= 3
    tensor[11] *= 3
    tensor[3] *= 4
    tensor[5] *= 4
    tensor[10] *= 4
    tensor[12] *= 4

    np.testing.assert_array_equal(output.numpy(), tensor)
