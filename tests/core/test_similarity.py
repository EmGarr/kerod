import numpy as np
import tensorflow as tf
from kerod.core.box_ops import convert_to_center_coordinates
from kerod.core.similarity import DetrSimilarity, IoUSimilarity
from kerod.core.standard_fields import BoxField


def test_iou_similarity():
    boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])
    boxes2 = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]],
                          [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]])
    exp_output = [[[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                  [[2.0 / 16.0, 0, 6.0 / 400.0], [0, 0, 0]]]
    iou_output = IoUSimilarity()({BoxField.BOXES: boxes1}, {BoxField.BOXES: boxes2})
    np.testing.assert_allclose(iou_output, exp_output)


def test_detr_similarity():
    boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])
    boxes1 = convert_to_center_coordinates(boxes1)

    boxes2 = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]],
                          [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0], [0.0, 0.0, 20.0, 20.0]]])
    boxes2 = convert_to_center_coordinates(boxes2)
    ground_truths = {
        BoxField.BOXES: boxes1,
        BoxField.LABELS: tf.constant([[1, 0], [1, 0]], tf.int32),
        BoxField.WEIGHTS: tf.constant([[1, 0], [1, 1]], tf.float32),
        BoxField.NUM_BOXES: tf.constant([[2], [1]], tf.int32)
    }
    classification_logits = tf.constant(
        [[[0, 100, 0], [0, 100, 0], [0, 100, 0]], [[100, 0, 0], [0, -100, 0], [0, -100, 0]]],
        tf.float32)

    inputs2 = {BoxField.BOXES: boxes2, BoxField.SCORES: classification_logits}

    similarity = DetrSimilarity()(ground_truths, inputs2)
    # Taken from test_box_ops.py::test_compute_giou_3d_tensor
    exp_iou = np.array([[[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                        [[2.0 / 16.0, 0, 6.0 / 400.0], [0, 0, 0]]])
    exp_term2 = np.array([[[4. / 20, 125 / 132, 0.], [12 / 28., 84 / 90., 0.]],
                          [[4. / 20, 125. / 132, 0.], [36. / 48, 224. / 225, 0.]]])
    exp_giou = -(exp_iou - exp_term2)

    exp_cost_class = np.array([[[-1., -1., -1.], [0., 0., 0.]], [[0., 0., -0.], [-1., -0.5, -0.5]]])

    exp_cost_bbox = np.array([[[5., 22.5, 45.5], [8.5, 19., 40.]],
                              [[5., 22.5, 45.5], [17.5, 31., 60.]]])

    exp_similarity = 2 * exp_giou + exp_cost_class + 5 * exp_cost_bbox

    np.testing.assert_allclose(similarity, exp_similarity)
