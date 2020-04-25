from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from kerod.core.standard_fields import BoxField, LossField
from kerod.model.detection.fast_rcnn import FastRCNN, compute_fast_rcnn_metrics


def mocked_random_shuffle(indices):
    """In the methods subsample_indicator a tf.random.shuffle is used we want it to return its
    input.
    """
    return indices


class FastRCNNGraphSupport(FastRCNN):

    @tf.function
    def call(self, inputs):
        return super().call(inputs)


@pytest.mark.parametrize("fast_rcnn_class", [FastRCNN, FastRCNNGraphSupport])
def test_fast_rcnn_full_inference_and_training(fast_rcnn_class):
    # args callable
    pyramid = [tf.zeros((2, shape, shape, 256)) for shape in [160, 80, 40, 20, 20]]
    boxes = [[0, 0, i, i] for i in range(1, 1000)]
    boxes = tf.constant([boxes, boxes], tf.float32)
    num_classes = 3
    fast_rcnn = fast_rcnn_class(num_classes)

    fast_rcnn([pyramid, boxes])


@mock.patch('tensorflow.random.shuffle')
def test_fast_rcnn_sample_boxes(mock_shuffle):
    # The mocking allows to make the test deterministic
    mock_shuffle.side_effect = mocked_random_shuffle

    boxes = [[0, 0, i, i] for i in range(1, 21)]
    boxes = tf.constant([boxes, boxes], tf.float32)

    num_classes = 3
    ground_truths = {
        BoxField.BOXES:
            tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
        BoxField.LABELS:
            tf.constant([[1, 0], [1, 0]], tf.int32),
        BoxField.WEIGHTS:
            tf.constant([[1, 0], [1, 1]], tf.float32),
        BoxField.NUM_BOXES:
            tf.constant([[2], [1]], tf.int32)
    }

    fast_rcnn = FastRCNN(num_classes)
    sampling_size = 10
    y_true, weights, sample_anchors = fast_rcnn.sample_boxes(boxes,
                                                             ground_truths,
                                                             sampling_size=sampling_size,
                                                             sampling_positive_ratio=0.2)

    expected_y_true_classification = np.zeros((2, sampling_size))
    expected_y_true_classification[0, 0] = 2
    expected_y_true_classification[1, 2] = 2
    # boxes 4 does match with 3: iou([0, 0, 3, 3], [0, 0, 4, 4]) = 0.5625 > 0.5
    expected_y_true_classification[1, 3] = 2

    expected_y_true_localization = np.zeros((2, sampling_size, 4))

    # boxe num 4 does match with num 3: iou([0, 0, 3, 3], [0, 0, 4, 4]) = 0.5625 > 0.5
    # You can recompute the encoding with this methods
    # encode_boxes_faster_rcnn(tf.constant([0, 0, 3, 3], tf.float32), tf.constant([0, 0, 4, 4], tf.float32))
    expected_y_true_localization[1, 3] = [-0.125, -0.125, -0.2876821, -0.2876821]

    expected_weights_classification = np.ones((2, sampling_size))
    expected_weights_localization = np.zeros((2, sampling_size))
    expected_weights_localization[0, 0] = 1
    expected_weights_localization[1, 2] = 1
    expected_weights_localization[1, 3] = 1

    np.testing.assert_array_equal(expected_y_true_classification, y_true[LossField.CLASSIFICATION])
    np.testing.assert_array_almost_equal(expected_y_true_localization,
                                         y_true[LossField.LOCALIZATION])
    np.testing.assert_array_equal(expected_weights_classification,
                                  weights[LossField.CLASSIFICATION])
    np.testing.assert_array_equal(expected_weights_localization, weights[LossField.LOCALIZATION])

    # In this batch [0, 0, 2, 2] isn't sampled because of the ground_truths weights
    # On the ground_truths [0, 0, 2, 2] the weights has been set to 0
    expected_sample_anchors_batch_1 = [[0., 0., 1., 1.], [0., 0., 3., 3.], [0., 0., 4., 4.],
                                       [0., 0., 5., 5.], [0., 0., 6., 6.], [0., 0., 7., 7.],
                                       [0., 0., 8., 8.], [0., 0., 9., 9.], [0., 0., 10., 10.],
                                       [0., 0., 11., 11.]]
    expected_sample_anchors_batch_2 = [[0., 0., 1., 1.], [0., 0., 2., 2.], [0., 0., 3., 3.],
                                       [0., 0., 4., 4.], [0., 0., 5., 5.], [0., 0., 6., 6.],
                                       [0., 0., 7., 7.], [0., 0., 8., 8.], [0., 0., 9., 9.],
                                       [0., 0., 10., 10.]]
    expected_sample_anchors = np.array(
        [expected_sample_anchors_batch_1, expected_sample_anchors_batch_2])
    np.testing.assert_array_equal(expected_sample_anchors, sample_anchors)


# We are forced to Mock the add_metric, add_loss because want it to be used inside the call
# You can see it works automatically in test_fast_rcnn
@mock.patch('kerod.model.detection.fast_rcnn.FastRCNN.add_metric', spec=True, return_value=None)
@mock.patch('kerod.model.detection.fast_rcnn.FastRCNN.add_loss', spec=True, return_value=None)
def test_fast_rcnn_compute_loss(mock_add_loss, mock_add_metric):
    localization_pred = tf.constant([
        [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2, 2, 2],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ],
    ], tf.float32)

    classification_pred = tf.constant([[[-100, 100, -100], [100, -100, -100], [100, -100, -100]],
                                       [[100, -100, -100], [-100, -100, 100], [-100, -100, 100]]],
                                      tf.float32)

    y_true_cls = tf.constant([[1, 0, 0], [2, 2, 1]])
    y_true_loc = tf.constant([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    y_true = {LossField.CLASSIFICATION: y_true_cls, LossField.LOCALIZATION: y_true_loc}
    weights = {
        LossField.CLASSIFICATION: tf.ones((2, 3), tf.float32),
        LossField.LOCALIZATION: tf.constant([[1, 0, 0], [0, 1, 1]], tf.float32)
    }
    num_classes = 3
    fast_rcnn = FastRCNN(num_classes)

    losses = fast_rcnn.compute_loss(y_true, weights, classification_pred, localization_pred)

    assert losses[LossField.CLASSIFICATION] == 400 / 3 / 2
    assert losses[LossField.LOCALIZATION] == 0.5
    assert len(losses) == 2


def test_compute_fast_rcnn_metrics():
    y_pred = tf.constant(
        [
            [[-100, 100, -100], [100, -100, -100], [100, -100, -100]],
            # TP foreground   - TP background    -  TP background
            [[100, -100, -100], [-100, -100, 100], [-100, -100, 100]]
        ],
        tf.float32)
    # FP background & FN predicted - TP foreground   - FP

    y_true = tf.constant([[1, 0, 0], [2, 2, 1]])

    accuracy, fg_accuracy, false_negative = compute_fast_rcnn_metrics(y_true, y_pred)
    assert accuracy == 4 / 6
    assert fg_accuracy == 0.5
    assert false_negative == 1 / 4
