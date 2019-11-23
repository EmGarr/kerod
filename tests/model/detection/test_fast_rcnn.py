from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K

from od.core.standard_fields import BoxField, LossField
from od.model.detection.fast_rcnn import FastRCNN


def mocked_random_shuffle(indices):
    """In the methods subsample_indicator a tf.random.shuffle is used we want it to return its
    input.
    """
    return indices


def test_fast_rcnn_full_inference_and_training():
    # args callable
    pyramid = [tf.zeros((2, shape, shape, 256)) for shape in [160, 80, 40, 20, 20]]
    boxes = [[0, 0, i, i] for i in range(1, 1000)]
    boxes = tf.constant([boxes, boxes], tf.float32)
    image_shape = (800., 800.)

    image_information = [[800., 700.], [700., 800.]]

    ground_truths = [{
        BoxField.BOXES: tf.constant([[0, 0, 1, 1], [0, 0, 2, 2]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1], [0, 1, 0]], tf.float32),
        BoxField.WEIGHTS: tf.constant([1, 0], tf.float32)
    }, {
        BoxField.BOXES: tf.constant([[0, 0, 3, 3]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1]], tf.float32)
    }]

    num_classes = 3
    fast_rcnn = FastRCNN(num_classes)

    fast_rcnn([pyramid, boxes, image_shape, image_information, ground_truths], training=True)
    fast_rcnn([pyramid, boxes, image_shape, image_information])


@mock.patch('tensorflow.random.shuffle')
def test_fast_rcnn_sample_boxes(mock_shuffle):
    # The mocking allows to make the test deterministic
    mock_shuffle.side_effect = mocked_random_shuffle

    boxes = [[0, 0, i, i] for i in range(1, 21)]
    boxes = tf.constant([boxes, boxes], tf.float32)

    num_classes = 3
    ground_truths = [{
        BoxField.BOXES: tf.constant([[0, 0, 1, 1], [0, 0, 2, 2]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1], [0, 1, 0]], tf.float32),
        BoxField.WEIGHTS: tf.constant([1, 0], tf.float32)
    }, {
        BoxField.BOXES: tf.constant([[0, 0, 3, 3]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1]], tf.float32)
    }]

    fast_rcnn = FastRCNN(num_classes)
    sampling_size = 10
    y_true, weights = fast_rcnn.sample_boxes(boxes,
                                             ground_truths,
                                             sampling_size=sampling_size,
                                             sampling_positive_ratio=0.2)

    expected_y_true_classification = np.zeros((2, sampling_size, 3))
    expected_y_true_classification[:, :, 0] = 1
    expected_y_true_classification[0, 0] = [0, 0, 1]
    expected_y_true_classification[1, 2] = [0, 0, 1]
    # boxes 4 does match with 3: iou([0, 0, 3, 3], [0, 0, 4, 4]) = 0.5625 > 0.5
    expected_y_true_classification[1, 3] = [0, 0, 1]

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


def test_fast_rcnn_sample_boxes_value_error():
    # The mocking allows to make the test deterministic
    boxes = [[0, 0, i, i] for i in range(1, 21)]
    boxes = tf.constant([boxes], tf.float32)

    num_classes = 3
    ground_truths = [{
        BoxField.BOXES: tf.constant([[0, 0, 1, 1], [0, 0, 2, 2]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1], [0, 1, 0]], tf.float32),
        BoxField.WEIGHTS: tf.constant([1, 0], tf.float32)
    }, {
        BoxField.BOXES: tf.constant([[0, 0, 3, 3]], tf.float32),
        BoxField.LABELS: tf.constant([[0, 0, 1]], tf.float32)
    }]

    fast_rcnn = FastRCNN(num_classes)
    # TODO allow tf.dunction to support it
    # with pytest.raises(ValueError):
    #     fast_rcnn.sample_boxes(boxes, ground_truths, sampling_size=10)
    with pytest.raises(ValueError):
        fast_rcnn.sample_boxes(boxes, ground_truths, sampling_size=100)


def test_fast_rcnn_compute_loss():
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

    y_true_cls = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0]], [[0, 0, 1], [0, 0, 1], [0, 1, 0]]])
    y_true_loc = tf.constant([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])

    y_true = {LossField.CLASSIFICATION: y_true_cls, LossField.LOCALIZATION: y_true_loc}
    weights = {
        LossField.CLASSIFICATION: tf.ones((2, 3), tf.float32),
        LossField.LOCALIZATION: tf.constant([[1, 0, 0], [0, 1, 1]], tf.float32)
    }
    num_classes = 3
    fast_rcnn = FastRCNN(num_classes)

    loss_cls, loss_loc = fast_rcnn.compute_loss(y_true, weights, classification_pred,
                                                localization_pred)

    assert loss_cls == 400 / 3 / 2
    assert loss_loc == 0.5
