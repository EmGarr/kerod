from unittest import mock

import numpy as np
import tensorflow as tf

from od.core.standard_fields import BoxField
from od.model.detection.rpn import RegionProposalNetwork


def mocked_random_shuffle(indices):
    """In the methods subsample_indicator a tf.random.shuffle is used we want it to return its
    input.
    """
    return indices


def test_rpn():
    rpn = RegionProposalNetwork()
    features = [tf.zeros((2, shape, shape, 256)) for shape in [160, 80, 40, 20]]
    boxes = np.array([[-3.5, -3.5, 3.5, 3.5]])
    labels = np.array([1])
    ground_truths = [{
        BoxField.BOXES: boxes,
        BoxField.LABELS: labels
    }, {
        BoxField.BOXES: boxes,
        BoxField.LABELS: labels
    }]
    rpn([features, ground_truths], training=True)
    rpn([features])


@mock.patch('tensorflow.random.shuffle')
def test_compute_loss_rpn(mock_shuffle):
    # The mocking allows to make the test deterministic
    mock_shuffle.side_effect = mocked_random_shuffle
    localization_pred = tf.constant(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
        tf.float32)

    classification_pred = tf.constant(
        [[[-100, 100], [100, -100], [100, -100]], [[-100, 100], [-100, 100], [-100, 100]]],
        tf.float32)

    anchors = tf.constant([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], tf.float32)

    ground_truths = [{
        BoxField.BOXES: tf.constant([[0, 0, 1, 1], [0, 0, 2, 2]], tf.float32),
    }, {
        BoxField.BOXES: tf.constant([[0, 0, 3, 3]], tf.float32),
    }]

    rpn = RegionProposalNetwork(classification_loss_weight=1.0, localization_loss_weight=2.0)
    classification_loss, localization_loss = rpn.compute_loss(localization_pred,
                                                              classification_pred, anchors,
                                                              ground_truths)

    assert localization_loss == 0
    assert classification_loss == 100
