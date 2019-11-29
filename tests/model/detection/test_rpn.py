from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import backend as K

from od.core.standard_fields import BoxField
from od.model.detection.rpn import RegionProposalNetwork


def mocked_random_shuffle(indices):
    """In the methods subsample_indicator a tf.random.shuffle is used we want it to return its
    input.
    """
    return indices


class RegionProposalNetworkGraphSupport(RegionProposalNetwork):

    @tf.function
    def call(self, inputs, training=None):
        return super().call(inputs, training=training)


@pytest.mark.parametrize("rpn_class", [RegionProposalNetwork, RegionProposalNetworkGraphSupport])
def test_rpn(rpn_class):
    rpn = rpn_class()
    image_information = tf.constant([[200, 200], [200, 200]])
    features = [tf.zeros((2, shape, shape, 256)) for shape in [160, 80, 40, 20]]
    ground_truths = {
        BoxField.BOXES:
            tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
        BoxField.LABELS:
            tf.constant([[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]], tf.float32),
        BoxField.NUM_BOXES:
            tf.constant([2, 1], tf.int32),
    }
    boxes, scores = rpn([features, image_information, ground_truths], training=True)
    assert (2, 2000, 4) == boxes.shape
    assert (2, 2000) == scores.shape
    boxes, scores = rpn([features, image_information])
    assert (2, 1000, 4) == boxes.shape
    assert (2, 1000) == scores.shape

# We are forced to Mock the add_metric and add_loss because Keras want it to be used inside the call
# You can see it works automatically in test_rpn 
@mock.patch('od.model.detection.rpn.RegionProposalNetwork.add_metric', spec=True, return_value=None)
@mock.patch('od.model.detection.rpn.RegionProposalNetwork.add_loss', spec=True, return_value=None)
@mock.patch('tensorflow.random.shuffle', side_effect=mocked_random_shuffle)
def test_compute_loss_rpn(mock_add_metric, mock_add_loss, mock_shuffle):
    localization_pred = tf.constant(
        [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
        tf.float32)

    classification_pred = tf.constant(
        [[[-100, 100], [100, -100], [100, -100]], [[-100, 100], [-100, 100], [-100, 100]]],
        tf.float32)

    anchors = tf.constant([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], tf.float32)

    ground_truths = {
        BoxField.BOXES:
            tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
        BoxField.NUM_BOXES:
            tf.constant([2, 1], tf.int32),
    }
    rpn = RegionProposalNetwork(classification_loss_weight=1.0)
    classification_loss, localization_loss = rpn.compute_loss(localization_pred,
                                                              classification_pred, anchors,
                                                              ground_truths)

    assert localization_loss == 0
    assert classification_loss == 100
