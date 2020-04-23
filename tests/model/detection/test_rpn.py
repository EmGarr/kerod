from unittest import mock

import pytest
import tensorflow as tf

from kerod.core.standard_fields import BoxField, LossField
from kerod.model.detection.rpn import RegionProposalNetwork, compute_rpn_metrics


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
    features = [tf.zeros((2, shape, shape, 256)) for shape in [160, 80, 40, 20]]

    localization_pred, logit_scores, anchors = rpn(features)

    num_anchors = (160**2 + 80**2 + 40**2 + 20**2) * 3
    assert (2, num_anchors, 4) == localization_pred.shape
    assert (2, num_anchors, 2) == logit_scores.shape
    assert (num_anchors, 4) == anchors.shape


# We are forced to Mock the add_metric and add_loss because Keras want it to be used inside the call
# You can see it works automatically in test_rpn
@mock.patch('kerod.model.detection.rpn.RegionProposalNetwork.add_metric', spec=True, return_value=None)
@mock.patch('kerod.model.detection.rpn.RegionProposalNetwork.add_loss', spec=True, return_value=None)
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
            tf.constant([[2], [1]], tf.int32),
    }
    rpn = RegionProposalNetwork(classification_loss_weight=1.0)
    losses = rpn.compute_loss(localization_pred, classification_pred, anchors, ground_truths)

    assert losses[LossField.CLASSIFICATION] == 100
    assert losses[LossField.LOCALIZATION] == 0
    assert len(losses) == 2


def test_compute_rpn_metrics():
    y_true = tf.constant([[0, 0, 0, 1, 1, 0, 1, 0, 0]], tf.float32)
    weights = tf.constant([[0, 1, 1, 2, 1, 1, 0.5, 1, 1]])

    y_pred = tf.constant([[
        [-100, 100],
        [100, -100],
        [100, -100],
        [-100, 100],
        [-100, 100],
        [-100, 100],
        [100, -100],
        [-100, 100],
        [-100, 100],
    ]], tf.float32)
    recall = compute_rpn_metrics(y_true, y_pred, weights)
    assert recall == 2 / 3
