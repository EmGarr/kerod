import math

import numpy as np
import tensorflow as tf

from od.core import losses


def test_smooth_L1_loss():
    batch_size = 2
    num_anchors = 3
    code_size = 4
    y_pred = tf.constant([[[2.5, 0, .4, 0], [0, 0, 0, 0], [0, 2.5, 0, .4]],
                          [[3.5, 0, 0, 0], [0, .4, 0, .9], [0, 0, 1.5, 0]]], tf.float32)
    y_true = tf.zeros([batch_size, num_anchors, code_size])
    sample_weight = tf.constant([[2, 1, 1], [0, 3, 0]], tf.float32)
    loss_op = losses.SmoothL1Localization(delta=1.0)
    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
    loss = tf.reduce_sum(loss)
    exp_loss = 7.695
    np.testing.assert_array_almost_equal(loss, exp_loss)
