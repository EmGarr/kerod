# Copyright 2017 The TensorFlow Authors modified by Emilien Garreau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
    loss_op = losses.SmoothL1Localization()
    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
    loss = tf.reduce_sum(loss)
    exp_loss = 7.695
    np.testing.assert_array_almost_equal(loss, exp_loss)


def test_binary_cross_entropy():
    y_pred = tf.constant([[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
                          [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]],
                         tf.float32)
    y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)
    sample_weight = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]], tf.float32)
    loss_op = losses.BinaryCrossentropy()
    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
    loss = tf.reduce_sum(loss)

    exp_loss = -2 * math.log(.5) / 3
    np.testing.assert_array_almost_equal(loss, exp_loss)


def test_binary_cross_entropy_anchor_wise():
    y_pred = tf.constant([[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
                          [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]],
                         tf.float32)
    y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)

    sample_weight = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]], tf.float32)

    loss_op = losses.BinaryCrossentropy()

    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)

    exp_loss = np.array([[0, 0, -math.log(.5) / 3, 0], [-math.log(.5) / 3, 0, 0, 0]])
    np.testing.assert_array_almost_equal(loss, exp_loss)


def test_categorical_cross_entropy():
    y_pred = tf.constant([[[-100, 100, -100], [100, -100, -100], [0, 0, -100], [-100, -100, 100]],
                          [[-100, 0, 0], [-100, 100, -100], [-100, 100, -100], [100, -100, -100]]],
                         tf.float32)
    y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]], tf.float32)
    sample_weight = tf.constant([[1, 1, 0.5, 1], [1, 1, 1, 0]], tf.float32)
    loss_op = losses.CategoricalCrossentropy()
    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
    loss = tf.reduce_sum(loss)

    exp_loss = -1.5 * math.log(.5)
    np.testing.assert_array_almost_equal(loss, exp_loss)


def test_categorical_cross_entropy_anchor_wise():
    y_pred = tf.constant([[[-100, 100, -100], [100, -100, -100], [0, 0, -100], [-100, -100, 100]],
                          [[-100, 0, 0], [-100, 100, -100], [-100, 100, -100], [100, -100, -100]]],
                         tf.float32)
    y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]], tf.float32)
    sample_weight = tf.constant([[1, 1, 0.5, 1], [1, 1, 1, 0]], tf.float32)
    loss_op = losses.CategoricalCrossentropy()
    loss = loss_op(y_true, y_pred, sample_weight=sample_weight)

    exp_loss = np.array([[0, 0, -0.5 * math.log(.5), 0], [-math.log(.5), 0, 0, 0]])

    np.testing.assert_array_almost_equal(loss, exp_loss)


if __name__ == '__main__':
    tf.test.main()
