# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for google3.research.vale.object_detection.losses."""

import math

import numpy as np
import tensorflow as tf

# from od.core import box_list
from od.core import losses
# from od.core import matcher
# import tensorflow.keras.losses as losses


class SmoothL1LocalizationTest(tf.test.TestCase):

    def testReturnsCorrectLoss(self):
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
        self.assertAllClose(loss, exp_loss)


class BinaryCrossentropyTest(tf.test.TestCase):

    def testReturnsCorrectLoss(self):
        y_pred = tf.constant(
            [[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
             [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]], tf.float32)
        y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                              [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)
        sample_weight = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]], tf.float32)
        loss_op = losses.BinaryCrossentropy()
        loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.reduce_sum(loss)

        exp_loss = -2 * math.log(.5) / 3
        self.assertAllClose(loss, exp_loss)

    def testReturnsCorrectAnchorWiseLoss(self):
        y_pred = tf.constant(
            [[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
             [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]], tf.float32)
        y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                              [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)

        sample_weight = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]], tf.float32)

        loss_op = losses.BinaryCrossentropy()

        loss = loss_op(y_true, y_pred, sample_weight=sample_weight)

        exp_loss = np.array([[0, 0, -math.log(.5) / 3, 0], [-math.log(.5) / 3, 0, 0, 0]])
        self.assertAllClose(loss, exp_loss)


# def _logit(probability):
#     return math.log(probability / (1. - probability))

# class SigmoidFocalClassificationLossTest(tf.test.TestCase):

#     def testEasyExamplesProduceSmallLossComparedToSigmoidXEntropy(self):
#         y_pred = tf.constant([[[_logit(0.97)], [_logit(0.91)], [_logit(0.73)], [_logit(0.27)],
#                                [_logit(0.09)], [_logit(0.03)]]], tf.float32)
#         y_true = tf.constant([[[1], [1], [1], [0], [0], [0]]], tf.float32)
#         sample_weight = tf.constant([[[1], [1], [1], [1], [1], [1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0, alpha=None)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                    axis=2)
#         sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                      axis=2)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             order_of_ratio = np.power(10, np.floor(np.log10(sigmoid_loss / focal_loss)))
#             self.assertAllClose(order_of_ratio, [[1000, 100, 10, 10, 100, 1000]])

#     def testHardExamplesProduceLossComparableToSigmoidXEntropy(self):
#         y_pred = tf.constant(
#             [[[_logit(0.55)], [_logit(0.52)], [_logit(0.50)], [_logit(0.48)], [_logit(0.45)]]],
#             tf.float32)
#         y_true = tf.constant([[[1], [1], [1], [0], [0]]], tf.float32)
#         sample_weight = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0, alpha=None)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                    axis=2)
#         sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                      axis=2)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             order_of_ratio = np.power(10, np.floor(np.log10(sigmoid_loss / focal_loss)))
#             self.assertAllClose(order_of_ratio, [[1., 1., 1., 1., 1.]])

#     def testNonAnchorWiseOutputComparableToSigmoidXEntropy(self):
#         y_pred = tf.constant(
#             [[[_logit(0.55)], [_logit(0.52)], [_logit(0.50)], [_logit(0.48)], [_logit(0.45)]]],
#             tf.float32)
#         y_true = tf.constant([[[1], [1], [1], [0], [0]]], tf.float32)
#         sample_weight = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0, alpha=None)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight))
#         sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight))

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             order_of_ratio = np.power(10, np.floor(np.log10(sigmoid_loss / focal_loss)))
#             self.assertAlmostEqual(order_of_ratio, 1.)

#     def testIgnoreNegativeExampleLossViaAlphaMultiplier(self):
#         y_pred = tf.constant(
#             [[[_logit(0.55)], [_logit(0.52)], [_logit(0.50)], [_logit(0.48)], [_logit(0.45)]]],
#             tf.float32)
#         y_true = tf.constant([[[1], [1], [1], [0], [0]]], tf.float32)
#         sample_weight = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0, alpha=1.0)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                    axis=2)
#         sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                      axis=2)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             self.assertAllClose(focal_loss[0][3:], [0., 0.])
#             order_of_ratio = np.power(10,
#                                       np.floor(np.log10(sigmoid_loss[0][:3] / focal_loss[0][:3])))
#             self.assertAllClose(order_of_ratio, [1., 1., 1.])

#     def testIgnorePositiveExampleLossViaAlphaMultiplier(self):
#         y_pred = tf.constant(
#             [[[_logit(0.55)], [_logit(0.52)], [_logit(0.50)], [_logit(0.48)], [_logit(0.45)]]],
#             tf.float32)
#         y_true = tf.constant([[[1], [1], [1], [0], [0]]], tf.float32)
#         sample_weight = tf.constant([[[1], [1], [1], [1], [1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(gamma=2.0, alpha=0.0)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                    axis=2)
#         sigmoid_loss = tf.reduce_sum(sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight),
#                                      axis=2)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             self.assertAllClose(focal_loss[0][:3], [0., 0., 0.])
#             order_of_ratio = np.power(10,
#                                       np.floor(np.log10(sigmoid_loss[0][3:] / focal_loss[0][3:])))
#             self.assertAllClose(order_of_ratio, [1., 1.])

#     def testSimilarToSigmoidXEntropyWithHalfAlphaAndZeroGammaUpToAScale(self):
#         y_pred = tf.constant(
#             [[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
#              [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]], tf.float32)
#         y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
#                               [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)
#         sample_weight = tf.constant([[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=0.5, gamma=0.0)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = focal_loss_op(y_pred, y_true, sample_weight=sample_weight)
#         sigmoid_loss = sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             self.assertAllClose(sigmoid_loss, focal_loss * 2)

#     def testSameAsSigmoidXEntropyWithNoAlphaAndZeroGamma(self):
#         y_pred = tf.constant(
#             [[[-100, 100, -100], [100, -100, -100], [100, 0, -100], [-100, -100, 100]],
#              [[-100, 0, 100], [-100, 100, -100], [100, 100, 100], [0, 0, -1]]], tf.float32)
#         y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
#                               [[0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 0, 0]]], tf.float32)
#         sample_weight = tf.constant([[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=None, gamma=0.0)
#         sigmoid_loss_op = losses.WeightedSigmoidClassificationLoss()
#         focal_loss = focal_loss_op(y_pred, y_true, sample_weight=sample_weight)
#         sigmoid_loss = sigmoid_loss_op(y_pred, y_true, sample_weight=sample_weight)

#         with self.test_session() as sess:
#             sigmoid_loss, focal_loss = sess.run([sigmoid_loss, focal_loss])
#             self.assertAllClose(sigmoid_loss, focal_loss)

#     def testExpectedLossWithAlphaOneAndZeroGamma(self):
#         # All zeros correspond to 0.5 probability.
#         y_pred = tf.constant([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], tf.float32)
#         y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
#                               [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]], tf.float32)
#         sample_weight = tf.constant([[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=1.0, gamma=0.0)

#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight))
#         with self.test_session() as sess:
#             focal_loss = sess.run(focal_loss)
#             self.assertAllClose(
#                 (
#                     -math.log(.5) *  # x-entropy per class per anchor
#                     1.0 *  # alpha
#                     8),  # positives from 8 anchors
#                 focal_loss)

#     def testExpectedLossWithAlpha75AndZeroGamma(self):
#         # All zeros correspond to 0.5 probability.
#         y_pred = tf.constant([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], tf.float32)
#         y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
#                               [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]]], tf.float32)
#         sample_weight = tf.constant([[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
#                                      [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]], tf.float32)
#         focal_loss_op = losses.SigmoidFocalClassificationLoss(alpha=0.75, gamma=0.0)

#         focal_loss = tf.reduce_sum(focal_loss_op(y_pred, y_true, sample_weight=sample_weight))
#         with self.test_session() as sess:
#             focal_loss = sess.run(focal_loss)
#             self.assertAllClose(
#                 (
#                     -math.log(.5) *  # x-entropy per class per anchor.
#                     ((
#                         0.75 *  # alpha for positives.
#                         8) +  # positives from 8 anchors.
#                      (
#                          0.25 *  # alpha for negatives.
#                          8 * 2))),  # negatives from 8 anchors for two classes.
#                 focal_loss)

#     def testExpectedLossWithLossesMask(self):
#         # All zeros correspond to 0.5 probability.
#         y_pred = tf.constant([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
#                               [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], tf.float32)
#         y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
#                               [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
#                               [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]], tf.float32)
#         sample_weight = tf.constant([[1, 1, 1, 1], [1, 1, 1, 1],[1, 1, 1, 1]], tf.float32)

#         focal_loss_op = losses.FocalBinaryCrossentropy(alpha=0.75, gamma=0.0)

#         focal_loss = tf.reduce_sum(
#             focal_loss_op(y_pred, y_true, sample_weight=sample_weight, losses_mask=losses_mask))
#         with self.test_session() as sess:
#             focal_loss = sess.run(focal_loss)
#             self.assertAllClose(
#                 (
#                     -math.log(.5) *  # x-entropy per class per anchor.
#                     ((
#                         0.75 *  # alpha for positives.
#                         8) +  # positives from 8 anchors.
#                      (
#                          0.25 *  # alpha for negatives.
#                          8 * 2))),  # negatives from 8 anchors for two classes.
#                 focal_loss)


class WeightedSoftmaxClassificationLossTest(tf.test.TestCase):

    def testReturnsCorrectLoss(self):
        y_pred = tf.constant(
            [[[-100, 100, -100], [100, -100, -100], [0, 0, -100], [-100, -100, 100]],
             [[-100, 0, 0], [-100, 100, -100], [-100, 100, -100], [100, -100, -100]]], tf.float32)
        y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                              [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]], tf.float32)
        sample_weight = tf.constant([[1, 1, 0.5, 1], [1, 1, 1, 0]], tf.float32)
        loss_op = losses.CategoricalCrossentropy()
        loss = loss_op(y_true, y_pred, sample_weight=sample_weight)
        loss = tf.reduce_sum(loss)

        exp_loss = -1.5 * math.log(.5)
        self.assertAllClose(loss, exp_loss)

    def testReturnsCorrectAnchorWiseLoss(self):
        y_pred = tf.constant(
            [[[-100, 100, -100], [100, -100, -100], [0, 0, -100], [-100, -100, 100]],
             [[-100, 0, 0], [-100, 100, -100], [-100, 100, -100], [100, -100, -100]]], tf.float32)
        y_true = tf.constant([[[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1]],
                              [[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0]]], tf.float32)
        sample_weight = tf.constant([[1, 1, 0.5, 1], [1, 1, 1, 0]], tf.float32)
        loss_op = losses.CategoricalCrossentropy()
        loss = loss_op(y_true, y_pred, sample_weight=sample_weight)

        exp_loss = np.array([[0, 0, -0.5 * math.log(.5), 0], [-math.log(.5), 0, 0, 0]])
        self.assertAllClose(loss, exp_loss)


if __name__ == '__main__':
    tf.test.main()
