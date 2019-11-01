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
"""Tests for object_detection.core.target_assigner."""
import numpy as np
import tensorflow as tf

from od.core.standard_fields import BoxField
from od.core import target_assigner as targetassigner
from od.core.box_ops import compute_iou
from od.core import argmax_matcher
from od.core.standard_fields import LossField


def encode_mean_stddev(boxes, anchors, stddev=0.1):
    return (boxes - anchors) / stddev


class TargetAssignerTest(tf.test.TestCase):

    def test_assign_agnostic(self):

        def graph_fn(anchor_means, groundtruth_box_corners):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
            box_coder = encode_mean_stddev
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {BoxField.BOXES: groundtruth_box_corners}
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=None)
            (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        anchor_means = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0]],
                                dtype=np.float32)
        groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]],
                                           dtype=np.float32)
        exp_cls_targets = [[1], [1], [0]]
        exp_cls_weights = [[1], [1], [1]]
        exp_reg_targets = [[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]]
        exp_reg_weights = [1, 1, 0]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)
        self.assertEqual(cls_targets_out.dtype, np.float32)
        self.assertEqual(cls_weights_out.dtype, np.float32)
        self.assertEqual(reg_targets_out.dtype, np.float32)
        self.assertEqual(reg_weights_out.dtype, np.float32)

    def test_assign_class_agnostic_with_ignored_matches(self):
        # Note: test is very similar to above. The third box matched with an IOU
        # of 0.35, which is between the matched and unmatched threshold. This means
        # That like above the expected classification targets are [1, 1, 0].
        # Unlike above, the third target is ignored and therefore expected
        # classification weights are [1, 1, 0].
        def graph_fn(anchor_means, groundtruth_box_corners):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.3)
            box_coder = encode_mean_stddev
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {BoxField.BOXES: groundtruth_box_corners}
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=None)
            (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        anchor_means = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0.0, 0.5, .9, 1.0]],
                                dtype=np.float32)
        groundtruth_box_corners = np.array([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9]],
                                           dtype=np.float32)
        exp_cls_targets = [[1], [1], [0]]
        exp_cls_weights = [[1], [1], [0]]
        exp_reg_targets = [[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0]]
        exp_reg_weights = [1, 1, 0]
        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)
        self.assertEqual(cls_targets_out.dtype, np.float32)
        self.assertEqual(cls_weights_out.dtype, np.float32)
        self.assertEqual(reg_targets_out.dtype, np.float32)
        self.assertEqual(reg_weights_out.dtype, np.float32)

    def test_assign_multiclass(self):

        def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
            box_coder = encode_mean_stddev
            unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {
                BoxField.BOXES: groundtruth_box_corners,
                BoxField.LABELS: groundtruth_labels
            }
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=unmatched_class_label)
            (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        anchor_means = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]],
            dtype=np.float32)
        groundtruth_box_corners = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9], [.75, 0, .95, .27]], dtype=np.float32)
        groundtruth_labels = np.array(
            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)

        exp_cls_targets = [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]]
        exp_cls_weights = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1]]
        exp_reg_targets = [[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0], [0, 0, -.5, .2]]
        exp_reg_weights = [1, 1, 0, 1]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)
        self.assertEqual(cls_targets_out.dtype, np.float32)
        self.assertEqual(cls_weights_out.dtype, np.float32)
        self.assertEqual(reg_targets_out.dtype, np.float32)
        self.assertEqual(reg_weights_out.dtype, np.float32)

    def test_assign_multiclass_with_groundtruth_weights(self):

        def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels,
                     groundtruth_weights):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
            box_coder = encode_mean_stddev
            unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {
                BoxField.BOXES: groundtruth_box_corners,
                BoxField.LABELS: groundtruth_labels,
                BoxField.WEIGHTS: groundtruth_weights
            }
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=unmatched_class_label)
            (_, cls_weights, _, reg_weights, _) = result
            return (cls_weights, reg_weights)

        anchor_means = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]],
            dtype=np.float32)
        groundtruth_box_corners = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9], [.75, 0, .95, .27]], dtype=np.float32)
        groundtruth_labels = np.array(
            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)
        groundtruth_weights = np.array([0.3, 0., 0.5], dtype=np.float32)

        # background class gets weight of 1.
        exp_cls_weights = [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], [0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
        exp_reg_weights = [0.3, 0., 0., 0.5]  # background class gets weight of 0.

        (cls_weights_out, reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners,
                                                      groundtruth_labels, groundtruth_weights)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_weights_out, exp_reg_weights)

    def test_assign_multidimensional_class_targets(self):

        def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
            box_coder = encode_mean_stddev

            unmatched_class_label = tf.constant([[0, 0], [0, 0]], tf.float32)
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {
                BoxField.BOXES: groundtruth_box_corners,
                BoxField.LABELS: groundtruth_labels
            }
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=unmatched_class_label)
            (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        anchor_means = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]],
            dtype=np.float32)
        groundtruth_box_corners = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9], [.75, 0, .95, .27]], dtype=np.float32)

        groundtruth_labels = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, .5]]],
                                      np.float32)

        exp_cls_targets = [[[0, 1], [1, 0]], [[1, 0], [0, 1]], [[0, 0], [0, 0]], [[0, 1], [1, .5]]]
        exp_cls_weights = [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]
        exp_reg_targets = [[0, 0, 0, 0], [0, 0, -1, 1], [0, 0, 0, 0], [0, 0, -.5, .2]]
        exp_reg_weights = [1, 1, 0, 1]
        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)
        self.assertEqual(cls_targets_out.dtype, np.float32)
        self.assertEqual(cls_weights_out.dtype, np.float32)
        self.assertEqual(reg_targets_out.dtype, np.float32)
        self.assertEqual(reg_weights_out.dtype, np.float32)

    def test_assign_empty_groundtruth(self):

        def graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels):
            similarity_calc = compute_iou
            matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
            box_coder = encode_mean_stddev
            unmatched_class_label = tf.constant([0, 0, 0], tf.float32)
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            groundtruth_boxlist = {
                BoxField.BOXES: groundtruth_box_corners,
                BoxField.LABELS: groundtruth_labels
            }
            target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)
            result = target_assigner.assign(anchors_boxlist,
                                            groundtruth_boxlist,
                                            unmatched_class_label=unmatched_class_label)
            (cls_targets, cls_weights, reg_targets, reg_weights, _) = result
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
        groundtruth_labels = np.zeros((0, 3), dtype=np.float32)
        anchor_means = np.array(
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]],
            dtype=np.float32)
        exp_cls_targets = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        exp_cls_weights = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        exp_reg_targets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        exp_reg_weights = [0, 0, 0, 0]
        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners, groundtruth_labels)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)
        self.assertEqual(cls_targets_out.dtype, np.float32)
        self.assertEqual(cls_weights_out.dtype, np.float32)
        self.assertEqual(reg_targets_out.dtype, np.float32)
        self.assertEqual(reg_weights_out.dtype, np.float32)

    def test_raises_error_on_incompatible_groundtruth_boxes_and_labels(self):
        similarity_calc = compute_iou
        matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
        box_coder = encode_mean_stddev
        unmatched_class_label = tf.constant([1, 0, 0, 0, 0, 0, 0], tf.float32)
        target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

        anchors_means = tf.constant([[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0],
                                     [.75, 0, 1.0, .25]])
        anchors = {BoxField.BOXES: anchors_means}

        box_corners = [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.8], [0.5, 0.5, 0.9, 0.9],
                       [.75, 0, .95, .27]]

        groundtruth_labels = tf.constant(
            [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0]], tf.float32)

        boxes = {BoxField.BOXES: box_corners, BoxField.LABELS: groundtruth_labels}

        #TODO designed a better tests Handling
        with self.assertRaises(Exception):
            target_assigner.assign(anchors, boxes, unmatched_class_label=unmatched_class_label)


class BatchTargetAssignerTest(tf.test.TestCase):

    def _get_target_assigner(self):
        similarity_calc = compute_iou
        matcher = argmax_matcher.ArgMaxMatcher(matched_threshold=0.5, unmatched_threshold=0.5)
        box_coder = encode_mean_stddev
        return targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

    def test_batch_assign_targets(self):

        def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2):
            box_list1 = {BoxField.BOXES: groundtruth_boxlist1, BoxField.LABELS: None}
            box_list2 = {BoxField.BOXES: groundtruth_boxlist2, BoxField.LABELS: None}
            gt_box_batch = [box_list1, box_list2]
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            agnostic_target_assigner = self._get_target_assigner()
            targets, weights, _ = targetassigner.batch_assign_targets(agnostic_target_assigner,
                                                                      anchors_boxlist, gt_box_batch)
            cls_targets = targets[LossField.CLASSIFICATION]
            cls_weights = weights[LossField.CLASSIFICATION]
            reg_targets = targets[LossField.LOCALIZATION]
            reg_weights = weights[LossField.LOCALIZATION]

            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
        groundtruth_boxlist2 = np.array(
            [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842]], dtype=np.float32)
        anchor_means = np.array(
            [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]], dtype=np.float32)

        exp_cls_targets = [[[1], [0], [0], [0]], [[0], [1], [1], [0]]]
        exp_cls_weights = [[[1], [1], [1], [1]], [[1], [1], [1], [1]]]
        exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                           [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                            [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
        exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)

    def test_batch_assign_multiclass_targets(self):

        def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2, class_targets1,
                     class_targets2):

            box_list1 = {BoxField.BOXES: groundtruth_boxlist1, BoxField.LABELS: class_targets1}
            box_list2 = {BoxField.BOXES: groundtruth_boxlist2, BoxField.LABELS: class_targets2}
            gt_box_batch = [box_list1, box_list2]
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            multiclass_target_assigner = self._get_target_assigner()
            num_classes = 3
            unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
            targets, weights, _ = targetassigner.batch_assign_targets(multiclass_target_assigner,
                                                                      anchors_boxlist, gt_box_batch,
                                                                      unmatched_class_label)
            cls_targets = targets[LossField.CLASSIFICATION]
            cls_weights = weights[LossField.CLASSIFICATION]
            reg_targets = targets[LossField.LOCALIZATION]
            reg_weights = weights[LossField.LOCALIZATION]

            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
        groundtruth_boxlist2 = np.array(
            [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842]], dtype=np.float32)
        class_targets1 = np.array([[0, 1, 0, 0]], dtype=np.float32)
        class_targets2 = np.array([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)

        anchor_means = np.array(
            [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]], dtype=np.float32)
        exp_cls_targets = [[[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                           [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]]
        exp_cls_weights = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
        exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                           [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                            [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
        exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                                     class_targets1, class_targets2)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)

    def test_batch_assign_multiclass_targets_with_padded_groundtruth(self):

        def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2, class_targets1,
                     class_targets2, groundtruth_weights1, groundtruth_weights2):
            box_list1 = {
                BoxField.BOXES: groundtruth_boxlist1,
                BoxField.LABELS: class_targets1,
                BoxField.WEIGHTS: groundtruth_weights1
            }
            box_list2 = {
                BoxField.BOXES: groundtruth_boxlist2,
                BoxField.LABELS: class_targets2,
                BoxField.WEIGHTS: groundtruth_weights2
            }

            gt_box_batch = [box_list1, box_list2]
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            multiclass_target_assigner = self._get_target_assigner()
            num_classes = 3
            unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
            targets, weights, _ = targetassigner.batch_assign_targets(multiclass_target_assigner,
                                                                      anchors_boxlist, gt_box_batch,
                                                                      unmatched_class_label)
            cls_targets = targets[LossField.CLASSIFICATION]
            cls_weights = weights[LossField.CLASSIFICATION]
            reg_targets = targets[LossField.LOCALIZATION]
            reg_weights = weights[LossField.LOCALIZATION]

            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2], [0., 0., 0., 0.]], dtype=np.float32)
        groundtruth_weights1 = np.array([1, 0], dtype=np.float32)
        groundtruth_boxlist2 = np.array(
            [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842], [0, 0, 0, 0]],
            dtype=np.float32)
        groundtruth_weights2 = np.array([1, 1, 0], dtype=np.float32)
        class_targets1 = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
        class_targets2 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.float32)

        anchor_means = np.array(
            [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]], dtype=np.float32)

        exp_cls_targets = [[[0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                           [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]]
        exp_cls_weights = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
        exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                           [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                            [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
        exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                                     class_targets1, class_targets2, groundtruth_weights1,
                                     groundtruth_weights2)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)

    def test_batch_assign_multidimensional_targets(self):

        def graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2, class_targets1,
                     class_targets2):
            box_list1 = {BoxField.BOXES: groundtruth_boxlist1, BoxField.LABELS: class_targets1}
            box_list2 = {BoxField.BOXES: groundtruth_boxlist2, BoxField.LABELS: class_targets2}
            gt_box_batch = [box_list1, box_list2]
            anchors_boxlist = {BoxField.BOXES: anchor_means}
            multiclass_target_assigner = self._get_target_assigner()
            target_dimensions = (2, 3)
            unmatched_class_label = tf.constant(np.zeros(target_dimensions), tf.float32)
            targets, weights, _ = targetassigner.batch_assign_targets(multiclass_target_assigner,
                                                                      anchors_boxlist, gt_box_batch,
                                                                      unmatched_class_label)
            cls_targets = targets[LossField.CLASSIFICATION]
            cls_weights = weights[LossField.CLASSIFICATION]
            reg_targets = targets[LossField.LOCALIZATION]
            reg_weights = weights[LossField.LOCALIZATION]

            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_boxlist1 = np.array([[0., 0., 0.2, 0.2]], dtype=np.float32)
        groundtruth_boxlist2 = np.array(
            [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842]], dtype=np.float32)
        class_targets1 = np.array([[[0, 1, 1], [1, 1, 0]]], dtype=np.float32)
        class_targets2 = np.array([[[0, 1, 1], [1, 1, 0]], [[0, 0, 1], [0, 0, 1]]],
                                  dtype=np.float32)

        anchor_means = np.array(
            [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]], dtype=np.float32)

        exp_cls_targets = [[[[0., 1., 1.], [1., 1., 0.]], [[0., 0., 0.], [0., 0., 0.]],
                            [[0., 0., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 0.]]],
                           [[[0., 0., 0.], [0., 0., 0.]], [[0., 1., 1.], [1., 1., 0.]],
                            [[0., 0., 1.], [0., 0., 1.]], [[0., 0., 0.], [0., 0., 0.]]]]
        exp_cls_weights = [[[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]],
                            [[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]],
                           [[[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]],
                            [[1., 1., 1.], [1., 1., 1.]], [[1., 1., 1.], [1., 1., 1.]]]]
        exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                           [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                            [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
        exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_boxlist1, groundtruth_boxlist2,
                                     class_targets1, class_targets2)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)

    def test_batch_assign_empty_groundtruth(self):

        def graph_fn(anchor_means, groundtruth_box_corners, gt_class_targets):
            groundtruth_boxlist = {
                BoxField.BOXES: groundtruth_box_corners,
                BoxField.LABELS: gt_class_targets
            }
            gt_box_batch = [groundtruth_boxlist]
            anchors_boxlist = {BoxField.BOXES: anchor_means}

            multiclass_target_assigner = self._get_target_assigner()
            num_classes = 3
            unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
            targets, weights, _ = targetassigner.batch_assign_targets(multiclass_target_assigner,
                                                                      anchors_boxlist, gt_box_batch,
                                                                      unmatched_class_label)
            cls_targets = targets[LossField.CLASSIFICATION]
            cls_weights = weights[LossField.CLASSIFICATION]
            reg_targets = targets[LossField.LOCALIZATION]
            reg_weights = weights[LossField.LOCALIZATION]
            return (cls_targets, cls_weights, reg_targets, reg_weights)

        groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
        anchor_means = np.array([[0, 0, .25, .25], [0, .25, 1, 1]], dtype=np.float32)
        exp_cls_targets = [[[1, 0, 0, 0], [1, 0, 0, 0]]]
        exp_cls_weights = [[[1, 1, 1, 1], [1, 1, 1, 1]]]
        exp_reg_targets = [[[0, 0, 0, 0], [0, 0, 0, 0]]]
        exp_reg_weights = [[0, 0]]
        num_classes = 3
        pad = 1
        gt_class_targets = np.zeros((0, num_classes + pad), dtype=np.float32)

        (cls_targets_out, cls_weights_out, reg_targets_out,
         reg_weights_out) = graph_fn(anchor_means, groundtruth_box_corners, gt_class_targets)
        self.assertAllClose(cls_targets_out, exp_cls_targets)
        self.assertAllClose(cls_weights_out, exp_cls_weights)
        self.assertAllClose(reg_targets_out, exp_reg_targets)
        self.assertAllClose(reg_weights_out, exp_reg_weights)


if __name__ == '__main__':
    tf.test.main()
