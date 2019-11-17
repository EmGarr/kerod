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
"""Base target assigner module.

The job of a TargetAssigner is, for a given set of anchors (bounding boxes) and
groundtruth detections (bounding boxes), to assign classification and regression
targets to each anchor as well as weights to each anchor (specifying, e.g.,
which anchors should not contribute to training loss).

It assigns classification/regression targets by performing the following steps:

1. Computing pairwise similarity between anchors and groundtruth boxes using a
  provided RegionSimilarity Calculator
2. Computing a matching based on the similarity matrix using a provided Matcher
3. Assigning regression targets based on the matching and a provided BoxCoder
4. Assigning classification targets based on the matching and groundtruth labels

Note that TargetAssigners only operate on detections from a single
image at a time, so any logic for applying a TargetAssigner to multiple
images must be handled externally.
"""
from typing import List, Callable

import tensorflow as tf
from tensorflow.keras import backend as K

from od.core import matcher as mat
from od.core.standard_fields import BoxField, LossField


class TargetAssigner(object):
    """Target assigner to compute classification and regression targets."""

    def __init__(self,
                 similarity_calc: Callable,
                 matcher: mat.Matcher,
                 box_encoder: Callable,
                 negative_class_weight=1.0,
                 dtype=None):
        """Construct Object Detection Target Assigner.

        Arguments:

        - *similarity_calc*: a method wich allow to compute a similarity between two set
        of boxes
        - *matcher*: an od.core.Matcher used to match groundtruth to
            anchors.
        - *box_encoder*: a method which allow to encode matching
            groundtruth boxes with respect to anchors.
        - *negative_class_weight*: classification weight to be associated to negative
            anchors (default: 1.0). The weight must be in [0., 1.].

        Raises:

        *ValueError*: if matcher is not a Matcher
        """
        if not isinstance(matcher, mat.Matcher):
            raise ValueError('matcher must be a Matcher')
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_encoder = box_encoder
        self._negative_class_weight = negative_class_weight
        if dtype is None:
            dtype = K.floatx()
        self.dtype = dtype

    @property
    def box_encoder(self):
        return self._box_encoder

    def assign(self, anchors: dict, groundtruth_boxes: dict, unmatched_class_label=None):
        """Assign classification and regression targets to each anchor.

        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth_boxes and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).

        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.

        Arguments:

        - *anchors*: a dict representing N anchors
        - *groundtruth_boxes*: a dict representing M groundtruth boxes
        - *unmatched_class_label*: A tensor of shape [d_1, d_2, ..., d_k]
            which is consistent with the classification target for each
            anchor (and can be empty for scalar targets).  This shape must thus be
            compatible with the groundtruth labels that are passed to the "assign"
            function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
            If set to None, unmatched_cls_target is set to be [0] for each anchor.

        Returns:

        - cls_targets: A tensor of shape [num_anchors, d_1, d_2 ... d_k],
            where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
            which has shape [num_gt_boxes, d_1, d_2, ... d_k].
        - cls_weights: A tensor with shape [num_anchors],
            representing weights for each anchors.
        - reg_targets: A tensor with shape [num_anchors, box_code_dimension]
        - reg_weights: A tensor with shape [num_anchors]
        - match: an int32 tensor of shape [num_anchors] containing result of anchor
        groundtruth_boxes matching. Each position in the tensor indicates an anchor
            and holds the following meaning:
            .1 if match[i] >= 0, anchor i is matched with groundtruth match[i].
            .2 if match[i]=-1, anchor i is marked to be background .
            .3 if match[i]=-2, anchor i is ignored since it is not background and
                does not have sufficient overlap to call it a foreground.
        """

        if unmatched_class_label is None:
            unmatched_class_label = tf.constant([0], self.dtype)

        num_gt_boxes = tf.shape(groundtruth_boxes[BoxField.BOXES])[0]

        groundtruth_labels = groundtruth_boxes.get(BoxField.LABELS)
        if groundtruth_labels is None:
            groundtruth_labels = tf.ones(tf.expand_dims(num_gt_boxes, 0), dtype=self.dtype)
            groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)

        groundtruth_weights = groundtruth_boxes.get(BoxField.WEIGHTS)
        if groundtruth_weights is None:
            groundtruth_weights = tf.ones([num_gt_boxes], dtype=self.dtype)

        # set scores on the gt boxes
        scores = 1 - groundtruth_labels[:, 0]
        groundtruth_boxes[BoxField] = scores

        match_quality_matrix = self._similarity_calc(groundtruth_boxes[BoxField.BOXES],
                                                     anchors[BoxField.BOXES])
        match = self._matcher.match(match_quality_matrix,
                                    valid_rows=tf.greater(groundtruth_weights, 0))
        reg_targets = self._create_regression_targets(anchors, groundtruth_boxes, match)
        cls_targets = self._create_classification_targets(groundtruth_labels, unmatched_class_label,
                                                          match)
        reg_weights = self._create_regression_weights(match, groundtruth_weights)

        cls_weights = self._create_classification_weights(match, groundtruth_weights)

        num_anchors = tf.shape(anchors[BoxField.BOXES])[0]
        if num_anchors is not None:
            reg_targets = self._reset_target_shape(reg_targets, num_anchors)
            cls_targets = self._reset_target_shape(cls_targets, num_anchors)
            reg_weights = self._reset_target_shape(reg_weights, num_anchors)
            cls_weights = self._reset_target_shape(cls_weights, num_anchors)

        return (cls_targets, cls_weights, reg_targets, reg_weights, match.match_results)

    def _reset_target_shape(self, target, num_anchors):
        """Sets the static shape of the target.

        Arguments:

        - *target*: the target tensor. Its first dimension will be overwritten.
        - *num_anchors*: the number of anchors, which is used to override the target's
            first dimension.

        Returns:

        A tensor with the shape info filled in.
        """
        target_shape = target.get_shape().as_list()
        target_shape[0] = num_anchors
        target.set_shape(target_shape)
        return target

    def _create_regression_targets(self, anchors: dict, groundtruth_boxes: dict,
                                   match: mat.Match) -> tf.Tensor:
        """Returns a regression target for each anchor.

        Arguments:

        - *anchors*: a dict representing N anchors
        - *groundtruth_boxes*: a dict representing M groundtruth_boxes
        - *match*: a matcher.Match object

        Returns:

        *reg_targets*: a float32 tensor with shape [N, box_code_dimension]
        """
        matched_gt_boxes = match.gather_based_on_match(groundtruth_boxes[BoxField.BOXES],
                                                       unmatched_value=tf.zeros(4,
                                                                                dtype=self.dtype),
                                                       ignored_value=tf.zeros(4, dtype=self.dtype))

        matched_reg_targets = self._box_encoder(matched_gt_boxes, anchors[BoxField.BOXES])
        match_results_shape = tf.shape(match.match_results)

        # Zero out the unmatched and ignored regression targets.
        unmatched_ignored_reg_targets = tf.tile(tf.constant([4 * [0]], self.dtype),
                                                [match_results_shape[0], 1])
        matched_anchors_mask = tf.expand_dims(match.matched_column_indicator(), axis=-1)
        reg_targets = tf.where(matched_anchors_mask,
                               x=matched_reg_targets,
                               y=unmatched_ignored_reg_targets)
        return reg_targets

    def _create_classification_targets(self, groundtruth_labels, unmatched_class_label, match):
        """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Arguments:

        - *groundtruth_labels*:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
            with labels for each of the ground_truth boxes. The subshape
            [d_1, ... d_k] can be empty (corresponding to scalar labels).
        - *unmatched_class_label*: A tensor of shape [d_1, d_2, ..., d_k]
            which is consistent with the classification target for each
            anchor (and can be empty for scalar targets).  This shape must thus be
            compatible with the groundtruth labels that are passed to the "assign"
            function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
        - *match*: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.

        Returns:

        A tensor of shape [num_anchors, d_1, d_2 ... d_k], where the
        subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
        shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
        return match.gather_based_on_match(groundtruth_labels,
                                           unmatched_value=unmatched_class_label,
                                           ignored_value=unmatched_class_label)

    def _create_regression_weights(self, match: mat.Matcher, groundtruth_weights: tf.Tensor):
        """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Arguments:

        - *match*: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
        - *groundtruth_weights*: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:

        A tensor of shape [num_anchors] representing regression weights.
        """
        return match.gather_based_on_match(groundtruth_weights,
                                           ignored_value=tf.constant(0, groundtruth_weights.dtype),
                                           unmatched_value=tf.constant(
                                               0, groundtruth_weights.dtype))

    def _create_classification_weights(self, match: mat.Matcher, groundtruth_weights: tf.Tensor):
        """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Arguments:

        *match*: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
        *groundtruth_weights*: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:

        A tensor of shape [num_anchors] representing classification
        weights.
        """
        return match.gather_based_on_match(groundtruth_weights,
                                           ignored_value=tf.constant(
                                               0, dtype=groundtruth_weights.dtype),
                                           unmatched_value=tf.cast(self._negative_class_weight,
                                                                   dtype=groundtruth_weights.dtype))


def batch_assign_targets(target_assigner: TargetAssigner,
                         anchors_batch: List[dict],
                         gt_box_batch: List[dict],
                         unmatched_class_label=None):
    """Batched assignment of classification and regression y_true.

    Arguments:

    - *target_assigner*: a target assigner.
    - *anchors_batch*: A dict representing N box anchors or a list of dict
        with length batch_size representing anchor sets.
    - *gt_box_batch*: a list of dict objects with length batch_size
        representing groundtruth boxes for each image in the batch and their labels, weights
    - *unmatched_class_label*: A tensor of shape [d_1, d_2, ..., d_k]
        which is consistent with the classification target for each
        anchor (and can be empty for scalar y_true).  This shape must thus be
        compatible with the groundtruth labels that are passed to the "assign"
        function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).

    Returns:

    - *y_true*: A dict with :
        - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors,
        num_classes],
        - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors,
        box_code_dimension]

    - *weights*: A dict with:
        - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors,
        num_classes],
        - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors],

    - *match*: an int32 tensor of shape [batch_size, num_anchors] containing result
        of anchor groundtruth matching. Each position in the tensor indicates an
        anchor and holds the following meaning:
        .1 if match[x, i] >= 0, anchor i is matched with groundtruth match[x, i].
        .2 if match[x, i]=-1, anchor i is marked to be background .
        .3 if match[x, i]=-2, anchor i is ignored since it is not background and
            does not have sufficient overlap to call it a foreground.

    Raises:

    *ValueError*: if input list lengths are inconsistent, i.e.,
        batch_size == len(gt_box_batch) == len(gt_class_targets_batch)
            and batch_size == len(anchors_batch) unless anchors_batch is a single dict.
    """
    if not isinstance(anchors_batch, list):
        anchors_batch = len(gt_box_batch) * [anchors_batch]

    if not all(isinstance(anchors, dict) for anchors in anchors_batch):
        raise ValueError('anchors_batch must be a dict.')

    cls_targets_list = []
    cls_weights_list = []
    reg_targets_list = []
    reg_weights_list = []
    match_list = []
    for anchors, gt_boxes in zip(anchors_batch, gt_box_batch):
        (cls_targets, cls_weights, reg_targets, reg_weights,
         match) = target_assigner.assign(anchors, gt_boxes, unmatched_class_label)
        cls_targets_list.append(cls_targets)
        cls_weights_list.append(cls_weights)
        reg_targets_list.append(reg_targets)
        reg_weights_list.append(reg_weights)
        match_list.append(match)

    cls_targets = tf.stack(cls_targets_list)
    cls_weights = tf.stack(cls_weights_list)
    reg_targets = tf.stack(reg_targets_list)
    reg_weights = tf.stack(reg_weights_list)
    match = tf.stack(match_list)

    y_true = {LossField.CLASSIFICATION: cls_targets, LossField.LOCALIZATION: reg_targets}
    weights = {LossField.CLASSIFICATION: cls_weights, LossField.LOCALIZATION: reg_weights}

    return y_true, weights, match
