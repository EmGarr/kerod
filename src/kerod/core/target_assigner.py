"""[Documentation taken from tensorflow/models/object_detection]. The code has been completely
rewritten.

Base target assigner module.

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
"""
from typing import Callable

import tensorflow as tf
from tensorflow.keras import backend as K

from kerod.core.matcher import Matcher
from kerod.core.standard_fields import BoxField, LossField
from kerod.utils import item_assignment, get_full_indices


class TargetAssigner:
    """Target assigner to compute classification and regression targets."""

    def __init__(self,
                 similarity_calc: Callable,
                 matcher: Matcher,
                 box_encoder: Callable,
                 dtype=None):
        """Construct Object Detection Target Assigner.

        Arguments:

        - *similarity_calc*: a method wich allow to compute a similarity between two batch of boxes

        - *matcher*: an od.core.Matcher used to match groundtruth to anchors.

        - *box_encoder*: a method which allow to encode matching
            groundtruth boxes with respect to anchors.
        """
        self._similarity_calc = similarity_calc
        self._matcher = matcher
        self._box_encoder = box_encoder
        if dtype is None:
            dtype = K.floatx()
        self.dtype = dtype

    @property
    def box_encoder(self):
        return self._box_encoder

    def assign(self, anchors: dict, groundtruth: dict):
        """Assign classification and regression targets to each anchor.

        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).

        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.

        Arguments:

        - *anchors*: a dict representing a batch of M anchors
            1. BoxField.BOXES: A tensor of shape [batch_size, num_anchors, (y1, x1, y2, x2)] representing the boxes and resized to the image shape.
        - *groundtruth*: a dict representing a batch of M groundtruth boxes
            1. BoxField.BOXES: A tensor of shape [batch_size, num_gt, (y1, x1, y2, x2)] representing
            the boxes and resized to the image shape
            2. BoxField.LABELS: A tensor of shape [batch_size, num_gt, ]
            3. BoxField.NUM_BOXES: A tensor of shape [batch_size].
            It is usefull to unpad the data in case of a batched training
            4. BoxField.WEIGHTS: A tensor of shape [batch_size, num_gt]

        Returns:

        - *y_true*: A dict with :
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors]
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors,
            box_code_dimension]

        - *weights*: A dict with:
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors],
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors],
        """
        shape = tf.shape(groundtruth[BoxField.BOXES])
        batch_size = shape[0]
        num_gt_boxes = shape[1]
        groundtruth_labels = groundtruth.get(BoxField.LABELS)
        groundtruth_weights = groundtruth.get(BoxField.WEIGHTS)
        if groundtruth_weights is None:
            groundtruth_weights = tf.ones([batch_size, num_gt_boxes], self.dtype)

        match_quality_matrix = self._similarity_calc(groundtruth, anchors)

        matches, matched_labels = self._matcher(match_quality_matrix,
                                                groundtruth[BoxField.NUM_BOXES])

        reg_targets = self._create_regression_targets(anchors, groundtruth, matches, matched_labels)
        cls_targets = self._create_classification_targets(groundtruth_labels, matches,
                                                          matched_labels)

        groundtruth_weights = self.gather(groundtruth_weights, matches)
        reg_weights = self._create_regression_weights(groundtruth_weights, matched_labels)
        cls_weights = self._create_classification_weights(groundtruth_weights, matched_labels)

        y_true = {
            LossField.CLASSIFICATION: tf.cast(cls_targets, self.dtype),
            LossField.LOCALIZATION: tf.cast(reg_targets, self.dtype)
        }
        weights = {
            LossField.CLASSIFICATION: tf.cast(cls_weights, self.dtype),
            LossField.LOCALIZATION: tf.cast(reg_weights, self.dtype)
        }

        return y_true, weights

    def gather(self, tensor, indices):
        indices = get_full_indices(indices)
        return tf.gather_nd(tensor, indices)

    def _create_regression_targets(self, anchors: dict, groundtruth: dict, matches: tf.Tensor,
                                   matched_labels: tf.Tensor) -> tf.Tensor:
        """Returns a regression target for each anchor.

        Arguments:

        - *anchors*: A tensor of shape [batch_size, num_anchors, (y1, x1, y2, x2)] representing the boxes
        and resized to the image shape.

        - *groundtruth*: a dict representing a batch of M groundtruth boxes
            1. BoxField.BOXES: A tensor of shape [batch_size, num_gt, (y1, x1, y2, x2)] representing
            the boxes and resized to the image shape
            2. BoxField.LABELS: A tensor of shape [batch_size, num_gt, ]
            3. BoxField.NUM_BOXES: A tensor of shape [batch_size].
            It is usefull to unpad the data in case of a batched training
            4. BoxField.WEIGHTS: A tensor of shape [batch_size, num_gt]

        - *matches*: a tensor of float32 and shape [batch_size, N], where matches[b, i] is a matched
               ground-truth index in [b, 0, M)
        - *match_labels*: a tensor of int8 and shape [batch_size, N], where match_labels[i] indicates
                whether a prediction is a true (1) or false positive (0) or ignored (-1)

        Returns:

        *reg_targets*: A tensor with shape [N, box_code_dimension]
        """
        matched_gt_boxes = self.gather(groundtruth[BoxField.BOXES], matches)

        matched_reg_targets = self._box_encoder(matched_gt_boxes, anchors[BoxField.BOXES])

        # Zero out the unmatched and ignored regression targets.
        unmatched_ignored_reg_targets = tf.zeros_like(matched_reg_targets,
                                                      dtype=matched_reg_targets.dtype)
        matched_anchors_mask = matched_labels >= 1
        reg_targets = tf.where(matched_anchors_mask[..., None],
                               x=matched_reg_targets,
                               y=unmatched_ignored_reg_targets)
        return reg_targets

    def _create_classification_targets(self, groundtruth_labels: tf.Tensor, matches: tf.Tensor,
                                       matched_labels: tf.Tensor):
        """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Arguments:

        - *groundtruth_labels*:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
            with labels for each of the ground_truth boxes. The subshape
            [d_1, ... d_k] can be empty (corresponding to scalar labels).

        - *matches*: a tensor of float32 and shape [batch_size, N], where matches[b, i] is a matched
               ground-truth index in [b, 0, M)
        - *match_labels*: a tensor of int8 and shape [batch_size, N], where match_labels[i] indicates
                whether a prediction is a true (1) or false positive (0) or ignored (-1)


        Returns:

        A tensor of shape [num_anchors, d_1, d_2 ... d_k], where the
        subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
        shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
        gathered_tensor = self.gather(groundtruth_labels, matches)
        # Set all the match values inferior or equal to 0 to background_classes
        indicator = matched_labels <= 0
        gathered_tensor = item_assignment(gathered_tensor, indicator, 0)
        return gathered_tensor

    def _create_regression_weights(self, groundtruth_weights: tf.Tensor, matched_labels: tf.Tensor):
        """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Arguments:

        - *groundtruth_weights*: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        - *match_labels*: a tensor of int8 and shape [batch_size, N], where match_labels[i] indicates
                whether a prediction is a true (1) or false positive (0) or ignored (-1)

        Returns:

        A tensor of shape [batch_size, num_anchors] representing the box regression weights.
        """
        indicator = matched_labels > 0
        weights = tf.where(indicator, groundtruth_weights, 0)
        return weights

    def _create_classification_weights(self, groundtruth_weights: tf.Tensor,
                                       matched_labels: tf.Tensor):
        """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Arguments:

        - *groundtruth_weights*: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        - *match_labels*: a tensor of int8 and shape [batch_size, N], where match_labels[i] indicates
                whether a prediction is a true (1) or false positive (0) or ignored (-1)

        Returns:

        A tensor of shape [batch_size, num_anchors] representing classification weights.
        """
        indicator = matched_labels < 0
        weights = tf.where(indicator, tf.constant(0., dtype=self.dtype), groundtruth_weights)
        indicator = matched_labels == 0
        weights = tf.where(indicator, tf.constant(1, dtype=self.dtype), weights)
        return weights
