from typing import List

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers
from tensorflow.keras.losses import CategoricalCrossentropy

from kerod.core.anchor_generator import Anchors
from kerod.core.argmax_matcher import ArgMaxMatcher
from kerod.core.box_coder import encode_boxes_faster_rcnn
from kerod.core.box_ops import compute_iou
from kerod.core.losses import SmoothL1Localization
from kerod.core.sampling_ops import batch_sample_balanced_positive_negative
from kerod.core.standard_fields import BoxField, LossField
from kerod.core.target_assigner import TargetAssigner, batch_assign_targets
from kerod.model.detection.abstract_detection_head import AbstractDetectionHead

SAMPLING_SIZE = 256
SAMPLING_POSITIVE_RATIO = 0.5


class RegionProposalNetwork(AbstractDetectionHead):
    """RPN which work only a pyramid network.
    It has been introduced in the [Faster R-CNN paper](https://arxiv.org/abs/1506.01497) and
    use the parameters from [Feature Pyramidal Networks for Object Detection](https://arxiv.org/abs/1612.03144).

    Arguments:

    - *anchor_ratios*: The ratios are the different shapes that you want to apply on your anchors.
            e.g: (0.5, 1, 2)
    """

    def __init__(self, anchor_ratios=(0.5, 1, 2), **kwargs):
        # Hyper parameter from tensorpack
        delta = 1. / 9
        super().__init__(
            2,
            CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True),
            SmoothL1Localization(delta),
            localization_loss_weight=1. / delta,
            multiples=len(anchor_ratios),
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.01),
            **kwargs)

        #Force each ground_truths to match to at least one anchor
        matcher = ArgMaxMatcher(0.7, 0.3, force_match_for_each_row=True, dtype=self._compute_dtype)
        self.target_assigner = TargetAssigner(compute_iou,
                                              matcher,
                                              encode_boxes_faster_rcnn,
                                              dtype=self._compute_dtype)

        anchor_strides = (4, 8, 16, 32, 64)
        anchor_zises = (32, 64, 128, 256, 512)
        self._anchor_ratios = anchor_ratios

        # Precompute a deterministic grid of anchors for each layer of the pyramid.
        # We will extract a subpart of the anchors according to
        self._anchors = [
            Anchors(stride, size, self._anchor_ratios)
            for stride, size in zip(anchor_strides, anchor_zises)
        ]

    def build(self, input_shape):
        self.rpn_conv2d = KL.Conv2D(512, (3, 3),
                                    padding='same',
                                    kernel_initializer=self._kernel_initializer_classification_head,
                                    kernel_regularizer=self._kernel_regularizer)
        super().build(input_shape)

    def build_rpn_head(self, inputs):
        """Predictions for the classification and the regression

        Arguments:

        - *inputs*: A tensor of  shape [batch_size, width, height, channel]

        Returns:

        A tuple of tensors of shape ([batch_size, num_anchors, 2], [batch_size, num_anchors, 4])
        """

        batch_size = tf.shape(inputs)[0]
        rpn_conv2d = self.rpn_conv2d(inputs)
        classification_head, localization_head = self.build_detection_head(rpn_conv2d)
        classification_head = tf.reshape(classification_head, (batch_size, -1, 2))
        localization_head = tf.reshape(localization_head, (batch_size, -1, 4))
        return classification_head, localization_head

    def call(self, inputs: List[tf.Tensor], training=None):
        """Create the computation graph for the rpn inference

        Argument:

        *inputs*: A List of tensors the output of the pyramid

        Returns:
        
        - *localization_pred*: A logit 3-D tensor of shape [batch_size, num_anchors, 4]
        - *classification_pred*: A logit 3-D tensor of shape [batch_size, num_anchors, 2]
        - *anchors*: A tensor of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]
        """
        anchors = tf.concat([anchors(tensor) for tensor, anchors in zip(inputs, self._anchors)], 0)

        rpn_predictions = [self.build_rpn_head(tensor) for tensor in inputs]
        localization_pred = tf.concat([prediction[1] for prediction in rpn_predictions], 1)
        classification_pred = tf.concat([prediction[0] for prediction in rpn_predictions], 1)

        return localization_pred, classification_pred, anchors

    def compute_loss(self, localization_pred, classification_pred, anchors, ground_truths):
        """Compute the loss

        Arguments:

        - *localization_pred*: A tensor of shape [batch_size, num_anchors, 4]
        - *classification_pred*: A tensor of shape [batch_size, num_anchors, 2]
        - *anchors*: A tensor of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]
        - *ground_truths*: A dict with BoxField as key and a tensor as value.

        ```python
        ground_truths = {
            BoxField.BOXES:
                tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
            BoxField.LABELS:
                tf.constant([[1, 0], [1, 0]], tf.float32),
            BoxField.WEIGHTS:
                tf.constant([[1, 0], [1, 1]], tf.float32),
            BoxField.NUM_BOXES:
                tf.constant([[2], [1]], tf.int32)
        }
        ```

        where `NUM_BOXES` allows to remove the padding created by tf.Data.

        Returns:

        - *classification_loss*: A scalar in tf.float32
        - *localization_loss*: A scalar in tf.float32
        """
        # Will be auto batch by the target assigner
        anchors = {BoxField.BOXES: anchors}

        # We only want the Localization field here the target assigner will understand that
        # it is the RPN mode.
        gt_boxes = tf.unstack(ground_truths[BoxField.BOXES])
        num_boxes = tf.unstack(ground_truths[BoxField.NUM_BOXES])
        ground_truths = [{BoxField.BOXES: b[:nb[0]]} for b, nb in zip(gt_boxes, num_boxes)]

        y_true, weights, _ = batch_assign_targets(self.target_assigner, anchors, ground_truths)

        ## Compute metrics
        recall = compute_rpn_metrics(tf.squeeze(y_true[LossField.CLASSIFICATION], axis=-1),
                                     classification_pred, weights[LossField.CLASSIFICATION])
        self.add_metric(recall, name='rpn_recall', aggregation='mean')

        ## Samling
        # y_true[LossField.CLASSIFICATION] is a [batch_size, num_anchors, 1]
        labels = tf.cast(y_true[LossField.CLASSIFICATION][:, :, 0], dtype=bool)
        sample_idx = batch_sample_balanced_positive_negative(
            weights[LossField.CLASSIFICATION],
            SAMPLING_SIZE,
            labels,
            positive_fraction=SAMPLING_POSITIVE_RATIO,
            dtype=self._compute_dtype)
        # Create one_hot encoding [batch_size, num_anchors, 1] -> [batch_size, num_anchors, 2]
        y_true[LossField.CLASSIFICATION] = tf.one_hot(tf.cast(
            y_true[LossField.CLASSIFICATION][:, :, 0], tf.int32),
                                                      depth=2)

        weights[LossField.CLASSIFICATION] = tf.multiply(sample_idx,
                                                        weights[LossField.CLASSIFICATION])
        weights[LossField.LOCALIZATION] = tf.multiply(sample_idx, weights[LossField.LOCALIZATION])

        y_pred = {
            LossField.CLASSIFICATION: classification_pred,
            LossField.LOCALIZATION: localization_pred
        }

        return self.compute_losses(y_true, y_pred, weights)

    def get_config(self):
        base_config = super().get_config()
        base_config['anchor_ratios'] = self._anchor_ratios
        return base_config


def compute_rpn_metrics(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
    """Useful metrics that allows to track how behave the training of the rpn head.

    Arguments:

    - *y_true*: A one-hot encoded vector with shape [batch_size, num_anchors]
    - *y_pred*: A tensor of shape [batch_size, num_anchors, 2],
    representing the classification logits.
    - *weights*: A tensor of shape [batch_size, num_anchors] where weights should

    Returns:

    - *recall*: Among all the boxes that we had to find how much did we found.
    """
    # Force the cast to avoid type issue when the mixed precision is activated
    y_true, y_pred, weights = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32), tf.cast(
        weights, tf.float32)

    # Sometimes the weights have decimal value we do not want that
    weights = tf.clip_by_value(tf.math.ceil(weights), 0, 1)
    masked_y_true = y_true * weights
    prediction = tf.cast(tf.argmax(y_pred, axis=-1, name='label_prediction'),
                         tf.float32) * weights  # 0 or 1
    correct = tf.cast(tf.equal(prediction, masked_y_true), tf.float32)

    fg_inds = tf.where(masked_y_true == 1)
    num_valid_anchor = tf.math.count_nonzero(masked_y_true)
    num_pos_foreground_prediction = tf.math.count_nonzero(tf.gather_nd(correct, fg_inds))
    recall = tf.truediv(num_pos_foreground_prediction, num_valid_anchor, name='recall')
    return recall
