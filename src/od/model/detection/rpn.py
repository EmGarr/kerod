import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras import initializers

from od.core.anchor_generator import generate_anchors
from od.core.argmax_matcher import ArgMaxMatcher
from od.core.box_coder import encode_boxes_faster_rcnn
from od.core.box_ops import compute_iou
from od.core.losses import CategoricalCrossentropy, SmoothL1Localization
from od.core.sampling_ops import batch_sample_balanced_positive_negative
from od.core.standard_fields import BoxField, LossField
from od.core.target_assigner import TargetAssigner, batch_assign_targets
from od.model.detection.abstract_detection_head import AbstractDetectionHead

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

        matcher = ArgMaxMatcher(0.7, 0.3, force_match_for_each_row=True)
        target_assigner = TargetAssigner(compute_iou, matcher, encode_boxes_faster_rcnn)
        super().__init__(
            2,
            target_assigner,
            CategoricalCrossentropy(),
            SmoothL1Localization(),
            multiples=len(anchor_ratios),
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.01),
            **kwargs)

        self._anchor_strides = (4, 8, 16, 32, 64)
        self._anchor_ratios = anchor_ratios
        self.rpn_conv2d = tfkl.Conv2D(
            512, (3, 3),
            padding='same',
            kernel_initializer=self._kernel_initializer_classification_head,
            kernel_regularizer=self._kernel_regularizer)

    def build_rpn_head(self, inputs):
        """Predictions for the classification and the regression

        Arguments:

        - *inputs*: A tensor of float32 and shape [batch_size, width, height, channel]

        Returns:

        A tuple of tensors of float32 and shape ([batch_size, num_anchors, 2], [batch_size, num_anchors, 4])
        """

        batch_size = tf.shape(inputs)[0]
        rpn_conv2d = self.rpn_conv2d(inputs)
        classification_head, localization_head = self.build_detection_head(rpn_conv2d)
        classification_head = tf.reshape(classification_head, (batch_size, -1, 2))
        localization_head = tf.reshape(localization_head, (batch_size, -1, 4))
        return classification_head, localization_head

    def call(self, inputs, training=None):
        """Does the rpn inference

        Arguments:

        - *inputs*: A list with the following schema:
          1. A List of tensors the output of the pyramid
          2. If training is true, The ground_truths which is a list of dict with
          BoxField as key and a tensor as value.
        - *training*: Boolean

        Returns:

        - *localization_pred*: A tensor of shape [batch_size, num_anchors, 4]
        - *classification_pred*: A tensor of shape [batch_size, num_anchors, 2]
        - *anchors*: A tensor of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]
        """

        if training:
            input_tensors, ground_truths = inputs
        else:
            input_tensors = inputs[0]

        rpn_predictions = [self.build_rpn_head(tensor) for tensor in input_tensors]
        rpn_anchors = []
        for tensor, anchor_stride in zip(input_tensors, self._anchor_strides):
            anchors = generate_anchors(anchor_stride, tf.constant([8], tf.float32),
                                       tf.constant(self._anchor_ratios, tf.float32),
                                       tf.shape(tensor))
            rpn_anchors.append(anchors)
        localization_pred = tf.concat([prediction[1] for prediction in rpn_predictions], 1)
        classification_pred = tf.concat([prediction[0] for prediction in rpn_predictions], 1)
        anchors = tf.concat([anchors for anchors in rpn_anchors], 0)

        if training:
            loss = self.compute_loss(localization_pred, classification_pred, anchors, ground_truths)
            self.add_loss(loss)

        return localization_pred, classification_pred, anchors

    def compute_loss(self, localization_pred, classification_pred, anchors, ground_truths):
        """Compute the loss

        Arguments:

        - *localization_pred*: A tensor of shape [batch_size, num_anchors, 4]
        - *classification_pred*: A tensor of shape [batch_size, num_anchors, 2]
        - *anchors*: A tensor of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]
        - *ground_truths*: a list of dict with BoxField as key and a tensor as value.

        Returns:

        - *classification_loss*: A scalar in tf.float32
        - *localization_loss*: A scalar in tf.float32
        """
        # Will be auto batch by the target assigner
        anchors = {BoxField.BOXES: anchors}
        # We only want the Localization field here the target assigner will understand that
        # it is the RPN mode.
        ground_truths = [{BoxField.BOXES: gt[BoxField.BOXES]} for gt in ground_truths]
        y_true, weights, _ = batch_assign_targets(self.target_assigner, anchors, ground_truths)

        # y_true[LossField.CLASSIFICATION] is a [batch_size, num_anchors, 1]
        labels = tf.cast(y_true[LossField.CLASSIFICATION][:, :, 0], dtype=bool)

        sample_idx = batch_sample_balanced_positive_negative(weights[LossField.CLASSIFICATION],
                                                             SAMPLING_SIZE,
                                                             labels,
                                                             positive_fraction=SAMPLING_POSITIVE_RATIO)
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
