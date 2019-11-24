import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers
from tensorflow.keras.losses import CategoricalCrossentropy

from od.core.anchor_generator import generate_anchors
from od.core.argmax_matcher import ArgMaxMatcher
from od.core.box_coder import encode_boxes_faster_rcnn
from od.core.box_ops import compute_iou
from od.core.losses import SmoothL1Localization
from od.core.sampling_ops import batch_sample_balanced_positive_negative
from od.core.standard_fields import BoxField, LossField
from od.core.target_assigner import TargetAssigner, batch_assign_targets
from od.model.detection.abstract_detection_head import AbstractDetectionHead
from od.model.post_processing import post_process_rpn

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

        # TODO check force_match_for_each_row
        matcher = ArgMaxMatcher(0.7, 0.3, force_match_for_each_row=True, dtype=self.dtype)
        self.target_assigner = TargetAssigner(compute_iou,
                                              matcher,
                                              encode_boxes_faster_rcnn,
                                              dtype=self.dtype)

        self._anchor_strides = (4, 8, 16, 32, 64)
        self._anchor_ratios = anchor_ratios
        self.rpn_conv2d = KL.Conv2D(512, (3, 3),
                                    padding='same',
                                    kernel_initializer=self._kernel_initializer_classification_head,
                                    kernel_regularizer=self._kernel_regularizer)

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

    def call(self, inputs, training=None):
        """Create the computation graph for the rpn inference

        Arguments:

        - *inputs*: A list with the following schema:
          - *input_tensors*: A List of tensors the output of the pyramid
          - *image_informations*A Tensor of shape [batch_size, (height, width)]
            The height and the width are without the padding.
          - *ground_truths*: If the training is true, The ground_truths which is a list of dict with
          BoxField as key and a tensor as value.
        - *training*: Boolean

        Returns:

        - *nmsed_boxes*: A Tensor of shape [batch_size, max_detections, 4]
        containing the non-max suppressed boxes.
        - *nmsed_scores*: A Tensor of shape [batch_size, max_detections] containing
        the scores for the boxes.
        """

        if training:
            input_tensors, image_information, ground_truths = inputs
        else:
            input_tensors, image_information = inputs

        rpn_predictions = [self.build_rpn_head(tensor) for tensor in input_tensors]
        rpn_anchors = []
        for tensor, anchor_stride in zip(input_tensors, self._anchor_strides):
            anchors = generate_anchors(anchor_stride, tf.constant([8], self.dtype),
                                       tf.constant(self._anchor_ratios, self.dtype),
                                       tf.shape(tensor))
            # TODO clipping to investigate
            rpn_anchors.append(anchors)
        localization_pred = tf.concat([prediction[1] for prediction in rpn_predictions], 1)
        classification_pred = tf.concat([prediction[0] for prediction in rpn_predictions], 1)
        anchors = tf.concat([anchors for anchors in rpn_anchors], 0)
        classification_prob = tf.nn.softmax(classification_pred)

        if training:
            loss = self.compute_loss(localization_pred, classification_pred, anchors, ground_truths)
            self.add_loss(loss)
            return post_process_rpn(classification_prob,
                                    localization_pred,
                                    anchors,
                                    image_information,
                                    pre_nms_topk=12000,
                                    post_nms_topk=2000)
        return post_process_rpn(classification_prob,
                                localization_pred,
                                anchors,
                                image_information,
                                pre_nms_topk=6000,
                                post_nms_topk=1000)

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

        sample_idx = batch_sample_balanced_positive_negative(
            weights[LossField.CLASSIFICATION],
            SAMPLING_SIZE,
            labels,
            positive_fraction=SAMPLING_POSITIVE_RATIO,
            dtype=self.dtype)
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
