from typing import List

import gin
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers

from od.core.argmax_matcher import ArgMaxMatcher
from od.core.box_coder import encode_boxes_faster_rcnn
from od.core.box_ops import compute_iou
from od.core.losses import CategoricalCrossentropy, SmoothL1Localization
from od.core.sampling_ops import batch_sample_balanced_positive_negative
from od.core.standard_fields import BoxField, LossField
from od.core.target_assigner import TargetAssigner, batch_assign_targets
from od.model.detection.abstract_detection_head import AbstractDetectionHead
from od.model.detection.pooling_ops import multilevel_roi_align


class FastRCNN(AbstractDetectionHead):
    """Build the Fast-RCNN on top of the FPN. The parameters used
    are from [Feature Pyramidal Networks for Object Detection](https://arxiv.org/abs/1612.03144).

    Arguments:

    - *num_classes*: The number of classes that predict the classification head (N+1).
    """

    def __init__(self, num_classes, **kwargs):
        matcher = ArgMaxMatcher(0.5)
        target_assigner = TargetAssigner(compute_iou, matcher, encode_boxes_faster_rcnn)

        super().__init__(
            num_classes,
            target_assigner,
            CategoricalCrossentropy(),
            SmoothL1Localization(),
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.001),
            **kwargs)

        self.denses = [
            KL.Dense(1024, kernel_initializer=initializers.VarianceScaling(), activation='relu')
            for _ in range(2)
        ]

    def call(self, inputs, training=None):
        if training:
            pyramid, boxes, image_shape, ground_truths = inputs
            y_true, weights = self.sample_boxes(boxes, ground_truths)
            boxes = y_true[LossField.LOCALIZATION]
        else:
            pyramid, boxes, image_shape = inputs

        # Remove P6
        pyramid = pyramid[:-1]
        boxe_tensors = multilevel_roi_align(pyramid, boxes, image_shape, crop_size=7)
        l = KL.Flatten()(boxe_tensors)
        for dense in self.denses:
            l = dense(l)

        classification_head, localization_head = self.build_detection_head(
            tf.reshape(l, (-1, 1, 1, 1024)))
        batch_size = tf.shape(boxes)[0]
        classification_head = tf.reshape(classification_head, (batch_size, -1, self._num_classes))
        localization_head = tf.reshape(localization_head,
                                       (batch_size, -1, (self._num_classes - 1) * 4))

        if training:
            losses = self.compute_loss(y_true, weights, classification_head, localization_head)
            self.add_loss(losses)

        return classification_head, localization_head

    @gin.configurable()
    def sample_boxes(self,
                     batch_boxes: tf.Tensor,
                     ground_truths: List[dict],
                     sampling_size: int = 512,
                     sampling_positive_ratio: float = 0.25):
        """Perform the sampling of the target boxes. During the training a set of RoIs is
        detected by the RPN. However, you do not want to analyse all the set. You only want
        to analyse the boxes that you sampled with this method.

        Arguments:

        - *batch_boxes*: A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
        - *ground_truths*: A list of dict objects with length batch_size
        representing groundtruth boxes for each image in the batch and their labels, weights
        - *sampling_size*: Desired sampling size. If None, keeps all positive samples and
        randomly selects negative samples so that the positive sample fraction
        matches positive_fraction.
        - *sampling_positive_ratio*: Desired fraction of positive examples (scalar in [0,1])
        in the batch.

        Returns:

        - *y_true*: A dict with :
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors,
            num_classes],
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors,
            box_code_dimension]

        - *weights*: A dict with:
            * *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors,
            num_classes],
            * *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors],

        Raises:

        - *ValueError*: If your sampling size is superior to the number of boxes in input.
        - *ValueError*: If the batch_size between your ground_truths and the boxes does not match.
        """
        boxes = [{BoxField.BOXES: boxes} for boxes in tf.unstack(batch_boxes)]

        if tf.shape(batch_boxes)[1] < sampling_size:
            raise ValueError('Your sampling size should be inferior or equal to the number '
                             'of boxes in input. Inspect the num_boxes dimension for batch_boxes'
                             'with [batch_size, num_boxes, 4]')
        if len(boxes) != len(ground_truths):
            raise ValueError(f'Length of boxes is {len(boxes)} and length of ground_truths is '
                             f'{len(ground_truths)} should be the same.')

        unmatched_class_label = tf.constant([1] + (self._num_classes - 1) * [0], batch_boxes.dtype)
        y_true, weights, _ = batch_assign_targets(self.target_assigner, boxes, ground_truths,
                                                  unmatched_class_label)

        # Here we have a tensor of shape [batch_size, num_anchors, num_classes]. We want
        # to know all the foreground boxes for sampling.
        # [0, 0, 0, 1] -> [1]
        # [1, 0, 0, 0] -> [0]
        labels = tf.logical_not(tf.cast(y_true[LossField.CLASSIFICATION][:, :, 0], dtype=bool))
        sample_idx = batch_sample_balanced_positive_negative(
            weights[LossField.CLASSIFICATION],
            sampling_size,
            labels,
            positive_fraction=sampling_positive_ratio,
            dtype=batch_boxes.dtype)

        weights[LossField.CLASSIFICATION] = tf.multiply(sample_idx,
                                                        weights[LossField.CLASSIFICATION])
        weights[LossField.LOCALIZATION] = tf.multiply(sample_idx, weights[LossField.LOCALIZATION])

        selected_boxes_idx = tf.where(tf.equal(sample_idx, 1))

        batch_size = tf.shape(sample_idx)[0]

        # Extract the selected boxes corresponding boxes
        # tf.gather_nd collaps the batch_together so we reshape with the proper batch_size
        y_true[LossField.LOCALIZATION] = tf.reshape(
            tf.gather_nd(y_true[LossField.LOCALIZATION], selected_boxes_idx), (batch_size, -1, 4))

        y_true[LossField.CLASSIFICATION] = tf.reshape(
            tf.gather_nd(y_true[LossField.CLASSIFICATION], selected_boxes_idx),
            (batch_size, -1, self._num_classes))

        for key in y_true.keys():
            weights[key] = tf.reshape(tf.gather_nd(weights[key], selected_boxes_idx),
                                      (batch_size, -1))
            weights[key] = tf.stop_gradient(weights[key])
            y_true[key] = tf.stop_gradient(y_true[key])
        return y_true, weights

    def compute_loss(self, y_true: dict, weights: dict, classification_pred: tf.Tensor,
                     localization_pred: tf.Tensor):
        """Compute the loss of the FastRCNN

        Arguments:

        - *y_true*: A dict with :
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors, num_classes]
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors, 4]
        - *weights*: A dict with:
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors, num_classes]
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors]
        - *classification_pred*: A tensor and shape
        [batch_size, num_anchors, num_classes]
        - *localization_pred*: A tensor and shape
        [batch_size, num_anchors, (num_classes - 1) * 4]

        Returns:

        - *classification_loss*: A scalar
        - *localization_loss*: A scalar
        """

        # y_true[LossField.CLASSIFICATION] is just 1 and 0 we are using it as mask to extract
        # the corresponding target boxes
        batch_size = tf.shape(classification_pred)[0]
        targets = tf.reshape(y_true[LossField.CLASSIFICATION], [-1])

        # We need to insert a fake background classes at the position 0
        localization_pred = tf.pad(localization_pred, [[0, 0], [0, 0], [4, 0]])
        localization_pred = tf.reshape(localization_pred, [-1, 4])

        extracted_localization_pred = tf.boolean_mask(localization_pred, tf.greater(targets, 0))
        extracted_localization_pred = tf.reshape(extracted_localization_pred, (batch_size, -1, 4))
        y_pred = {
            LossField.CLASSIFICATION: classification_pred,
            LossField.LOCALIZATION: extracted_localization_pred
        }
        return self.compute_losses(y_true, y_pred, weights)
