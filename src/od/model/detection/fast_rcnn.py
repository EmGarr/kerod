from typing import List

import gin
import tensorflow as tf
import tensorflow.keras.layers as KL

from tensorflow.keras import initializers
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError

from od.core.argmax_matcher import ArgMaxMatcher
from od.core.box_coder import encode_boxes_faster_rcnn
from od.core.box_ops import compute_iou
from od.core.sampling_ops import batch_sample_balanced_positive_negative
from od.core.standard_fields import BoxField, LossField
from od.core.target_assigner import TargetAssigner, batch_assign_targets
from od.model.detection.abstract_detection_head import AbstractDetectionHead
from od.model.detection.pooling_ops import multilevel_roi_align
from od.model.post_processing import post_process_fast_rcnn_boxes


class FastRCNN(AbstractDetectionHead):
    """Build the Fast-RCNN on top of the FPN. The parameters used
    are from [Feature Pyramidal Networks for Object Detection](https://arxiv.org/abs/1612.03144).

    Arguments:

    - *num_classes*: The number of classes that predict the classification head (N+1).
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(
            num_classes,
            CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True),
            MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE),  # like in tensorpack
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.001),
            **kwargs)

        matcher = ArgMaxMatcher(0.5, dtype=self.dtype)
        self.target_assigner = TargetAssigner(compute_iou,
                                              matcher,
                                              encode_boxes_faster_rcnn,
                                              dtype=self.dtype)
    def build(self, input_shape):
        self.denses = [
            KL.Dense(1024, kernel_initializer=initializers.VarianceScaling(), activation='relu')
            for _ in range(2)
        ]
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Build the computational graph of the fast RCNN HEAD.

        Arguments:

        *inputs*: A list with the following schema:

        1. *pyramid*: A List of tensors the output of the pyramid
        2. *anchors*: A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
        3. *image_informations*: A Tensor of shape [batch_size, (height, width)]
        The height and the width are without the padding.
        4. *ground_truths*: If the training is true, a dict with

        ```python
        ground_truths = {
            BoxField.BOXES:
                tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
            BoxField.LABELS:
                tf.constant([[[0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 0, 0]]], tf.float32),
            BoxField.WEIGHTS:
                tf.constant([[1, 0], [1, 1]], tf.float32),
            BoxField.NUM_BOXES:
                tf.constant([2, 1], tf.int32)
        }
        ```

        where `NUM_BOXES` allows to remove the padding created by tf.Data.

        - *training*: Boolean

        Returns:

        - *nmsed_boxes*: A Tensor of shape [batch_size, max_detections, 4]
        containing the non-max suppressed boxes.
        - *nmsed_scores*: A Tensor of shape [batch_size, max_detections] containing
        the scores for the boxes.
        - *nmsed_classes*: A Tensor of shape [batch_size, max_detections] 
        containing the class for boxes.
        - *valid_detections*: A [batch_size] int32 tensor indicating the number of
        valid detections per batch item. Only the top valid_detections[i] entries
        in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
        entries are zero paddings.
        """
        if training:
            pyramid, anchors, image_shape, image_information, ground_truths = inputs
            y_true, weights = self.sample_boxes(anchors, ground_truths)
            anchors = y_true[LossField.LOCALIZATION]
        else:
            pyramid, anchors, image_shape, image_information = inputs

        # Remove P6
        pyramid = pyramid[:-1]
        boxe_tensors = multilevel_roi_align(pyramid, anchors, image_shape, crop_size=7)
        l = KL.Flatten()(boxe_tensors)
        for dense in self.denses:
            l = dense(l)

        classification_pred, localization_pred = self.build_detection_head(
            tf.reshape(l, (-1, 1, 1, 1024)))
        batch_size = tf.shape(anchors)[0]
        classification_pred = tf.reshape(classification_pred, (batch_size, -1, self._num_classes))
        localization_pred = tf.reshape(localization_pred,
                                       (batch_size, -1, (self._num_classes - 1) * 4))

        if training:
            losses = self.compute_loss(y_true, weights, classification_pred, localization_pred)
            self.add_loss(losses)

        classification_pred = tf.nn.softmax(classification_pred)

        return post_process_fast_rcnn_boxes(classification_pred, localization_pred, anchors,
                                            image_information, self._num_classes)

    @gin.configurable()
    def sample_boxes(self,
                     anchors: tf.Tensor,
                     ground_truths: List[dict],
                     sampling_size: int = 512,
                     sampling_positive_ratio: float = 0.25):
        """Perform the sampling of the target anchors. During the training a set of RoIs is
        detected by the RPN. However, you do not want to analyse all the set. You only want
        to analyse the anchors that you sampled with this method.

        Arguments:

        - *anchors*: A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
        - *ground_truths*: A dict with the following format:

        ```python
        ground_truths = {
            BoxField.BOXES:
                tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
            BoxField.LABELS:
                tf.constant([[[0, 0, 1], [0, 1, 0]], [[0, 0, 1], [0, 0, 0]]], tf.float32),
            BoxField.WEIGHTS:
                tf.constant([[1, 0], [1, 1]], tf.float32),
            BoxField.NUM_BOXES:
                tf.constant([2, 1], tf.int32)
        }
        ```
        where `NUM_BOXES` allows to remove the padding created by tf.Data.

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
            - *LossField.CLASSIFICATION*: a tensor with shape [batch_size, num_anchors,
            num_classes],
            - *LossField.LOCALIZATION*: a tensor with shape [batch_size, num_anchors],

        Raises:

        - *ValueError*: If the batch_size is None.
        - *ValueError*: If the batch_size between your ground_truths and the anchors does not match.
        """
        # In graph mode unstack need to be aware of the batch_shape
        batch_size = anchors.get_shape().as_list()[0]
        if batch_size is None:
            raise ValueError("In training the batch size cannot be None. You should specify it"
                             " in tf.Keras.layers.Input using the argument batch_size.")
        anchors = [{BoxField.BOXES: anchor} for anchor in tf.unstack(anchors)]

        # Remove the padding and convert the ground_truths to the format
        # expected by the target_assigner
        gt_boxes = tf.unstack(ground_truths[BoxField.BOXES])
        gt_labels = tf.unstack(ground_truths[BoxField.LABELS])
        gt_weights = tf.unstack(ground_truths[BoxField.WEIGHTS])
        num_boxes = tf.unstack(ground_truths[BoxField.NUM_BOXES])
        ground_truths = []
        for b, l, w, nb in zip(gt_boxes, gt_labels, gt_weights, num_boxes):
            ground_truths.append({
                BoxField.BOXES: b[:nb],
                BoxField.LABELS: l[:nb],
                BoxField.WEIGHTS: w[:nb],
            })

        unmatched_class_label = tf.constant([1] + (self._num_classes - 1) * [0], self.dtype)
        y_true, weights, _ = batch_assign_targets(self.target_assigner, anchors, ground_truths,
                                                  unmatched_class_label)

        # Here we have a tensor of shape [batch_size, num_anchors, num_classes]. We want
        # to know all the foreground anchors for sampling.
        # [0, 0, 0, 1] -> [1]
        # [1, 0, 0, 0] -> [0]
        labels = tf.logical_not(tf.cast(y_true[LossField.CLASSIFICATION][:, :, 0], dtype=bool))
        sample_idx = batch_sample_balanced_positive_negative(
            weights[LossField.CLASSIFICATION],
            sampling_size,
            labels,
            positive_fraction=sampling_positive_ratio,
            dtype=self.dtype)

        weights[LossField.CLASSIFICATION] = tf.multiply(sample_idx,
                                                        weights[LossField.CLASSIFICATION])
        weights[LossField.LOCALIZATION] = tf.multiply(sample_idx, weights[LossField.LOCALIZATION])

        selected_boxes_idx = tf.where(tf.equal(sample_idx, 1))

        batch_size = tf.shape(sample_idx)[0]

        # Extract the selected anchors corresponding anchors
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
        # the corresponding target anchors
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
