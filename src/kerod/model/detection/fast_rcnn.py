from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanAbsoluteError

from kerod.core.matcher import Matcher
from kerod.core.box_coder import encode_boxes_faster_rcnn
from kerod.core.box_ops import compute_iou
from kerod.core.sampling_ops import batch_sample_balanced_positive_negative
from kerod.core.standard_fields import BoxField, LossField
from kerod.core.target_assigner import TargetAssigner
from kerod.model.detection.abstract_detection_head import AbstractDetectionHead
from kerod.model.detection.pooling_ops import multilevel_roi_align


class FastRCNN(AbstractDetectionHead):
    """Build the Fast-RCNN on top of the FPN. The parameters used
    are from [Feature Pyramidal Networks for Object Detection](https://arxiv.org/abs/1612.03144).

    Arguments:

    - *num_classes*: The number of classes that predict the classification head (N+1) where N
    is the number of classes of your dataset and 1 is the background.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(
            num_classes,
            SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                          from_logits=True),
            MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE),  # like in tensorpack
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.001),
            **kwargs)

        matcher = Matcher([0.5], [0, 1])
        self.target_assigner = TargetAssigner(compute_iou,
                                              matcher,
                                              encode_boxes_faster_rcnn,
                                              dtype=self._compute_dtype)

    def build(self, input_shape):
        self.denses = [
            KL.Dense(1024,
                     kernel_initializer=initializers.VarianceScaling(),
                     kernel_regularizer=self._kernel_regularizer,
                     activation='relu') for _ in range(2)
        ]
        super().build(input_shape)

    def call(self, inputs):
        """Build the computational graph of the fast RCNN HEAD.

        Arguments:

        *inputs*: A list with the following schema:

        1. *pyramid*: A List of tensors the output of the pyramid
        2. *anchors*: A tensor of shape [batch_size, num_boxes, (y_min, x_min, y_max, x_max)]
        3. *ground_truths*: If the training is true, a dict with

        ```python
        ground_truths = {
            BoxField.BOXES:
                tf.constant([[[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 3, 3], [0, 0, 0, 0]]], tf.float32),
            BoxField.LABELS:
                tf.constant([[2,1], [2, 0]], tf.int32),
            BoxField.WEIGHTS:
                tf.constant([[1, 0], [1, 1]], tf.float32),
            BoxField.NUM_BOXES:
                tf.constant([2, 1], tf.int32)
        }
        ```

        where `NUM_BOXES` allows to remove the padding created by tf.Data.

        - *training*: Boolean

        Returns:


        - *classification_pred*: A logit Tensor of shape [batch_size, num_boxes, num_classes]
        - *localization_pred*: A Tensor of shape [batch_size, num_boxes, 4 * (num_classes - 1)]
        - *anchors*: A Tensor of shape [batch_size, num_boxes, 4]

        Raw predictions from the FastRCNN head you can post_process them using:

        ```python
        from kerod.model.post_processing import post_process_fast_rcnn_boxes

        outputs = post_process_fast_rcnn_boxes(classification_pred, localization_pred, anchors,
                                    images_information, num_classes)
        ```

        where `images_information` is provided as input of your model and `num_classes` includes
        the background.
        """
        # Remove P6
        pyramid = inputs[0][:-1]
        anchors = inputs[1]

        # We can compute the original image shape regarding
        # TODO compute it more automatically without knowing that the last layer is stride 32
        image_shape = tf.cast(tf.shape(pyramid[-1])[1:3] * 32, dtype=self._compute_dtype)
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
        return classification_pred, localization_pred

    def sample_boxes(self,
                     anchors: tf.Tensor,
                     ground_truths: Dict[str, tf.Tensor],
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
                tf.constant([[8, 0], [1, 0]], tf.float32),
            BoxField.WEIGHTS:
                tf.constant([[1, 0], [1, 1]], tf.float32),
            BoxField.NUM_BOXES:
                tf.constant([[2], [1]], tf.int32)
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

        ground_truths = {
            # We add one because the background is not counted in ground_truths[BoxField.LABELS]
            BoxField.LABELS:
                ground_truths[BoxField.LABELS] + 1,
            BoxField.BOXES:
                ground_truths[BoxField.BOXES],
            BoxField.WEIGHTS:
                ground_truths[BoxField.WEIGHTS],
            BoxField.NUM_BOXES:
                ground_truths[BoxField.NUM_BOXES]
        }
        y_true, weights = self.target_assigner.assign(anchors, ground_truths)

        labels = y_true[LossField.CLASSIFICATION] > 0
        sample_idx = batch_sample_balanced_positive_negative(
            weights[LossField.CLASSIFICATION],
            sampling_size,
            labels,
            positive_fraction=sampling_positive_ratio,
            dtype=self._compute_dtype)

        weights[LossField.CLASSIFICATION] = tf.multiply(sample_idx,
                                                        weights[LossField.CLASSIFICATION])
        weights[LossField.LOCALIZATION] = tf.multiply(sample_idx, weights[LossField.LOCALIZATION])

        selected_boxes_idx = tf.where(sample_idx == 1)

        batch_size = tf.shape(sample_idx)[0]

        # Extract the selected anchors corresponding anchors
        # tf.gather_nd collaps the batch_together so we reshape with the proper batch_size
        anchors = tf.reshape(tf.gather_nd(anchors, selected_boxes_idx), (batch_size, -1, 4))

        y_true[LossField.LOCALIZATION] = tf.reshape(
            tf.gather_nd(y_true[LossField.LOCALIZATION], selected_boxes_idx), (batch_size, -1, 4))

        y_true[LossField.CLASSIFICATION] = tf.reshape(
            tf.gather_nd(y_true[LossField.CLASSIFICATION], selected_boxes_idx), (batch_size, -1))

        for key in y_true.keys():
            weights[key] = tf.reshape(tf.gather_nd(weights[key], selected_boxes_idx),
                                      (batch_size, -1))
            weights[key] = tf.stop_gradient(weights[key])
            y_true[key] = tf.stop_gradient(y_true[key])
        return y_true, weights, anchors

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
        accuracy, fg_accuracy, false_negative = compute_fast_rcnn_metrics(
            y_true[LossField.CLASSIFICATION], classification_pred)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        self.add_metric(fg_accuracy, name='fg_accuracy', aggregation='mean')
        self.add_metric(false_negative, name='false_negative', aggregation='mean')

        # y_true[LossField.CLASSIFICATION] is just 1 and 0 we are using it as mask to extract
        # the corresponding target anchors
        batch_size = tf.shape(classification_pred)[0]
        # We create a boolean mask to extract the desired localization prediction to compute
        # the loss
        one_hot_targets = tf.one_hot(y_true[LossField.CLASSIFICATION],
                                     self._num_classes,
                                     dtype=tf.int8)
        one_hot_targets = tf.reshape(one_hot_targets, [-1])

        # We need to insert a fake background classes at the position 0
        localization_pred = tf.pad(localization_pred, [[0, 0], [0, 0], [4, 0]])
        localization_pred = tf.reshape(localization_pred, [-1, 4])

        extracted_localization_pred = tf.boolean_mask(localization_pred, one_hot_targets > 0)
        extracted_localization_pred = tf.reshape(extracted_localization_pred, (batch_size, -1, 4))

        y_pred = {
            LossField.CLASSIFICATION: classification_pred,
            LossField.LOCALIZATION: extracted_localization_pred
        }

        return self.compute_losses(y_true, y_pred, weights)


def compute_fast_rcnn_metrics(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Useful metrics that allows to track how behave the training of the fast rcnn head.

    TODO Handle the sample_weights

    Arguments:

    - *y_true*: A one-hot encoded vector with shape [batch_size, num_sample_anchors, num_classes]
    - *y_pred*: A tensor with shape [batch_size, num_sample_anchors, num_classes],
    representing the classification logits.

    Returns:

    - *accuracy*: A scalar tensor representing the accuracy with the background classes included
    - *fg_accuracy*: A scalar tensor representing the accuracy without the background classes included
    - *false_negative*: A scalar tensor representing the ratio of boxes predicted as background instead of
    their respective class among the foreground example to predict.

    Warning:

    This function should be used if the ground_truths have been added to the RoIs.
    It won't work if the there are no foreground ground_truths in the sample_boxes which isn't
    possible if they have been added.
    """
    # compute usefull metrics
    #Even if the softmax has not been applyed the argmax can be usefull
    prediction = tf.argmax(y_pred, axis=-1, name='label_prediction', output_type=tf.int32)
    correct = tf.cast(prediction == y_true, tf.float32)
    # The accuracy allows to determine if the models perform well (background included)
    accuracy = tf.reduce_mean(correct, name='accuracy')

    # Compute accuracy and false negative on all the foreground boxes
    fg_inds = tf.where(y_true > 0)
    num_fg = tf.shape(fg_inds)[0]
    fg_label_pred = tf.argmax(tf.gather_nd(y_pred, fg_inds), axis=-1)
    num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int32), name='num_zero')

    # Number of example predicted as background instead of one of our classes
    false_negative = tf.cast(tf.truediv(num_zero, num_fg), tf.float32, name='false_negative')

    fg_accuracy = tf.reduce_mean(tf.gather_nd(correct, fg_inds), name='fg_accuracy')

    return accuracy, fg_accuracy, false_negative
