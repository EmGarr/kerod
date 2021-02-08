from typing import List

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from kerod.core.box_coder import encode_boxes_faster_rcnn
from kerod.core.losses import L1Loss
from kerod.core.matcher import Matcher
from kerod.core.sampling_ops import batch_sample_balanced_positive_negative
from kerod.core.similarity import IoUSimilarity
from kerod.core.standard_fields import BoxField
from kerod.core.target_assigner import TargetAssigner
from kerod.layers import Anchors
from kerod.layers.detection.abstract_detection_head import \
    AbstractDetectionHead
from kerod.utils.documentation import remove_unwanted_doc

__pdoc__ = {}
SAMPLING_SIZE = 256
SAMPLING_POSITIVE_RATIO = 0.5


class RegionProposalNetwork(AbstractDetectionHead):
    """It has been introduced in the [Faster R-CNN paper](https://arxiv.org/abs/1506.01497) and
    use the parameters from [Feature Pyramidal Networks for Object Detection](https://arxiv.org/abs/1612.03144).

    Arguments:
        anchor_ratios: The ratios are the different shapes that you want to apply on your anchors.
            e.g: (0.5, 1, 2)

    Call arguments:
        inputs: A List of tensors the output of the pyramid

    Call returns:
        Tuple:
            - `localization_pred`: A list of logits 3-D tensor
            of shape [batch_size, num_anchors, 4]
            - `classification_pred`: A lost of logits 3-D tensor
            of shape [batch_size, num_anchors, 2]
            - `anchors`: A list of tensors of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]

    """

    def __init__(self, anchor_ratios=(0.5, 1, 2), **kwargs):
        super().__init__(
            2,
            SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                          from_logits=True),
            L1Loss(reduction=tf.keras.losses.Reduction.NONE),
            multiples=len(anchor_ratios),
            kernel_initializer_classification_head=initializers.RandomNormal(stddev=0.01),
            kernel_initializer_box_prediction_head=initializers.RandomNormal(stddev=0.01),
            **kwargs)

        #Force each ground_truths to match to at least one anchor
        matcher = Matcher([0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True)
        self.target_assigner = TargetAssigner(IoUSimilarity(),
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
            inputs: A tensor of  shape [batch_size, width, height, channel]

        Returns:
            Tuple:
                classification_head: a tensor of shape [batch_size, num_anchors, 2]
                localization_head: a tensor of shape [batch_size, num_anchors, 4]
        """

        batch_size = tf.shape(inputs)[0]
        rpn_conv2d = self.rpn_conv2d(inputs)
        classification_head, localization_head = self.build_detection_head(rpn_conv2d)
        classification_head = tf.reshape(classification_head, (batch_size, -1, 2))
        localization_head = tf.reshape(localization_head, (batch_size, -1, 4))
        return classification_head, localization_head

    def call(self, inputs: List[tf.Tensor]):
        """Create the computation graph for the rpn inference

        Arguments:
            inputs: A List of tensors the output of the pyramid

        Returns:
            Tuple:
                - `localization_pred`: A list of logits 3-D tensor
                of shape [batch_size, num_anchors, 4]
                - `classification_pred`: A lost of logits 3-D tensor
                of shape [batch_size, num_anchors, 2]
                - `anchors`: A list of tensors of shape [batch_size, num_anchors, (y_min, x_min, y_max, x_max)]
        """
        anchors = [anchors(tensor) for tensor, anchors in zip(inputs, self._anchors)]

        rpn_predictions = [self.build_rpn_head(tensor) for tensor in inputs]
        localization_pred = [prediction[1] for prediction in rpn_predictions]
        classification_pred = [prediction[0] for prediction in rpn_predictions]

        return localization_pred, classification_pred, anchors

    def compute_loss(self, localization_pred, classification_pred, anchors, ground_truths):
        """Compute the loss

        Arguments:
            localization_pred: A list of tensors of shape [batch_size, num_anchors, 4].
            classification_pred: A list of tensors of shape [batch_size, num_anchors, 2]
            anchors: A list of tensors of shape [num_anchors, (y_min, x_min, y_max, x_max)]
            ground_truths: A dict
                - `BoxField.LABELS`: A 3-D tensor of shape [batch_size, num_gt, num_classes],
                - `BoxField.BOXES`: A 3-D tensor of shape [batch_size, num_gt, (y1, x1, y2, x2)]
                - `BoxField.LABELS`: A 3-D tensor of int32 and shape [batch_size, num_gt]
                - `BoxField.WEIGHTS`: A 3-D tensor of float and shape [batch_size, num_gt]
                - `BoxField.NUM_BOXES`: A 2-D tensor of int32 and shape [batch_size, 1]
                    which allows to remove the padding created by tf.Data.
                    Example: if batch_size=2 and this field equal tf.constant([[2], [1]], tf.int32)
                    then my second box has a padding of 1

        Returns:
            Tuple:
                - `classification_loss`: A scalar in tf.float32
                - `localization_loss`: A scalar in tf.float32
        """
        localization_pred = tf.concat(localization_pred, 1)
        classification_pred = tf.concat(classification_pred, 1)
        anchors = tf.concat(anchors, 0)

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
        # anchors are deterministic duplicate them to create a batch
        anchors = tf.tile(anchors[None], (tf.shape(ground_truths[BoxField.BOXES])[0], 1, 1))
        y_true, weights = self.target_assigner.assign({BoxField.BOXES: anchors}, ground_truths)
        y_true[BoxField.LABELS] = tf.minimum(y_true[BoxField.LABELS], 1)

        ## Compute metrics
        recall = compute_rpn_metrics(y_true[BoxField.LABELS], classification_pred,
                                     weights[BoxField.LABELS])
        self.add_metric(recall, name='rpn_recall', aggregation='mean')

        # All the boxes which are not -1 can be sampled
        labels = y_true[BoxField.LABELS] > 0
        sample_idx = batch_sample_balanced_positive_negative(
            weights[BoxField.LABELS],
            SAMPLING_SIZE,
            labels,
            positive_fraction=SAMPLING_POSITIVE_RATIO,
            dtype=self._compute_dtype)

        weights[BoxField.LABELS] = sample_idx * weights[BoxField.LABELS]
        weights[BoxField.BOXES] = sample_idx * weights[BoxField.BOXES]

        y_pred = {BoxField.LABELS: classification_pred, BoxField.BOXES: localization_pred}

        return self.compute_losses(y_true, y_pred, weights)

    def get_config(self):
        base_config = super().get_config()
        base_config['anchor_ratios'] = self._anchor_ratios
        return base_config


def compute_rpn_metrics(y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor):
    """Useful metrics that allows to track how behave the training of the rpn head.

    Arguments:
        y_true: A tensor vector with shape [batch_size, num_anchors] where 0 = background and
            1 = foreground.
        y_pred: A tensor of shape [batch_size, num_anchors, 2],
            representing the classification logits.
        weights: A tensor of shape [batch_size, num_anchors] where weights should

    Returns:
        tf.Tensor: Recall, among all the boxes that we had to find how much did we found.
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


remove_unwanted_doc(RegionProposalNetwork, __pdoc__)
