from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers, regularizers

from od.core.standard_fields import LossField


class AbstractDetectionHead(KL.Layer):
    """Abstract object detector. It encapsulates the main functions of an object detector.
        The build of the detection head, the segmentation head, the post_processing.

        Arguments:

        - *num_classes*: Number of classes of the classification head (e.g: Your n classes +
                the background class)
        - *target_assigner*: An object od.core.target_assigner.TargetAssigner
        - *classification_loss*: An object tf.keras.losses usually CategoricalCrossentropy.
                This object should have a reduction value to None and the parameter from_y_pred to True.
        - *localization_loss*: An object tf.keras.losses usually CategoricalCrossentropy.
                This object should have a reduction value to None and the parameter from_y_pred to True.
        - *classification_loss_weight*: A float 32 representing the weight of the loss in the
                total loss.
        - *localization_loss_weight*: A float 32 representing the weight of the loss in the
                total loss.
        - *multiples*: How many time will you replicate the output of the head.
                For a rpn multiples can be the number of anchors.
                For a fast_rcnn multiples is 1 we just want the number of classes
        - *kernel_initializer_classification_head*: Initializer for the `kernel` weights matrix of
        the classification head (see [initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)).
        - *kernel_initializer_box_prediction_head*: Initializer for the `kernel` weights matrix of
        the box prediction head (see [initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)).
        - *kernel_regularizer*: Regularizer function applied to
            the `kernel` weights matrix of every layers
            (see [regularizer](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)).
        """

    def __init__(self,
                 num_classes,
                 target_assigner,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight=1.0,
                 localization_loss_weight=2.0,
                 multiples=1,
                 kernel_initializer_classification_head=None,
                 kernel_initializer_box_prediction_head=None,
                 kernel_regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)

        self._num_classes = num_classes
        self.target_assigner = target_assigner

        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._kernel_initializer_classification_head = kernel_initializer_classification_head
        self._kernel_initializer_box_prediction_head = kernel_initializer_box_prediction_head

        if kernel_regularizer is None:
            self._kernel_regularizer = regularizers.l2(0.0005)
        else:
            self._kernel_regularizer = regularizers.get()

        self._conv_classification_head = KL.Conv2D(
            multiples * self._num_classes, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_classification_head,
            kernel_regularizer=self._kernel_regularizer,
            name=f'{self.name}classification_head')
        self._conv_box_prediction_head = KL.Conv2D(
            (self._num_classes - 1) * multiples * 4, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_box_prediction_head,
            kernel_regularizer=self._kernel_regularizer,
            name=f'{self.name}box_prediction_head')

    def build_segmentation_head(self, inputs, num_convs, dim=256):
        """Build the detection head

        Arguments:

        - *inputs*: A tensor of  shape [N, H, W, C]
                num_convs:
        - *dim*: Default to 256. Is the channel size

        Returns:

        A tensor and shape [N, H*2, W*2, num_classes - 1]
        """

        layer = inputs
        for _ in range(num_convs):
            layer = KL.Conv2D(dim, (3, 3),
                                padding='valid',
                                activation='relu',
                                kernel_initializer=initializers.VarianceScaling(scale=2.,
                                                                                mode='fan_out'),
                                kernel_regularizer=self._kernel_regularizer)(layer)

        layer = KL.Conv2DTranspose(dim, (2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer=initializers.VarianceScaling(
                                         scale=2., mode='fan_out'),
                                     kernel_regularizer=self._kernel_regularizer)(layer)

        return KL.Conv2D(self._num_classes, (3, 3),
                           padding='valid',
                           activation='relu',
                           kernel_initializer=initializers.VarianceScaling(scale=2.,
                                                                           mode='fan_out'),
                           kernel_regularizer=self._kernel_regularizer)(layer)

    def build_detection_head(self, inputs):
        """ Build a detection head composed 

        Arguments:

        - *inputs*: A tensor of shape [batch_size, H, W, C]
        """
        classification_head = self._conv_classification_head(inputs)

        box_prediction_head = self._conv_box_prediction_head(inputs)

        return classification_head, box_prediction_head

    def compute_losses(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor],
                       weights: Dict[str, tf.Tensor]):
        """Compute the losses of the object detection head.
        Each dictionary is composed of the same key (classification, localization, segmentation)

        Arguments:

        - *y_pred*: A dict of tensors of shape [N, nb_boxes, num_output].
        - *y_true*: A dict of tensors of shape [N, nb_boxes, num_output].
        - *weights*: A dict of tensors ofshape [N, nb_boxes, num_output].
                This tensor is composed of one hot vectors.

        Returns:

        A scalar
        """

        def _compute_loss(loss, loss_weight, target):
            losses = loss(y_true[target], y_pred[target], sample_weight=weights[target])
            return loss_weight * tf.reduce_mean(tf.reduce_sum(losses, axis=1) / normalizer)

        normalizer = tf.maximum(tf.reduce_sum(weights[LossField.CLASSIFICATION], axis=1), 1.0)

        classification_loss = _compute_loss(self._classification_loss,
                                            self._classification_loss_weight,
                                            LossField.CLASSIFICATION)
        tf.summary.scalar(f'{self.name}_classification_loss', classification_loss)

        localization_loss = _compute_loss(self._localization_loss, self._localization_loss_weight,
                                          LossField.LOCALIZATION)
        tf.summary.scalar(f'{self.name}_localization_loss', classification_loss)

        return classification_loss, localization_loss
