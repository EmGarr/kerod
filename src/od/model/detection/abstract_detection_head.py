from abc import ABCMeta, abstractmethod

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.initializers import VarianceScaling


class AbstractDetectionHead(tfkl.Layer, metaclass=ABCMeta):

    def __init__(self,
                 name,
                 num_classes,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight=1.0,
                 localization_loss_weight=2.0,
                 multiples=1,
                 **kwargs
    ):
        """Abstract object detector. It encapsulates the main functions of an object detector.
        The build of the detection head, the segmentation head, the post_processing.

        Arguments:

        - *name*: The name of your object
        - ùnum_classesù: Number of classes of the classification head (e.g: Your n classes +
                the background class)
        - *classification_loss*: An object tf.keras.losses usually CategoricalCrossentropy.
                This object should have a reduction value to None and the parameter from_logits to True.
        - *localization_loss*: An object tf.keras.losses usually CategoricalCrossentropy.
                This object should have a reduction value to None and the parameter from_logits to True.
        - *classification_loss_weight*: A float 32 representing the weight of the loss in the
                total loss.
        - *localization_loss_weight*: A float 32 representing the weight of the loss in the
                total loss.
        - *multiples*: How many time will you replicate the output of the head.
                For a rpn multiples can be the number of anchors.
                For a fast_rcnn multiples is 1 we just want the number of classes

        """
        super().__init__(**kwargs)

        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight

        self._num_classes = num_classes
        self._conv_classification_head = tfkl.Conv2D(
            multiples * self._num_classes, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_classification_head,
            kernel_regularization=self._kernel_regularizer,
            name=f'{name}classification_head')
        self._conv_box_prediction_head = tfkl.Conv2D(
            (self._num_classes - 1) * multiples * 4, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_box_regression_head,
            kernel_regularization=self._kernel_regularizer,
            name=f'{name}box_prediction_head')

    @abstractmethod
    def post_processing(self, inputs):
        pass

    def build_segmentation_head(self, inputs, num_convs, dim=256):
        """Build the detection head

        Arguments:

        - *inputs*: A tensor EOFError(${1:args})$0 float32 and shape [N, H, W, C]
                num_convs:
        - *dim*: Default to 256. Is the channel size

        Returns:

        A tensor of float32 and shape [N, H*2, W*2, num_classes - 1]
        """

        layer = inputs
        for _ in range(num_convs):
            layer = tfkl.Conv2D(dim, (3, 3),
                                padding='valid',
                                activation='relu',
                                kernel_initializer=VarianceScaling(scale=2., mode='fan_out'))(layer)

        layer = tfkl.Conv2DTranspose(dim, (2, 2),
                                     strides=(2, 2),
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer=VarianceScaling(scale=2.,
                                                                        mode='fan_out'))(layer)

        return tfkl.Conv2D(self._num_classes, (3, 3),
                           padding='valid',
                           activation='relu',
                           kernel_initializer=VarianceScaling(scale=2., mode='fan_out'))(layer)

    def build_detection_head(self, inputs):
        """ Build a detection head composed 

        Arguments:

        - *inputs*: A tensor of float32 and shape [batch_size, H, W, C]
        """
        classification_head = self._conv_classification_head(inputs)

        box_prediction_head = self._conv_box_prediction_head(inputs)

        return classification_head, box_prediction_head

    def compute_losses(self, logits, targets, weights):
        """Compute the losses of the object detection head.
        Each dictionary is composed of the same key (classification, localization, segmentation)

        Arguments:

        - *logits*: A dict of tensors of type float32 of shape [N, nb_boxes, num_output].
        - *targets*: A dict of tensors of type float32 of shape [N, nb_boxes, num_output].
        - *weights*: A dict of tensors of type float32 of shape [N, nb_boxes, num_output].
                This tensor is composed of one hot vectors.
        - *classification_weights*: A tensor of type float32 of shape [N,nb_boxes, 1].

        Returns:

        *classification_loss*: A scalar of type float32
        """

        def _compute_loss(loss, loss_weight, target):
            losses = loss(logits[target], targets[target], weights=weights[target])
            return loss_weight * tf.reduce_mean(tf.reduce_sum(losses, axis=1) / normalizer)

        normalizer = tf.maximum(tf.reduce_sum(weights['classification'], axis=1), 1.0)

        classification_loss = _compute_loss(self._classification_loss,
                                            self._classification_loss_weights, 'classification')
        localization_loss = _compute_loss(self._localization_loss, self._localization_loss_weights,
                                          'localization')
