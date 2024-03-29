from typing import Dict

import tensorflow as tf
import tensorflow.keras.layers as KL
from kerod.core.standard_fields import BoxField
from kerod.utils.documentation import remove_unwanted_doc
from tensorflow.keras import initializers

__pdoc__ = {}


class AbstractDetectionHead(KL.Layer):
    """Abstract object detector. It encapsulates the main functions of an object detector.

    Arguments:
        num_classes: Number of classes of the classification head (e.g: Your n classes +
            the background class)
        classification_loss: An object tf.keras.losses usually CategoricalCrossentropy.
            This object should have a reduction value to None and the parameter from_y_pred to True.
        localization_loss: An object tf.keras.losses usually CategoricalCrossentropy.
            This object should have a reduction value to None and the parameter from_y_pred to True.
        segmentation_loss: An object tf.keras.losses usually CategoricalCrossentropy.
            This object should have a reduction value to None and the parameter from_y_pred to True.
        classification_loss_weight: A float 32 representing the weight of the loss in the
            total loss.
        localization_loss_weight: A float 32 representing the weight of the loss in the
            total loss.
        segmentation_loss_weight: A float 32 representing the weight of the loss in the
            total loss.
        multiples: How many time will you replicate the output of the head.
            For a rpn multiples can be the number of anchors.
            For a fast_rcnn multiples is 1 we just want the number of classes
        kernel_initializer_classification_head: Initializer for the `kernel` weights matrix of
            the classification head (see [initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)).
        kernel_initializer_box_prediction_head: Initializer for the `kernel` weights matrix of
            the box prediction head (see [initializers](https://www.tensorflow.org/api_docs/python/tf/keras/initializers)).
        kernel_regularizer: Regularizer function applied to the kernel weights matrix
            ([see keras.regularizers](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers)).
        use_mask: Boolean define if the segmentation_head will be used.
    """

    def __init__(self,
                 num_classes,
                 classification_loss,
                 localization_loss,
                 segmentation_loss=None,
                 classification_loss_weight=1.0,
                 localization_loss_weight=1.0,
                 segmentation_loss_weight=1.0,
                 multiples=1,
                 kernel_initializer_classification_head=None,
                 kernel_initializer_box_prediction_head=None,
                 kernel_regularizer=None,
                 use_mask=False,
                 **kwargs):

        super().__init__(**kwargs)

        self._num_classes = num_classes
        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._multiples = multiples
        self._kernel_initializer_classification_head = kernel_initializer_classification_head
        self._kernel_initializer_box_prediction_head = kernel_initializer_box_prediction_head
        self._kernel_regularizer = kernel_regularizer
        self._use_mask = use_mask

        if self._use_mask:
            self._segmentation_loss_weight = segmentation_loss_weight
            self._segmentation_loss = segmentation_loss

    def build(self, input_shape):
        self._conv_classification_head = KL.Conv2D(
            self._multiples * self._num_classes, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_classification_head,
            kernel_regularizer=self._kernel_regularizer,
            name=f'{self.name}classification_head')
        self._conv_box_prediction_head = KL.Conv2D(
            (self._num_classes - 1) * self._multiples * 4, (1, 1),
            padding='valid',
            activation=None,
            kernel_initializer=self._kernel_initializer_box_prediction_head,
            kernel_regularizer=self._kernel_regularizer,
            name=f'{self.name}box_prediction_head')

        if self._use_mask:
            self._segmentation_layers = [
                KL.Conv2D(256, (3, 3),
                          padding='valid',
                          activation='relu',
                          kernel_initializer=initializers.VarianceScaling(scale=2., mode='fan_out'),
                          kernel_regularizer=self._kernel_regularizer),
                KL.Conv2DTranspose(256, (2, 2),
                                   strides=(2, 2),
                                   padding='valid',
                                   activation='relu',
                                   kernel_initializer=initializers.VarianceScaling(scale=2.,
                                                                                   mode='fan_out'),
                                   kernel_regularizer=self._kernel_regularizer),
                KL.Conv2D(self._num_classes, (3, 3),
                          padding='valid',
                          activation='relu',
                          kernel_initializer=initializers.VarianceScaling(scale=2., mode='fan_out'),
                          kernel_regularizer=self._kernel_regularizer)
            ]

        super().build(input_shape)

    def build_segmentation_head(self, inputs):
        """Build the detection head

        Arguments:
            inputs: A tensor of float and shape [N, H, W, C]

        Returns:
            tf.Tensor: A tensor and shape [N, H*2, W*2, num_classes - 1]
        """

        x = inputs
        for layer in self._segmentation_layers:
            x = layer(x)
        return x

    def build_detection_head(self, inputs):
        """ Build a detection head composed of a classification and box_detection.

        Arguments:
            inputs: A tensor of shape [batch_size, H, W, C]

        Returns:
            Tuple:
                classification_head: a tensor of shape [batch_size, num_anchors, 2]
                localization_head: a tensor of shape [batch_size, num_anchors, 4]
        """
        classification_head = self._conv_classification_head(inputs)

        box_prediction_head = self._conv_box_prediction_head(inputs)

        return classification_head, box_prediction_head

    def compute_losses(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor],
                       weights: Dict[str, tf.Tensor]) -> dict:
        """Compute the losses of the object detection head.

        Each dictionary is composed of the same keys (classification, localization, segmentation)

        Arguments:
            y_pred: A dict of tensors of shape [N, nb_boxes, num_output].
            y_true: A dict of tensors of shape [N, nb_boxes, num_output].
            weights: A dict of tensors ofshape [N, nb_boxes, num_output].
                This tensor is composed of one hot vectors.

        Returns:
            dict : A dict of different losses
        """

        def _compute_loss(loss, loss_weight, target):
            losses = loss(tf.cast(y_true[target], tf.float32),
                          tf.cast(y_pred[target], tf.float32),
                          sample_weight=tf.cast(weights[target], tf.float32))
            return loss_weight * tf.reduce_mean(tf.reduce_sum(losses, axis=1) / normalizer)

        normalizer = tf.maximum(tf.reduce_sum(weights[BoxField.LABELS], axis=1), 1.0)
        normalizer = tf.cast(normalizer, tf.float32)

        classification_loss = _compute_loss(self._classification_loss,
                                            self._classification_loss_weight, BoxField.LABELS)

        self.add_metric(classification_loss,
                        name=f'{self.name}_classification_loss',
                        aggregation='mean')

        localization_loss = _compute_loss(self._localization_loss, self._localization_loss_weight,
                                          BoxField.BOXES)

        self.add_metric(localization_loss,
                        name=f'{self.name}_localization_loss',
                        aggregation='mean')

        self.add_loss([classification_loss, localization_loss])

        if self._use_mask:
            segmentation_loss = _compute_loss(self._segmentation_loss,
                                              self._segmentation_loss_weight, BoxField.MASKS)

            self.add_metric(segmentation_loss,
                            name=f'{self.name}_segmentation_loss',
                            aggregation='mean')
            self.add_loss(segmentation_loss)
            return {
                BoxField.LABELS: classification_loss,
                BoxField.BOXES: localization_loss,
                BoxField.MASKS: segmentation_loss
            }

        return {BoxField.LABELS: classification_loss, BoxField.BOXES: localization_loss}

    def get_config(self):
        base_config = super().get_config()
        base_config['num_classes'] = self._num_classes
        base_config['classification_loss_weight'] = self._classification_loss_weight
        base_config['localization_loss_weight'] = self._localization_loss_weight
        base_config['multiples'] = self._multiples
        base_config['use_mask'] = self._use_mask
        if self._use_mask:
            base_config['segmentation_loss_weight'] = self._segmentation_loss_weight
        return base_config


remove_unwanted_doc(AbstractDetectionHead, __pdoc__)
