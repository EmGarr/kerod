import tensorflow as tf
import tensorflow.keras.layers as tfkl
from od.model.detection.abstract_detection_head import AbstractDetectionHead
from od.core.anchor_generator import generate_anchors


class RegionProposalNetwork(AbstractDetectionHead):

    def __init__(self, anchor_ratios=(0.5, 1, 2)):
        super(RegionProposalNetwork).__init__('rpn', 2, multiples=len(anchor_ratios))
        self._anchor_strides = (4, 8, 16, 32, 64)
        self._anchor_ratios = anchor_ratios
        self.rpn_conv2d = tfkl.Conv2D(
            512, (3, 3),
            padding='same',
            kernel_initializer=self._kernel_initializer_classification_head,
            kernel_regularization=self._kernel_regularizer)

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

    def call(self, inputs: list):
        if len(inputs) != len(self._anchor_areas):
            raise Exception("Len anchor_areas should match the length of the inputs")
        rpn_predictions = [self.build_rpn_head(tensor) for tensor in inputs]
        rpn_anchors = []
        for tensor, anchor_stride in zip(inputs, self._anchor_strides):
            anchors = generate_anchors(anchor_stride, tf.constant([8], tf.float32),
                                       tf.constant(self._anchor_ratios, tf.float32),
                                       tf.shape(tensor))
            rpn_anchors.append(anchors)
        localization_pred = tf.concat([prediction[1] for prediction in rpn_predictions], 1)
        classification_pred = tf.concat([prediction[0] for prediction in rpn_predictions], 1)
        rpn_anchors = tf.concat([anchors for anchors in rpn_anchors], 1)

        return self.post_processing((localization_pred, classification_pred, rpn_anchors))

    def post_processing(self, inputs):
        localization_pred, classification_pred, rpn_anchors = inputs
        return None
