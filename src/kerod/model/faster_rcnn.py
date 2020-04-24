import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter, training
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras import metrics

from kerod.model.backbone.fpn import Pyramid
from kerod.model.backbone.resnet import Resnet50
from kerod.model.detection.fast_rcnn import FastRCNN
from kerod.model.detection.rpn import RegionProposalNetwork
from kerod.model.post_processing import post_process_fast_rcnn_boxes
from kerod.core.standard_fields import DatasetField, BoxField

from kerod.model.post_processing import post_process_rpn


class FasterRcnnFPNResnet50(tf.keras.Model):
    """Build a FPN Resnet 50 Faster RCNN network ready to use for training.

    You can use it as follow:

    ```python
    model_faster_rcnn = FasterRcnnFPNResnet50(80)
    base_lr = 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model_faster_rcnn.compile(optimizer=optimizer, loss=None)
    model_faster_rcnn.fit(ds_train, validation_data=ds_test, epochs=11,)
    ```

    Arguments:

    - *num_classes*: The number of classes of your dataset
    (**do not include the background class** it is handle for you)
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        # Seems that kernel regularizer make the network diverge
        self.resnet = Resnet50(kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        # self.resnet = Resnet50()
        self.fpn = Pyramid()
        self.rpn = RegionProposalNetwork()
        self.fast_rcnn = FastRCNN(num_classes + 1)

        self._loss_metric = metrics.Mean(name='loss')  # Total loss.

    def call(self, inputs, training=None):
        """Perform an inference in training.

        Arguments:

        - *inputs*: A list with the following schema:

        1. *features*:
        1.1 *pyramid*: A List of tensors the output of the pyramid
        1.2 *image_informations*: A Tensor of shape [batch_size, (height, width)]
        The height and the width are without the padding.

        2.. *ground_truths*: If the training is true, a dict with

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

        - *training*: Is automatically set to `True` in train and test mode
        (normally test should be at false). Why? Through the call we the losses and the metrics
        of the rpn and fast_rcnn. They are automatically added with `add_loss` and `add_metrics`.
        In test we want to benefit from those and therefore we compute them. It is an inheritance
        from tensorflow 2.0 and 2.1 and I'll think to move them in a more traditional way inside the
        train_step and test_step. However for now this method benefit of the encapsulation of
        the `self.compiled_loss` method.

        Returns:

        - *classification_pred*: A Tensor of shape [batch_size, num_boxes, num_classes] representig
        the class probability.
        - *localization_pred*: A Tensor of shape [batch_size, num_boxes, 4 * (num_classes - 1)]
        - *anchors*: A Tensor of shape [batch_size, num_boxes, 4]
        """
        if training:
            inputs, ground_truths = inputs

        images = inputs[DatasetField.IMAGES]
        images_information = inputs[DatasetField.IMAGES_INFO]

        x = self.resnet(images)
        pyramid = self.fpn(x)

        localization_pred, classification_pred, anchors = self.rpn(pyramid)

        if training:
            self._arg_loss_rpn = [localization_pred, classification_pred, anchors, ground_truths]

        num_boxes = 2000 if training else 1000
        rois, _ = post_process_rpn(tf.nn.softmax(classification_pred),
                                   localization_pred,
                                   anchors,
                                   images_information,
                                   pre_nms_topk=num_boxes * len(pyramid),
                                   post_nms_topk=num_boxes)

        if training:
            # Include the ground_truths as RoIs for the training
            rois = tf.concat([rois, ground_truths[BoxField.BOXES]], axis=1)
            # Sample the boxes needed for inference
            y_true, weights, rois = self.fast_rcnn.sample_boxes(rois, ground_truths)

        classification_pred, localization_pred = self.fast_rcnn([pyramid, rois])

        if training:
            self._arg_loss_fast_rcnn = [y_true, weights, classification_pred, localization_pred]

        classification_pred = tf.nn.softmax(classification_pred)

        return classification_pred, localization_pred, rois

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        classification_pred, localization_pred, rois = self(x, training=False)

        return post_process_fast_rcnn_boxes(classification_pred, localization_pred, rois,
                                            x[DatasetField.IMAGES_INFO], self.num_classes + 1)

    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self([x, y], training=True)
            loss = self.compute_loss()

        training._minimize(self.distribute_strategy, tape, self.optimizer, loss,
                           self.trainable_variables)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        # In our graph all the metrics are computed inside the call method
        # So we set training to True to benefit from those metrics
        # Of course there is no backpropagation at the test step
        y_pred = self([x, y], training=True)

        loss = self.compute_loss()

        return {m.name: m.result() for m in self.metrics}

    def compute_loss(self):
        loss_values = []  # Used for gradient calculation.
        loss_metric_values = []  # Used for loss metric calculation.

        rpn_loss = self.rpn.compute_loss(*self._arg_loss_rpn)
        loss_values.append(losses_utils.scale_loss_for_distribution(rpn_loss))
        loss_metric_values.append(rpn_loss)

        fast_rcnn_loss = self.fast_rcnn.compute_loss(*self._arg_loss_fast_rcnn)
        loss_values.append(losses_utils.scale_loss_for_distribution(fast_rcnn_loss))
        loss_metric_values.append(fast_rcnn_loss)

        regularization_losses = losses_utils.cast_losses_to_common_dtype(self.losses)
        reg_loss = tf.math.add_n(regularization_losses)
        loss_metric_values.append(reg_loss)
        loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

        loss_metric_values = losses_utils.cast_losses_to_common_dtype(loss_metric_values)
        total_loss_metric_value = tf.math.add_n(loss_metric_values)
        self._loss_metric.update_state(total_loss_metric_value)

        loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
        total_loss = tf.math.add_n(loss_values)
        return total_loss
