import tensorflow as tf

from tensorflow.python.keras.engine import data_adapter, training

from kerod.utils.training import apply_kernel_regularization
from kerod.model.backbone.resnet import ResNet50, ResNet50PytorchStyle
from kerod.model.backbone.fpn import FPN
from kerod.model.detection.fast_rcnn import FastRCNN
from kerod.model.detection.rpn import RegionProposalNetwork
from kerod.model.post_processing import post_process_fast_rcnn_boxes
from kerod.core.standard_fields import DatasetField, BoxField

from kerod.model.post_processing import post_process_rpn


class FasterRcnnFPN(tf.keras.Model):
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

    def __init__(self, num_classes, backbone, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.l2 = tf.keras.regularizers.l2(1e-4)

        self.backbone = backbone
        self.fpn = FPN(kernel_regularizer=self.l2)
        self.rpn = RegionProposalNetwork(kernel_regularizer=self.l2)
        self.fast_rcnn = FastRCNN(self.num_classes + 1, kernel_regularizer=self.l2)
        # FasterRcnn cannot handle batch of unknown shape in training.
        # It will raise an error if you save.
        # serving false allows to bypass the check
        self._serving = False

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
        images = inputs[DatasetField.IMAGES]
        images_information = inputs[DatasetField.IMAGES_INFO]

        # The preprocessing dedicated to the backbone is done inside the model.
        x = self.backbone(images)
        pyramid = self.fpn(x)

        rpn_loc_pred_per_lvl, rpn_cls_pred_per_lvl, anchors_per_lvl = self.rpn(pyramid)

        if training and not self._serving:
            apply_kernel_regularization(self.l2, self.backbone)
            loss_rpn = self.rpn.compute_loss(rpn_loc_pred_per_lvl, rpn_cls_pred_per_lvl,
                                             anchors_per_lvl, inputs['ground_truths'])

        num_boxes = 2000 if training else 1000
        rois = post_process_rpn(rpn_cls_pred_per_lvl,
                                rpn_loc_pred_per_lvl,
                                anchors_per_lvl,
                                images_information,
                                pre_nms_topk_per_lvl=num_boxes,
                                post_nms_topk=num_boxes)

        if training and not self._serving:
            ground_truths = inputs['ground_truths']
            # Include the ground_truths as RoIs for the training
            rois = tf.concat([rois, ground_truths[BoxField.BOXES]], axis=1)
            # Sample the boxes needed for inference
            y_true, weights, rois = self.fast_rcnn.sample_boxes(rois, ground_truths)

        classification_pred, localization_pred = self.fast_rcnn([pyramid, rois])

        if training and not self._serving:
            loss_fast_rcnn = self.fast_rcnn.compute_loss(y_true, weights, classification_pred,
                                                         localization_pred)

        classification_pred = tf.nn.softmax(classification_pred)

        return classification_pred, localization_pred, rois

    def train_step(self, data):
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            x['ground_truths'] = y
            y_pred = self(x, training=True)
            # All the losses are computed in the call. It's weird but it those the job
            # They are added automatically to self.losses
            loss = self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)

        training._minimize(self.distribute_strategy, tape, self.optimizer, loss,
                           self.trainable_variables)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, _ = data_adapter.unpack_x_y_sample_weight(data)

        # In our graph all the metrics are computed inside the call method
        # So we set training to True to benefit from those metrics
        # Of course there is no backpropagation at the test step
        x['ground_truths'] = y

        y_pred = self(x, training=True)

        loss = self.compiled_loss(None, y_pred, None, regularization_losses=self.losses)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        classification_pred, localization_pred, rois = self(x, training=False)

        return post_process_fast_rcnn_boxes(classification_pred, localization_pred, rois,
                                            x[DatasetField.IMAGES_INFO], self.num_classes + 1)

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        try:
            super().save(filepath,
                         overwrite=overwrite,
                         include_optimizer=include_optimizer,
                         save_format=save_format,
                         signatures=signatures,
                         options=options)
        except Exception as e:
            raise Exception(
                'Saving does not work with dynamic inputs the ground_truths are injected in the inputs. '
                'Please use export_model method instead to bypass this error.')

    def export_model(self, filepath):
        """Allow to bypass the save_model behavior the graph in serving mode.
        Currently, the issue is that in training the ground_truths are passed to the call method but
        not in inference. For the serving only the `images` and `images_information` are defined.
        It means the inputs link to the ground_truths won't be defined in serving. However, in tensorflow
        when the `training` arguments is defined int the method `call`, `tf.save_model.save` method
        performs a check on the graph for training=False and training=True.
        However, we don't want this check to be perform because our ground_truths inputs aren't defined.
        """
        self._serving = True
        tf.saved_model.save(self, filepath)
        self._serving = False


class FasterRcnnFPNResnet50Caffe(FasterRcnnFPN):

    def __init__(self, num_classes, **kwargs):
        resnet = ResNet50(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, **kwargs)


class FasterRcnnFPNResnet50Pytorch(FasterRcnnFPN):

    def __init__(self, num_classes, **kwargs):
        resnet = ResNet50PytorchStyle(input_shape=[None, None, 3], weights='imagenet')
        super().__init__(num_classes, resnet, **kwargs)
