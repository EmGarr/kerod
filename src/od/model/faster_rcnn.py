import tensorflow as tf

from od.model.backbone.fpn import Pyramid
from od.model.backbone.resnet import Resnet50
from od.model.detection.fast_rcnn import FastRCNN
from od.model.detection.rpn import RegionProposalNetwork
from od.model.post_processing import post_process_fast_rcnn_boxes
from od.core.standard_fields import DatasetField


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

    TODO check if Warnings:

    The inputs of the model are flatten. It is a list defined by the following keys:
    [DatasetField.IMAGES, DatasetField.IMAGES_INFO, BoxField.BOXES, BoxField.LABELS,
    BoxField.WEIGHTS, BoxField.NUM_BOXES] in training and [DatasetField.IMAGES, DatasetField.IMAGES_INFO]
    in inference.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        # Seems that kernel regularizer make the network diverge
        # self.resnet = Resnet50(kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.resnet = Resnet50()
        self.fpn = Pyramid()
        self.rpn = RegionProposalNetwork()
        self.fast_rcnn = FastRCNN(num_classes + 1)

    def call(self, inputs, training=None):
        images = inputs[DatasetField.IMAGES]
        images_information = inputs[DatasetField.IMAGES_INFO]
        x = self.resnet(images)
        pyramid = self.fpn(x)
        if training:
            # During the saving operation the check_mutation method raise
            # an error if we modify the inputs dictionary
            ground_truths = {
                key: value
                for key, value in inputs.items()
                if key not in [DatasetField.IMAGES, DatasetField.IMAGES_INFO]
            }
            rois, _ = self.rpn([pyramid, images_information, ground_truths], training=training)
            outputs = self.fast_rcnn([pyramid, rois, ground_truths], training=training)
        else:
            rois, _ = self.rpn([pyramid, images_information], training=training)
            outputs = self.fast_rcnn([pyramid, rois], training=training)
        return outputs

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], name=DatasetField.IMAGES),
        tf.TensorSpec(shape=[None, 2], name=DatasetField.IMAGES_INFO)
    ])
    def serve(self, images, images_information):
        """In this method we apply a postprocessing on the outputs of the model"""
        inputs = {DatasetField.IMAGES: images, DatasetField.IMAGES_INFO: images_information}
        classification_pred, localization_pred, anchors = self(inputs, training=False)
        return post_process_fast_rcnn_boxes(classification_pred, localization_pred, anchors,
                                            images_information, self.num_classes + 1)

    def export_for_serving(self, path, **kwargs):
        """Allow to bypass the save_model behavior the graph in serving mode.
        Currently, the issue is that in training the ground_truths are passed to the call method but
        not in inference. For the serving only the `images` and `images_information` are defined.
        It means the inputs link to the ground_truths won't be defined in serving. However, in tensorflow
        when the `training` arguments is defined int the method `call`, `tf.save_model.save` method
        performs a check on the graph for training=False and training=True.
        However, we don't want this check to be perform because our ground_truths inputs aren't defined.
        """
        self.rpn.serving = True
        self.fast_rcnn.serving = True
        self.save(path, signatures=self.serve.get_concrete_function(), **kwargs)

        # Make the state of the network unchanged after the call of the function
        self.rpn.serving = False
        self.fast_rcnn.serving = False
