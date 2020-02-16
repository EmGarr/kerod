import tensorflow as tf

from od.model import factory
from od.model.backbone.fpn import Pyramid
from od.model.backbone.resnet import ResNet50
from od.model.detection.fast_rcnn import FastRCNN
from od.model.detection.rpn import RegionProposalNetwork
from od.model.post_processing import post_process_fast_rcnn_boxes


def build_fpn_resnet50_faster_rcnn(num_classes: int,
                                   batch_size: int,
                                   training=True) -> tf.keras.Model:
    """Build a FPN Resnet 50 Faster RCNN network ready to use for training. 

    You can use it as follow:

    ```python
    model_faster_rcnn = build_fpn_resnet50_faster_rcnn(80, 1) 

    base_lr = 0.1
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model_faster_rcnn.compile(optimizer=optimizer, loss=None)

    model_faster_rcnn.fit(ds_train, validation_data=ds_test, epochs=11,)
    ```

    Arguments:

    - *num_classes*: The number of classes of your dataset
    (**do not include the background class** it is handle for you)
    - *batch_size*: The batch_size needs to specify for the training mode. In inference it should be
    set to None.

    Return:

    A FPN Resnet50 Faster RCNN

    Warnings:

    The inputs of the model are flatten. It is a list defined by the following keys:
    [DatasetField.IMAGES, DatasetField.IMAGES_INFO, BoxField.BOXES, BoxField.LABELS,
    BoxField.WEIGHTS, BoxField.NUM_BOXES] in training and [DatasetField.IMAGES, DatasetField.IMAGES_INFO]
    in inference.
    """
    if training:
        images, images_information, ground_truths = factory.build_input_layers(
            training=True, batch_size=batch_size)
    else:
        images, images_information = factory.build_input_layers(training=False)

    resnet = ResNet50(input_tensor=images, weights='imagenet')
    pyramid = Pyramid()(resnet.outputs)

    if training:
        rois, _ = RegionProposalNetwork()([pyramid, images_information, ground_truths],
                                          training=True)
        outputs = FastRCNN(num_classes + 1)([pyramid, rois, ground_truths], training=True)
        inputs = [images, images_information, ground_truths]
    else:
        rois, _ = RegionProposalNetwork(serving=True)([pyramid, images_information], training=False)
        classification_pred, localization_pred, anchors = FastRCNN(num_classes + 1,
                                                                   serving=True)([pyramid, rois],
                                                                                 training=False)
        outputs = post_process_fast_rcnn_boxes(classification_pred, localization_pred, anchors,
                                               images_information, num_classes + 1)

        inputs = [images, images_information]

    return tf.keras.Model(inputs=inputs, outputs=outputs)
