import tensorflow as tf

from od.model import factory
from od.model.backbone.fpn import Pyramid
from od.model.backbone.resnet import ResNet50
from od.model.detection.fast_rcnn import FastRCNN
from od.model.detection.rpn import RegionProposalNetwork


def build_fpn_resnet50_faster_rcnn(num_classes: int, batch_size: int) -> tf.keras.Model:
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
    - *batch_size*: The batch_size needs to specify for the training mode.

    Return:

    A FPN Resnet50 Faster RCNN

    Warnings:

    The inputs of the model are flatten. It is a list defined by the following keys:
    [DatasetField.IMAGES, DatasetField.IMAGES_INFO, BoxField.BOXES, BoxField.LABELS,
    BoxField.WEIGHTS, BoxField.NUM_BOXES]
    """
    images, images_information, ground_truths = factory.build_input_layers(training=True,
                                                                     batch_size=batch_size)

    resnet = ResNet50(input_tensor=images, weights='imagenet')
    pyramid = Pyramid()(resnet.outputs)
    rois, _ = RegionProposalNetwork()([pyramid, images_information, ground_truths], training=True)

    outputs = FastRCNN(num_classes + 1)([pyramid, rois, images_information, ground_truths],
                                        training=True)

    model_faster_rcnn = tf.keras.Model(inputs=[images, images_information, ground_truths],
                                       outputs=outputs)
    return model_faster_rcnn
