import tensorflow as tf

from kerod.model.faster_rcnn import FasterRcnnFPNResnet50
from kerod.utils.training import (freeze_batch_normalization, freeze_layers_before)


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a FasterRcnnFPNResnet50 with all the `tf.keras.layers.BatchNormalization` frozen and
    all the layers before second residual block.

    Argument:

    - *num_classes*: Number of classes of your model. Do not include the background class.

    Return:

    A `keras.Model` instance.
    """
    model = FasterRcnnFPNResnet50(num_classes)
    freeze_batch_normalization(model.resnet)
    freeze_layers_before(model.resnet, 'conv2_block3_out')
    return model
