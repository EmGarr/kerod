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

    # Apply l2 on all the required layered of the resnet
    l2 = tf.keras.regularizers.l2(1e-4)
    for layer in model.resnet.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and layer.trainable:
            model.add_loss(lambda layer=layer: l2(layer.kernel))
    return model
