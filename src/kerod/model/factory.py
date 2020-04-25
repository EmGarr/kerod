import tensorflow as tf

from kerod.model.faster_rcnn import FasterRcnnFPNResnet50Caffe, FasterRcnnFPNResnet50Pytorch
from kerod.utils.training import (freeze_batch_normalization, freeze_layers_before)


def build_model(num_classes: int, name='resnet50_pytorch') -> tf.keras.Model:
    """Build a FasterRcnnFPNResnet50 with all the `tf.keras.layers.BatchNormalization` frozen and
    all the layers before second residual block.

    Argument:

    - *num_classes*: Number of classes of your model. Do not include the background class.
    - *name*: Target model that you wish to use: 'resnet50_pytorch', 'resnet50_caffe'

    Return:

    A `keras.Model` instance.

    Raises:

    NotImplementedError: If the provided isn't supported
    """
    if name == 'resnet50_pytorch':
        model = FasterRcnnFPNResnet50Pytorch(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'resnet50/group0/block2/last_relu')
        return model
    elif name == 'resnet50_caffe':
        model = FasterRcnnFPNResnet50Caffe(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'conv2_block3_out')
        return model

    raise NotImplementedError(f'Name: {name} is not implemented.')
