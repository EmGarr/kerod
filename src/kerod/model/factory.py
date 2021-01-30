from enum import Enum

import tensorflow as tf
from kerod.model.detr import DeTrResnet50, DeTrResnet50Pytorch
from kerod.model.faster_rcnn import (FasterRcnnFPNResnet50Caffe, FasterRcnnFPNResnet50Pytorch)
from kerod.model.smca_detr import SMCAR50Pytorch
from kerod.utils.training import (freeze_batch_normalization, freeze_layers_before)


class KerodModel(str, Enum):
    faster_rcnn_resnet50_pytorch = 'resnet50_pytorch'
    faster_rcnn_resnet50_caffe = 'resnet50_caffe'
    detr_resnet50 = 'detr_resnet50_pytorch'
    detr_resnet50_caffe = 'detr_resnet50'
    smca_r50 = 'smca_resnet50'


def build_model(num_classes: int,
                name: str = KerodModel.faster_rcnn_resnet50_pytorch.value) -> tf.keras.Model:
    """Build a FasterRcnnFPNResnet50 with all the `tf.keras.layers.BatchNormalization` frozen and
    all the layers before second residual block.

    Args:
        num_classes: Number of classes of your model. Do not include the background class.
        name: Target model that you wish to use: 'resnet50_pytorch', 'resnet50_caffe', 'detrresnet50_pytorch', 'smca_resnet50'.

    Returns:
        A `keras.Model` instance.

    Raises:
        NotImplementedError: If the provided isn't supported
    """
    if name == KerodModel.faster_rcnn_resnet50_pytorch:
        model = FasterRcnnFPNResnet50Pytorch(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'resnet50/group0/block2/last_relu')
        return model
    elif name == KerodModel.faster_rcnn_resnet50_caffe:
        model = FasterRcnnFPNResnet50Caffe(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'conv2_block3_out')
        return model
    elif name == KerodModel.detr_resnet50:
        model = DeTrResnet50Pytorch(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'resnet50/group0/block2/last_relu')
        return model
    elif name == KerodModel.detr_resnet50_caffe:
        model = DeTrResnet50(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'conv2_block3_out')
        return model
    elif name == KerodModel.smca_r50:
        model = SMCAR50Pytorch(num_classes)
        freeze_batch_normalization(model.backbone)
        freeze_layers_before(model.backbone, 'resnet50/group0/block2/last_relu')
        return model

    raise NotImplementedError(f'Name: {name} is not implemented.')
