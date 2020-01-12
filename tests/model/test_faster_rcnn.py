import tensorflow as tf
from od.model.faster_rcnn import build_fpn_resnet50_faster_rcnn

def test_build_fpn_resnet50_faster_rcnn():
    num_classes = 20
    batch_size = 3
    model = build_fpn_resnet50_faster_rcnn(num_classes, batch_size)
    model_inference = build_fpn_resnet50_faster_rcnn(num_classes, None, training=False)


def test_build_fpn_resnet50_faster_rcnn_mixed_precision():
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    num_classes = 20
    batch_size = 3
    model = build_fpn_resnet50_faster_rcnn(num_classes, batch_size)
    mixed_precision.set_policy('float32')

