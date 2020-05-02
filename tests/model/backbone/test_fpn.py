import tensorflow as tf

from kerod.model.backbone.fpn import FPN


def test_build_fpn():
    shapes = [160, 80, 40, 20]
    features = [tf.zeros((1, shape, shape, 3)) for shape in shapes]
    pyramid = FPN()(features)
    assert len(pyramid) == len(shapes) + 1
    for p, shape in zip(pyramid[:-1], shapes):
        assert p.shape[1] == shape
        assert p.shape[2] == shape
    assert pyramid[-1].shape[1] == 10
    assert pyramid[-1].shape[2] == 10
