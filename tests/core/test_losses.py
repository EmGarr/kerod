import tensorflow as tf
from kerod.core.losses import L1Loss


def test_class_l1_loss():
    boxes1 = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                          [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])

    boxes2 = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]],
                          [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]]])
    assert 25. == L1Loss()(boxes1, boxes2)
