import tensorflow as tf

from kerod.layers.patches import Patches


def test_patches():
    patches = Patches(7)
    images = tf.zeros((5, 650, 850, 3))

    out = patches(images)
    assert out.shape == (5, (650 // 7) * (850 // 7), 7 * 7 * 3)
