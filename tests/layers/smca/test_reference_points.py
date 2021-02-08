import tensorflow as tf
import numpy as np

from kerod.layers import SMCAReferencePoints


def test_reference_points():
    layer = SMCAReferencePoints(128, 8)
    object_query = tf.random.normal((2, 50, 128))
    ref_points, embed = layer(object_query)
    assert ref_points.shape == (2, 50, 8, 4)
    assert embed.shape == (2, 50, 2)
