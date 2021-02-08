import tensorflow as tf
import numpy as np

from kerod.layers import DynamicalWeightMaps


def test_weight_map():
    beta = 10.
    layer = DynamicalWeightMaps(beta=beta)
    # shape = [1, N=2, heads=2, 4]
    reference_points = tf.constant([
        [
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.1, 0.1],
            ],
        ],
        [
            [
                [0.2, 0.1, 0.4, 0.3],
                [0.3, 0.2, 0.1, 0.1],
            ],
        ],
    ])
    height, width = 10, 6
    x = tf.cast(tf.linspace(0, 1, width), tf.float32)
    y = tf.cast(tf.linspace(0, 1, height), tf.float32)
    weight_map = layer(height, width, reference_points)

    assert weight_map.shape == (2, 2, 1, height * width)
    weight_map = tf.reshape(weight_map, (2, 2, 1, height, width))

    def scalar_formula(x, y, ref_points):
        y = -(y - ref_points[0])**2 / (beta * ref_points[2]**2)
        x = -(x - ref_points[1])**2 / (beta * ref_points[3]**2)
        return tf.math.exp(x + y)

    # test some values to check if our calculus hasn't mixed up some dim
    for batch in range(len(reference_points)):
        for i in range(height):
            for j in range(width):
                np.testing.assert_almost_equal(
                    weight_map[batch, 0, 0, i, j],
                    scalar_formula(
                        x[j],
                        y[i],
                        reference_points[batch, 0, 0],
                    ))
