import tensorflow as tf
import numpy as np
from kerod.dataset.utils import filter_bad_area
from kerod.core.standard_fields import BoxField


def test_filter_bad_area():
    boxes = tf.constant([
        [0., 0., 0., 1.],
        [0., 1., 0.5, 1.],
        [0., 0., 0., 0.],
        [0., 0., 1., 1.],
        [1., 1., 1., 1.],
    ])
    out_boxes = filter_bad_area({BoxField.BOXES: boxes})
    np.testing.assert_array_equal(out_boxes[BoxField.BOXES], np.array([
        [0., 0., 1., 1.],
    ]))
