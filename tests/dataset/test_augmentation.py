from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from kerod.core import box_ops
from kerod.dataset.augmentation import random_horizontal_flip


@pytest.mark.parametrize("flip", [True, False])
@mock.patch('kerod.dataset.augmentation.tf.random.uniform', spec=True)
def test_random_horizontal_flip(mock, flip):
    # Above 0.5 perform a flip
    if flip:
        mock.return_value = 0.6
    else:
        mock.return_value = 0.3

    image = np.random.randn(3, 3, 3)

    boxes = tf.constant([[0.4, 0.3, 0.6, 0.6], [0.5, 0.6, 0.9, 0.65]])

    image_out, boxes_out = random_horizontal_flip(image, boxes)

    if flip:
        expected_boxes = np.array([[0.4, 0.4, 0.6, 0.7], [0.5, 0.35, 0.9, 0.4]], np.float32)
        image = tf.image.flip_left_right(image)
    else:
        expected_boxes = boxes

    np.testing.assert_allclose(boxes_out, expected_boxes)
    np.testing.assert_array_equal(image_out, image)
