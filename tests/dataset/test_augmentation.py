from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
from kerod.core.standard_fields import BoxField
from kerod.dataset.augmentation import random_crop, random_horizontal_flip


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


@mock.patch('kerod.dataset.augmentation.tf.random.uniform', spec=True)
def test_random_crop(mock):
    y, x = 170, 80
    mock.return_value = np.array([y, x, 0])

    image = np.ones((300, 200, 3)) * np.arange(300).reshape(300, 1, 1)

    boxes = tf.constant([
        [0.4, 0.3, 0.6, 0.6],
        [0.5, 0.40, 0.9, 0.45],
        [0.1, 0.1, 0.101, 0.101],
        [0.85, 0.4, 0.95, 0.5],
        [0.98, 0.98, 0.99, 0.99],
    ])
    labels = tf.constant([1, 2, 3, 4, 5])

    image_out, gt_out = random_crop(image, (20, 20, 3), {
        BoxField.BOXES: boxes,
        BoxField.LABELS: labels
    })

    np.testing.assert_array_equal(image_out, image[y:y + 20, x:x + 20])
    np.testing.assert_array_equal(gt_out[BoxField.BOXES], [[0, 0, .5, 1], [0, 0, 1., 0.5]])

    np.testing.assert_array_equal(gt_out[BoxField.LABELS], [1, 2])
