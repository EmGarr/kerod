from unittest import mock

import numpy as np
import pytest

from kerod.core.constants import MAX_IMAGE_DIMENSION
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.dataset.preprocessing import (expand_dims_for_single_batch, preprocess,
                                         resize_to_min_dim)

def test_resize_to_min_dim():
    image = np.zeros((100, 50, 3))
    max_size = 1000
    target_size = 800
    image_out = resize_to_min_dim(image, target_size, max_size)
    image_exp = np.zeros((1000, 500, 3))

    np.testing.assert_array_equal(image_exp, image_out)

    image = np.zeros((50, 50, 3))
    target_size = 800
    image_out = resize_to_min_dim(image, target_size, max_size)
    image_exp = np.zeros((800, 800, 3))

    np.testing.assert_array_equal(image_exp, image_out)

    with pytest.raises(ValueError):
        image_out = resize_to_min_dim(image, target_size, MAX_IMAGE_DIMENSION + 1)


def test_preprocess():
    inputs = {
        'image': np.zeros((100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
            BoxField.LABELS: np.array([1])
        }
    }
    x, y = preprocess(inputs, horizontal_flip=False)

    np.testing.assert_array_equal(np.zeros((1333, 666, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([1333, 666]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.array([[0, 0, 1333, 666]]), y[BoxField.BOXES])
    np.testing.assert_array_equal(np.array([1]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([1.]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1,)


# Since the return value is 0.6 it means the image and the boxes will be flipped
@mock.patch('kerod.dataset.augmentation.tf.random.uniform', spec=True, return_value=0.6)
def test_preprocess_horizontal_flip(mock):
    inputs = {
        'image': np.zeros((100, 50, 3)),
        'objects': {
            BoxField.BOXES:
                np.array([[0, 0.1, 1, 0.8], [0, 0, 0.9, 0.8], [1, 0, -1, 1], [0, 0, 0, 0]],
                         dtype=np.float32),
            BoxField.LABELS:
                np.array([4, 2, 3, 3]),
            'is_crowd':
                np.array([False, True, False, False])
        }
    }
    x, y = preprocess(inputs, padded_mask=True)

    np.testing.assert_array_equal(np.zeros((1333, 666, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([1333, 666]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.ones([1333, 666], np.int8), x[DatasetField.IMAGES_PMASK])
    np.testing.assert_allclose(np.array([[0, 133.2, 1333, 599.39996]]), y[BoxField.BOXES], 1e-5)
    np.testing.assert_array_equal(np.array([4]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([1.]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1,)


@mock.patch('kerod.dataset.augmentation.tf.random.uniform', spec=True)
def test_preprocess_random_crop(mock):
    # Offset cropping operation
    y, x = 170, 80

    # Here we will have two calls at tf.random.uniform
    # So we provide two values for each call
    # 0.6 will activate the random_cropping
    # (y, x, 0) is the return value of tf.random.uniform
    # for the offset in _random_crop
    mock.side_effect = [0.6, np.array([y, x, 0])]
    inputs = {
        'image': np.zeros((300, 200, 3)),
        'objects': {
            BoxField.BOXES:
                np.array([
                    [0.4, 0.3, 0.6, 0.6],
                    [0.5, 0.40, 0.9, 0.45],
                    [0.1, 0.1, 0.101, 0.101],
                    [0.85, 0.4, 0.95, 0.5],
                    [0.98, 0.98, 0.99, 0.99],
                ], np.float32),
            BoxField.LABELS:
                np.array([4, 2, 3, 3, 3]),
            'is_crowd':
                np.array([False, True, True, False, False])
        }
    }
    x, y = preprocess(inputs, horizontal_flip=False, random_crop_size=(20, 20, 3), padded_mask=True)

    np.testing.assert_array_equal(np.zeros((800, 800, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([800, 800]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.ones([800, 800], np.int8), x[DatasetField.IMAGES_PMASK])
    np.testing.assert_allclose(np.array([[0, 0, 400, 800.]]), y[BoxField.BOXES], 1e-5)
    np.testing.assert_array_equal(np.array([
        4,
    ]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([1.]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1,)


def test_expand_dims_for_single_batch():
    inputs = {
        'image': np.zeros((100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
            BoxField.LABELS: np.array([1])
        }
    }

    x, y = expand_dims_for_single_batch(*preprocess(inputs))

    np.testing.assert_array_equal(np.zeros((1, 1333, 666, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([[1333, 666]]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.array([[[0, 0, 1333, 666]]]), y[BoxField.BOXES])
    np.testing.assert_array_equal(np.array([[1]]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([[1.]]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1, 1)
