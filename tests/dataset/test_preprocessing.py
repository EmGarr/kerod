from unittest import mock

import numpy as np

from kerod.core.standard_fields import BoxField, DatasetField
from kerod.dataset.preprocessing import (expand_dims_for_single_batch, preprocess,
                                      preprocess_coco_example, resize_to_min_dim)


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


def test_preprocess():
    inputs = {
        'image': np.zeros((100, 50, 3)),
        'objects': {
            BoxField.BOXES: np.array([[0, 0, 1, 1]], dtype=np.float32),
            BoxField.LABELS: np.array([1])
        }
    }
    x, y = preprocess(inputs)

    np.testing.assert_array_equal(np.zeros((1300, 650, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([1300, 650]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.array([[0, 0, 1300, 650]]), y[BoxField.BOXES])
    np.testing.assert_array_equal(np.array([1]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([1.]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1,)


# Since the return value is 0.6 it means the image and the boxes will be flipped
@mock.patch('kerod.dataset.augmentation.tf.random.uniform', spec=True, return_value=0.6)
def test_preprocess_coco_example(mock):
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
    x, y = preprocess_coco_example(inputs)

    np.testing.assert_array_equal(np.zeros((1300, 650, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([1300, 650]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_allclose(np.array([[0, 130., 1300, 585]]), y[BoxField.BOXES], 1e-5)
    np.testing.assert_array_equal(np.array([4]), y[BoxField.LABELS])
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

    np.testing.assert_array_equal(np.zeros((1, 1300, 650, 3)), x[DatasetField.IMAGES])
    np.testing.assert_array_equal(np.array([[1300, 650]]), x[DatasetField.IMAGES_INFO])
    np.testing.assert_array_equal(np.array([[[0, 0, 1300, 650]]]), y[BoxField.BOXES])
    np.testing.assert_array_equal(np.array([[1]]), y[BoxField.LABELS])
    np.testing.assert_array_equal(np.array([[1.]]), y[BoxField.WEIGHTS])
    assert y[BoxField.NUM_BOXES].shape == (1, 1)
