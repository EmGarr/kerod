import numpy as np

from od.dataset.preprocessing import resize_to_min_dim


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
