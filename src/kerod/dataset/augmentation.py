from typing import Dict

import tensorflow as tf
from kerod.core import box_ops
from kerod.core.standard_fields import BoxField
from kerod.dataset.utils import filter_bad_area


def random_horizontal_flip(image: tf.Tensor, boxes: tf.Tensor, seed=None):
    """Randomly flips the image and detections horizontally.
    The probability of flipping the image is 50%.

    Arguments:

    - *image*: rank 3 float32 tensor with shape [height, width, channels].
    - *boxes*: rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:

    - *image*: image which is the same shape as input image.
    - *boxes*: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    """

    with tf.name_scope('RandomHorizontalFlip'):
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        if uniform_random > .5:
            image = tf.image.flip_left_right(image)
            boxes = box_ops.flip_left_right(boxes)
    return image, boxes


def _random_crop(value: tf.Tensor, size: tf.Tensor, seed=None) -> (tf.Tensor, tf.Tensor):
    """Randomly crops a tensor to a given size in a deterministic manner.
    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).

    Arguments:

    - *value*: Input tensor to crop.
    - *size*: 1-D tensor with size the rank of `value`.
    - *seed*: A shape [2] Tensor, the seed to the random number generator. Must have

    Returns:

    - *crop*: A cropped tensor of the same rank as `value` and shape `size`.
    - *top_left_corner*: A 2-D tensor of int and shape [1, (y, x)]
    """
    with tf.name_scope("RandomCrop"):
        value = tf.convert_to_tensor(value, name="value")
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        shape = tf.shape(value)

        limit = shape - size + 1
        offset = tf.random.uniform(
            tf.shape(shape), dtype=size.dtype, maxval=size.dtype.max, seed=seed) % limit
        return tf.slice(value, offset, size), offset[:2]


def random_crop(image: tf.Tensor,
                size: tf.Tensor,
                groundtruths: Dict[str, tf.Tensor],
                seed=None) -> (tf.Tensor, tf.Tensor):
    """Randomly crops a tensor to a given size in a deterministic manner.
    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).

    Arguments:

    - *image*: Input tensor to crop.
    - *size*: 1-D tensor with size the rank of `image`.
    - *groundtruths*: A dict with the following keys:
        1. bbox: rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        2. label: A tensorf of shape [N, ]
    - *seed*: A shape [2] Tensor, the seed to the random number generator.

    Returns:

    - *crop*: A cropped tensor of the same rank as `value` and shape `size`.
    - *groundtruths*: Diction
        1. bbox: 2-D float32 tensor containing the bounding boxes -> [N_crop <=N, 4].
            If cropped, the boxes with no area or outside the crop or removed.
        2. label: A tensor of shape [N_crop <= N, ]
    """
    boxes = groundtruths[BoxField.BOXES]
    crop, top_left_corner = _random_crop(image, size, seed=seed)

    shape = tf.cast(tf.shape(image)[:2], boxes.dtype)
    size = tf.tile(tf.constant(size[:2], boxes.dtype)[None], (1, 2))

    # scale the boxes to the size of the image
    boxes *= tf.tile(shape[tf.newaxis], (1, 2))
    top_left_corner = tf.tile(top_left_corner[tf.newaxis], (1, 2))
    top_left_corner = tf.cast(top_left_corner, boxes.dtype)

    # Translate according to top_left_corner
    # top_left_corner is now (y=0, x=0)
    cropped_boxes = boxes - top_left_corner
    # MinClip to 0 and MaxClip to size all the coords outside the crop.
    cropped_boxes = tf.maximum(cropped_boxes, tf.cast(0, boxes.dtype))
    cropped_boxes = tf.minimum(cropped_boxes, size)
    # Renomarlized to have the boxes between 0 and 1
    cropped_boxes = cropped_boxes / size

    # Copy groundtruths
    cropped_gt = {key: val for key, val in groundtruths.items()}
    cropped_gt[BoxField.BOXES] = cropped_boxes

    return crop, filter_bad_area(cropped_gt)


def random_random_crop(image: tf.Tensor,
                       size: tf.Tensor,
                       groundtruths: Dict[str, tf.Tensor],
                       seed=None) -> (tf.Tensor, tf.Tensor):
    """Will `randomly` perform a random crop of a tensor to a given size.
    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Guarantees the same results given the same `seed` independent of how many
    times the function is called, and independent of global seed settings (e.g.
    `tf.random.set_seed`).

    Arguments:

    - *image*: Input tensor to crop or not.
    - *size*: 1-D tensor with size the rank of `image`.
    - *groundtruths*: A dict with the following keys:
        1. bbox: rank 2 float32 tensor with shape [N, 4]
            containing the bounding boxes.
            Boxes are in normalized form meaning their coordinates vary
            between [0, 1].
            Each row is in the form of [ymin, xmin, ymax, xmax].
        2. label: A tensorf of shape [N, ]
    - *seed*: A shape [2] Tensor, the seed to the random number generator.

    Returns:

    - *image*: Either a cropped tensor or the same image of the same rank as
        `value` and shape `size`.
    - *boxes*: 2-D float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form, meaning their coordinates vary
           between [0, 1]. If cropped, the boxes with no area or outside
           the crop or removed.
    """
    with tf.name_scope('RandomRandomCrop'):
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        if uniform_random > .5:
            image, targets = random_crop(image, size, groundtruths, seed=seed)
    return image, targets
