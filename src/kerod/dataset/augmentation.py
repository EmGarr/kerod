import tensorflow as tf
from kerod.core import box_ops


def random_horizontal_flip(image, boxes, seed=None):
    """Randomly flips the image and detections horizontally.
    The probability of flipping the image is 50%.

    Arguments:

    - image: rank 3 float32 tensor with shape [height, width, channels].
    - boxes: rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].

    Returns:

    - image: image which is the same shape as input image.
    - boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    """

    with tf.name_scope('RandomHorizontalFlip'):
        uniform_random = tf.random.uniform([], 0, 1.0, seed=seed)
        if uniform_random > .5:
            image = tf.image.flip_left_right(image)
            boxes = box_ops.flip_left_right(boxes)
    return image, boxes
