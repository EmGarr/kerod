import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras import backend as K

from od.core.box_ops import (compute_area, normalize_box_coordinates, transform_fpcoor_for_tf)


def _crop_and_resize(tensor, boxes, box_indices, crop_size: int, pad_border=True):
    """Taken from tensorpack (https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_box.py)
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Arguments:

    - *tensor*: A 4-D tensor of shape [batch, image_height, image_width, depth].
            Both image_height and image_width need to be positive.
    - *image_shape*: A

    - *boxes*: A 2-D tensor of shape [num_boxes, 4].
            The i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is
            specified in normalized coordinates [y1, x1, y2, x2]. A normalized coordinate value of y is
            mapped to the image coordinate at y * (image_height - 1), so as the [0, 1] interval of
            normalized image height is mapped to [0, image_height - 1] in image height coordinates.
            We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the
            original image. The width dimension is treated similarly. Normalized coordinates outside the
            [0, 1] range are allowed, in which case we use extrapolation_value to extrapolate the input
            image values.

    - *box_indices*: A 1-D tensor of shape [num_boxes] with int32 values in [0, batch).
            The value of box_ind[i] specifies the image that the i-th box refers to.

    - *crop_size*: An int representing the ouput size of the crop.

    - *pad_border*: Pad the border of the images

    Returns:

    A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        boxes = boxes + 1

    tensor_shape = tf.shape(tensor)[1:]
    boxes = transform_fpcoor_for_tf(boxes, tensor_shape, [crop_size, crop_size])

    ret = tf.image.crop_and_resize(image,
                                   tf.cast(boxes, tf.float32), # crop and resize needs float32
                                   tf.cast(box_indices, tf.int32),
                                   crop_size=[crop_size, crop_size])
    return ret


def roi_align(inputs, boxes, box_indices, image_shape, crop_size: int):
    """RoI align like operation from the paper Mask-RCNN.

    Arguments:

    - *inputs*: A 4-D tensor of shape [batch, height, tensor_width, depth].
            Both image_height and image_width need to be positive.

    - *boxes*: A 2-D tensor of shape [num_boxes, 4].
            The i-th row of the tensor specifies the coordinates of a box in the box_ind[i] image and is
            specified in normalized coordinates [y1, x1, y2, x2]. A normalized coordinate value of y is
            mapped to the image coordinate at y * (image_height - 1), so as the [0, 1] interval of
            normalized image height is mapped to [0, image_height - 1] in image height coordinates.
            We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the
            original image. The width dimension is treated similarly. Normalized coordinates outside the
            [0, 1] range are allowed, in which case we use extrapolation_value to extrapolate the input
            image values.

    - *box_indices*: A 1-D tensor of shape [num_boxes] with int32 values in [0, batch).
            The value of box_ind[i] specifies the image that the i-th box refers to.
    - *image_shape*: A tuple with the height and the width of the original image input image

    - *crop_size*: An int representing the ouput size of the crop.

    Returns:

    A 4-D tensor of shape [num_boxes, crop_height, crop_width, depth].

    """
    normalized_boxes = normalize_box_coordinates(boxes, image_shape[0], image_shape[1])
    ret = _crop_and_resize(inputs, normalized_boxes, box_indices, crop_size * 2)
    return KL.AveragePooling2D(padding='same')(ret)


def multilevel_roi_align(inputs, boxes, image_shape, crop_size: int = 7):
    """Perform a batch multilevel roi_align on the inputs

    Arguments:

    - *inputs*: A list of tensors of shape [batch_size, width, height, channel]
            representing the pyramid.
    - *boxes*: A tensor  and shape [batch_size, num_boxes, (y1, x1, y2, x2)]

    - *image_shape*: A tuple with the height and the width of the original image input image

    Returns:

    A tensor and shape [batch_size * num_boxes, 7, 7, channel]
    """
    boxes_per_level, box_indices_per_level, pos_per_level = match_boxes_to_their_pyramid_level(
        boxes, len(inputs))

    tensors_per_level = []
    for tensor, target_boxes, box_indices in zip(inputs, boxes_per_level, box_indices_per_level):
        tensors_per_level.append(
            roi_align(tensor, target_boxes, box_indices, image_shape, crop_size))

    tensors = tf.concat(values=tensors_per_level, axis=0)
    original_pos = tf.concat(values=pos_per_level, axis=0)

    # Reorder the tensor per batch
    indices_to_reorder_boxes = tf.math.invert_permutation(original_pos)
    tensors = tf.gather(tensors, indices_to_reorder_boxes)
    return tensors


def match_boxes_to_their_pyramid_level(boxes, num_level):
    """Match the boxes to the proper level based on their area

    Arguments:

    - *boxes*: A tensor of shape [batch_size, num_boxes, 4]
    - *num_level*: Number of level of the target pyramid

    Returns:

    - *boxes_per_level*
    - *box_indices_per_level*
    - *original_pos_per_level*

    """

    batch_size, num_boxes, _ = tf.shape(boxes)
    boxes = tf.reshape(boxes, (-1, 4))

    box_levels = assign_pyramid_level_to_boxes(boxes, num_level)
    box_indices = tf.tile(tf.expand_dims(tf.range(0, batch_size), 1), [1, num_boxes])
    box_indices = tf.reshape(box_indices, (-1,))
    box_original_pos = tf.range(batch_size * num_boxes)

    levels = [tf.squeeze(tf.where(tf.equal(box_levels, i))) for i in range(num_level)]
    boxes_per_level = [tf.gather(boxes, selected_level) for selected_level in levels]
    box_indices_per_level = [tf.gather(box_indices, selected_level) for selected_level in levels]
    original_pos_per_level = [
        tf.gather(box_original_pos, selected_level) for selected_level in levels
    ]
    return boxes_per_level, box_indices_per_level, original_pos_per_level


def assign_pyramid_level_to_boxes(boxes, num_level, level_target=2):
    """Compute the pyramid level of an RoI

    Arguments:

    - *boxes*: A tensor of shape
            [nb_batches * nb_boxes, 4]
    - *num_level*: Assign all the boxes mapped to a superior level to the num_level.
        level_target: Will affect all the boxes of area 224^2 to the level_target of the pyramid.

    Returns:

    A 2-D tensor of type int32 and shape [nb_batches * nb_boxes]
    corresponding to the target level of the pyramid.
    """

    denominator = tf.constant(224, dtype=boxes.dtype)
    area = compute_area(boxes)
    k = level_target + tf.math.log(tf.sqrt(area) / denominator + K.epsilon()) * tf.cast(
        1. / tf.math.log(2.0), dtype=boxes.dtype)
    k = tf.cast(k, tf.int32)
    k = tf.clip_by_value(k, 0, num_level - 1)
    k = tf.reshape(k, [-1])
    return k
