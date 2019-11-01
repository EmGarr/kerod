import tensorflow as tf
import tensorflow.keras.layers as tfkl

from od.core.box_ops import transform_fpcoor_for_tf, normalize_box_coordinates


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
                                   boxes,
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
    return tfkl.AveragePooling2D(padding='same')(ret)
