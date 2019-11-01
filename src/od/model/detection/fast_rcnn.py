import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.initializers import VarianceScaling
from od.core.box_ops import compute_area
from od.core.losses import SmoothL1Localization, CategoricalCrossentropy
from od.model.detection.abstract_detection_head import AbstractDetectionHead
from od.model.detection.pooling_ops import roi_align


class FastRCNN(AbstractDetectionHead):

    def __init__(self, num_classes):
        """Constructor of the FastRCNN head. It will build

        Arguments:

        - *num_classes*: The number of classes that predict the classification head (N+1).
        """
        super().__init__(
            'fast_rcnn',
            num_classes,
            SmoothL1Localization(),
            CategoricalCrossentropy(),
        )

    def call(self, inputs):
        pyramid, boxes = inputs
        # Remove P6
        pyramid = pyramid[:-1]
        boxe_tensors = multilevel_roi_align(pyramid, boxes, crop_size=7)
        l = tfkl.Flatten()(boxe_tensors)
        l = tfkl.Dense(1024, kernel_initializer=VarianceScaling(), activation='relu')(l)
        l = tfkl.Dense(1024, kernel_initializer=VarianceScaling(), activation='relu')(l)

        classification_head, localization_head = self.build_detection_head(
            tf.reshape(l, (-1, 1, 1, 1024)))
        batch_size = tf.shape(boxes)[0]
        classification_head = tf.reshape(classification_head, (batch_size, -1, self._num_classes))
        localization_head = tf.reshape(localization_head,
                                       (batch_size, -1, (self._num_classes - 1) * 4))
        return classification_head, localization_head


def multilevel_roi_align(inputs, boxes, image_shape, crop_size: int = 7):
    """Perform a batch multilevel roi_align on the inputs

    Arguments:

    - *inputs*: A list of tensors of type tf.float32 and shape [batch_size, width, height, channel]
            representing the pyramid.
    - *boxes*: A tensor of type tf.float32 and shape [batch_size, num_boxes, (y1, x1, y2, x2)]

    - *image_shape*: A tuple with the height and the width of the original image input image

    Returns:

    A tensor of type tf.float32 and shape [batch_size * num_boxes, 7, 7, channel]
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

    - *boxes*: A tensor of type tf.float32 and shape [batch_size, num_boxes, 4]
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

    - *boxes*: A tensor of type float32 and shape
            [nb_batches * nb_boxes, 4]
    - *num_level*: Assign all the boxes mapped to a superior level to the num_level.
        level_target: Will affect all the boxes of area 224^2 to the level_target of the pyramid.

    Returns:

    A 2-D tensor of type int32 and shape [nb_batches * nb_boxes]
    corresponding to the target level of the pyramid.
    """

    area = compute_area(boxes)
    k = level_target + tf.math.log(tf.sqrt(area) / 224 + 1e-6) * 1. / tf.math.log(2.0)
    k = tf.clip_by_value(k, 0, num_level - 1)
    k = tf.cast(k, tf.int32)
    k = tf.reshape(k, [-1])
    return k
