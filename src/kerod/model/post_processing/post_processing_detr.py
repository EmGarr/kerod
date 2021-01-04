import tensorflow as tf
from kerod.core.standard_fields import BoxField
from kerod.utils.ops import get_full_indices
from kerod.core.box_ops import convert_to_xyxy_coordinates


def post_processing(boxes: tf.Tensor,
                    logits: tf.Tensor,
                    image_information: tf.Tensor,
                    image_padded_information: tf.Tensor,
                    sorted=True):
    """PostProcessing described in the paper Object Detection with transformers

    "To optimize for AP, we override the prediction of these slots
    with the second highest scoring class, using the corresponding confidence"
    Part 4.

    Example: background + 3 classes
    [0.54, 0.40, 0.03, 0.03] => score = 0.40, label = 0 (1 - 1)

    Arguments:

    - *logits*: A Tensor of shape [batch_size, num_queries, num_classes + 1] representing
        the class probability.
    - *localization_pred*: A Tensor of shape [batch_size, num_queries, (y_cent, x_cent, h, w)]
    - *image_information*: A 2-D tensor of float32 and shape [2, (height, width)]. It contains the shape
        of the image without any padding.
    - *image_padded_information*: A 2-D tensor of float32 and shape [(height_pad, width_pad)]. It contains the shape
        of the image without any padding. This padding is added during the dataset step when we batch the images together
    (padded_batch).
    - *sorted*: Return all the elements sorted by scores in descending order.

    Returns:

    - *boxes*: A Tensor of shape [batch_size, self.num_queries, (y1,x1,y2,x2)]
    containing the boxes with the coordinates between 0 and 1.
    - *scores*: A Tensor of shape [batch_size, self.num_queries] containing
    the score of the boxes.
    - *classes*: A Tensor of shape [batch_size, self.num_queries]
    containing the class of the boxes [0, num_classes).
    """
    probabilities = tf.nn.softmax(logits, axis=-1)
    # Remove the background at pos 0
    scores = tf.reduce_max(probabilities[:, :, 1:], axis=-1, name=BoxField.SCORES)
    labels = tf.argmax(probabilities[:, :, 1:], axis=-1, name=BoxField.LABELS)
    # Prediction between 0 and 1 are performed with padding
    # Boxes (y1,x1,y2,x2) * Padded_image_(h,w,h,w) /unpadded_image_(h,w,h,w)
    # where padded_image and unpadded_image are in the image space
    image_padded_information = tf.cast(image_padded_information, boxes.dtype)
    image_information = tf.cast(image_information, boxes.dtype)
    # [batch_size, (y1_coeff, x1_coeff, y2_coeff, x2_coeff)]
    coeffs = tf.tile(image_padded_information, [2]) / tf.tile(image_information, [1, 2])
    boxes = convert_to_xyxy_coordinates(boxes)
    boxes_without_padding = boxes * coeffs[:, None]
    boxes_without_padding = tf.clip_by_value(boxes_without_padding, 0, 1, name=BoxField.BOXES)

    if not sorted:
        return boxes_without_padding, scores, labels

    sorted_scores, indices = tf.math.top_k(scores, k=tf.shape(scores)[-1], sorted=True)
    indices = get_full_indices(indices)
    sorted_labels = tf.gather_nd(labels, indices)
    sorted_boxes_without_padding = tf.gather_nd(boxes_without_padding, indices)

    return sorted_boxes_without_padding, sorted_scores, sorted_labels
