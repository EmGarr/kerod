"""Methods using non maximum_suppression to handle overlaps between boxes.
"""
import tensorflow as tf

from kerod.core.box_coder import decode_boxes_faster_rcnn
from kerod.core.box_ops import clip_boxes
from kerod.core.standard_fields import BoxField


def get_full_indices(indices, k, batch_size):
    """ This operation allows to extract full indices from indices.
    These full-indices have the proper format for gather_nd operations.

    Arguments:

    - *indices*: Indices with the top_k format [batch_size, k].
    - *k*: k
    - *batch_size*: N

    Returns:

    Full-indices tensor [batch_size, k, 2]
    """
    # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    my_range = tf.expand_dims(tf.range(0, batch_size), 1)  # will be [[0], [1]]
    my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]
    # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    full_indices = tf.concat(
        [tf.expand_dims(my_range_repeated, 2),
         tf.expand_dims(indices, 2)], axis=2
    )
    return full_indices


def post_process_rpn(classification_pred: tf.Tensor,
                     localization_pred: tf.Tensor,
                     anchors: tf.Tensor,
                     image_information,
                     pre_nms_topk,
                     post_nms_topk=None,
                     iou_threshold: float = 0.7):
    """Sample RPN proposals by the following steps:

    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Arguments:

    - *classification_pred*: A Tensor of shape [batch_size, num_boxes, 2]
    - *localization_pred*: A Tensor of shape [batch_size, num_boxes, 4 * (num_anchors)]
    - *anchors*: A Tensor of shape [batch_size, num_boxes * num_anchors, 4]
    - *image_informations*: A Tensor of shape [batch_size, (height, width)] The height and the
    width are without the padding.
    - *pre_nms_topk*, post_nms_topk (int): See above.
    - post_nms_topk:


    Returns:

    - *nmsed_boxes*: A Tensor of shape [batch_size, max_detections, 4]
      containing the non-max suppressed boxes.
    - *nmsed_scores*: A Tensor of shape [batch_size, max_detections] containing
      the scores for the boxes.
    """
    batch_size = tf.shape(classification_pred)[0]
    localization_pred = tf.reshape(localization_pred, (batch_size, -1, 4))
    boxes = decode_boxes_faster_rcnn(localization_pred, anchors)
    boxes = clip_boxes(boxes, image_information)

    # Remove the background classes
    scores = classification_pred[:, :, 1]

    topk = tf.minimum(pre_nms_topk, tf.size(scores[0]))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_indices = get_full_indices(topk_indices, pre_nms_topk, batch_size)
    topk_boxes = tf.cast(tf.gather_nd(boxes, topk_indices), tf.float32)
    topk_scores = tf.cast(topk_scores, tf.float32)

    nmsed_boxes, nmsed_scores, _, _ = tf.image.combined_non_max_suppression(
        tf.expand_dims(topk_boxes, 2),
        tf.expand_dims(topk_scores, -1),
        post_nms_topk,
        post_nms_topk,
        iou_threshold=iou_threshold,
        score_threshold=0,
        clip_boxes=False)

    return tf.stop_gradient(nmsed_boxes), tf.stop_gradient(nmsed_scores)


def post_process_fast_rcnn_boxes(classification_pred: tf.Tensor,
                                 localization_pred: tf.Tensor,
                                 anchors: tf.Tensor,
                                 image_information: tf.Tensor,
                                 num_classes,
                                 max_output_size_per_class: int = 100,
                                 max_total_size: int = 100,
                                 iou_threshold: float = 0.5,
                                 score_threshold: float = 0.05):
    """This is the classical post_processing for the Faster RCNN paper.
    This operation performs non_max_suppression on the inputs per batch, across
    all classes.
    Prunes away boxes that have high intersection-over-union (IOU) overlap
    with previously selected boxes.  Bounding boxes are supplied as
    [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
    diagonal pair of box corners and the coordinates can be provided as normalized
    (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
    is agnostic to where the origin is in the coordinate system. Also note that
    this algorithm is invariant to orthogonal transformations and translations
    of the coordinate system; thus translating or reflections of the coordinate
    system result in the same boxes being selected by the algorithm.
    The output of this operation is the final boxes, scores and classes tensor
    returned after performing non_max_suppression.

    Arguments:

    - *classification_pred*: A Tensor of shape [batch_size, num_boxes, num_classes]
    - *localization_pred*: A Tensor of shape [batch_size, num_boxes, 4 * (num_classes - 1)]
    - *anchors*: A Tensor of shape [batch_size, num_boxes, 4]
    - *image_informations*: A Tensor of shape [batch_size, (height, width)] The height and the
    width are without the padding.
    - *num_classes*: The number of classes (background is included).
    - *max_output_size_per_class*: A scalar integer `Tensor` representing the
      maximum number of boxes to be selected by non max suppression per class
    - *max_total_size*: A scalar representing maximum number of boxes retained over
      all classes.
    - *iou_threshold*: A float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    - *score_threshold*: A float representing the threshold for deciding when to
      remove boxes based on score. `O.05` is used like in Detectron or Tensorpack.

    Returns:

    - *nmsed_boxes*: A Tensor of shape [batch_size, max_detections, 4]
      containing the non-max suppressed boxes. The coordinates returned are
    between 0 and 1.
    - *nmsed_scores*: A Tensor of shape [batch_size, max_detections] containing
      the scores for the boxes.
    - *nmsed_classes*: A Tensor of shape [batch_size, max_detections] 
      containing the class for boxes.
    - *valid_detections*: A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top valid_detections[i] entries
      in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.
    """

    batch_size = tf.shape(classification_pred)[0]
    # Remove the background classes
    classification_pred = classification_pred[:, :, 1:]
    localization_pred = tf.reshape(localization_pred, (batch_size, -1, 4))
    anchors = tf.reshape(tf.tile(anchors, [1, 1, num_classes - 1]), (batch_size, -1, 4))
    boxes = decode_boxes_faster_rcnn(localization_pred, anchors)
    boxes = clip_boxes(boxes, image_information)
    boxes = tf.reshape(boxes, (batch_size, -1, num_classes - 1, 4))

    nmsed_boxes, nmsed_scores, nmsed_labels, valid_detections = tf.image.combined_non_max_suppression(
        tf.cast(boxes, tf.float32),
        tf.cast(classification_pred, tf.float32),
        max_output_size_per_class,
        max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        clip_boxes=False)

    normalizer_boxes = tf.tile(tf.expand_dims(image_information, axis=1), [1, 1, 2])
    nmsed_boxes = tf.math.divide(nmsed_boxes, normalizer_boxes, name=BoxField.BOXES)
    nmsed_scores = tf.identity(nmsed_scores, name=BoxField.SCORES)
    nmsed_labels = tf.identity(nmsed_labels, name=BoxField.LABELS)
    valid_detections = tf.identity(valid_detections, name=BoxField.NUM_BOXES)
    return nmsed_boxes, nmsed_scores, nmsed_labels, valid_detections
