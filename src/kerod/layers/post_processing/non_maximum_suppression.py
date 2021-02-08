"""Methods using non maximum_suppression to handle overlaps between boxes.
"""
from typing import List
import tensorflow as tf

from kerod.core.box_coder import decode_boxes_faster_rcnn
from kerod.core.box_ops import clip_boxes
from kerod.core.standard_fields import BoxField
from kerod.utils.ops import get_full_indices


def post_process_rpn(cls_pred_per_lvl: List[tf.Tensor],
                     loc_pred_per_lvl: List[tf.Tensor],
                     anchors_per_lvl: List[tf.Tensor],
                     image_information,
                     pre_nms_topk_per_lvl,
                     post_nms_topk=None,
                     iou_threshold: float = 0.7):
    """Sample RPN proposals by the following steps:

    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Arguments:

    - *cls_pred_per_lvl*: A list of Tensor of shape [batch_size, num_boxes, 2].
    One item per level of the pyramid.
    - *loc_pred_per_lvl*: A list of Tensor of shape [batch_size, num_boxes, 4 * (num_anchors)].
    One item per level of the pyramid.
    - *anchors_per_lvl*: A list of Tensor of shape [batch_size, num_boxes * num_anchors, 4]
    One item per level of the pyramid.
    - *image_informations*: A Tensor of shape [batch_size, (height, width)] The height and the
    width are without the padding.
    - *pre_nms_topk_per_lvl*: Will extract at each level this amount of boxes
    - post_nms_topk: Number of boxes selected after the nms


    Returns:

    - *nmsed_boxes*: A Tensor of shape [batch_size, max_detections, 4]
      containing the non-max suppressed boxes.
    - *nmsed_scores*: A Tensor of shape [batch_size, max_detections] containing
      the scores for the boxes.
    """
    topk_boxes_per_lvl = []
    topk_scores_per_lvl = []
    for cls_pred, loc_pred, anchors in zip(cls_pred_per_lvl, loc_pred_per_lvl, anchors_per_lvl):
        batch_size = tf.shape(cls_pred)[0]
        loc_pred = tf.reshape(loc_pred, (batch_size, -1, 4))
        boxes = decode_boxes_faster_rcnn(loc_pred, anchors)
        boxes = clip_boxes(boxes, image_information)

        # Remove the background classes
        scores = cls_pred[:, :, 1]

        topk = tf.minimum(pre_nms_topk_per_lvl, tf.size(scores[0]))
        topk_scores, topk_indices = tf.math.top_k(scores, k=topk, sorted=False)
        topk_indices = get_full_indices(topk_indices)
        topk_boxes_per_lvl.append(tf.cast(tf.gather_nd(boxes, topk_indices), tf.float32))
        topk_scores_per_lvl.append(tf.cast(topk_scores, tf.float32))

    topk_boxes = tf.concat(topk_boxes_per_lvl, 1)
    topk_scores = tf.concat(topk_scores_per_lvl, 1)

    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk_per_lvl
    nmsed_boxes, _, _, _ = tf.image.combined_non_max_suppression(topk_boxes[:, :, None],
                                                                 topk_scores[..., None],
                                                                 post_nms_topk,
                                                                 post_nms_topk,
                                                                 iou_threshold=iou_threshold,
                                                                 score_threshold=0,
                                                                 clip_boxes=False)

    return tf.stop_gradient(nmsed_boxes)


def post_process_fast_rcnn_boxes(cls_pred: tf.Tensor,
                                 loc_pred: tf.Tensor,
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

    - *cls_pred*: A Tensor of shape [batch_size, num_boxes, num_classes]
    - *loc_pred*: A Tensor of shape [batch_size, num_boxes, 4 * num_classes]
    - *anchors*: A Tensor of shape [batch_size, num_boxes, 4]
    - *image_informations*: A Tensor of shape [batch_size, (height, width)] The height and the
    width are without the padding.
    - *num_classes*: The number of classes (background is not included).
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

    batch_size = tf.shape(cls_pred)[0]

    loc_pred = tf.reshape(loc_pred, (batch_size, -1, 4))
    anchors = tf.reshape(tf.tile(anchors, [1, 1, num_classes]), (batch_size, -1, 4))

    boxes = decode_boxes_faster_rcnn(loc_pred, anchors, scale_factors=(10.0, 10.0, 5.0, 5.0))
    boxes = clip_boxes(boxes, image_information)
    boxes = tf.reshape(boxes, (batch_size, -1, num_classes, 4))

    nmsed_boxes, nmsed_scores, nmsed_labels, valid_detections = tf.image.combined_non_max_suppression(
        tf.cast(boxes, tf.float32),
        tf.cast(cls_pred, tf.float32),
        max_output_size_per_class,
        max_total_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        clip_boxes=False)

    normalizer_boxes = tf.tile(image_information[:, None], [1, 1, 2])
    nmsed_boxes = tf.math.divide(nmsed_boxes, normalizer_boxes, name=BoxField.BOXES)
    nmsed_scores = tf.identity(nmsed_scores, name=BoxField.SCORES)
    nmsed_labels = tf.identity(nmsed_labels, name=BoxField.LABELS)
    valid_detections = tf.identity(valid_detections, name=BoxField.NUM_BOXES)
    return nmsed_boxes, nmsed_scores, nmsed_labels, valid_detections
