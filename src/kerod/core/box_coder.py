import tensorflow as tf

from kerod.core import box_ops

EPSILON = 1e-8


def encode_boxes_faster_rcnn(boxes, anchors, scale_factors=None):
    """Encode a box collection with respect to anchor collection according to the
    [Faster RCNN paper](http://arxiv.org/abs/1506.01497).

    Faster RCNN box coder follows the coding schema described below:

    t_y = (y - y_a) / h_a  
    t_x & = (x - x_a) / w_a   
    t_h & = log(h / h_a)  
    t_w & = log(w / w_a)  

    where y, x h, w denote the box's center coordinates, width and height
    respectively. Similarly,  y_a, x_a, h_a, w_a denote the anchor's center
    coordinates, width and height. t_y, t_x, t_h and t_w denote the anchor-encoded
    center, height and width respectively.
    
    Arguments:

    - *boxes*: BoxList holding N boxes to be encoded.
    - *anchors*: BoxList of anchors.
    - *scale_factors*: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].

    Returns:

    A tensor representing N anchor-encoded boxes of the format [ty, tx, th, tw].
    """
    # Convert anchors to the center coordinate representation.
    anchors = box_ops.convert_to_center_coordinates(anchors)
    ycenter_a, xcenter_a, ha, wa = tf.split(value=anchors, num_or_size_splits=4, axis=-1)
    boxes = box_ops.convert_to_center_coordinates(boxes)
    ycenter, xcenter, h, w = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
    # Avoid NaN in division and log below.
    ha += EPSILON
    wa += EPSILON
    h += EPSILON
    w += EPSILON

    ty = (ycenter - ycenter_a) / ha
    tx = (xcenter - xcenter_a) / wa
    th = tf.math.log(h / ha)
    tw = tf.math.log(w / wa)
    # Scales location targets as used in paper for joint training.

    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]

    return tf.concat([ty, tx, th, tw], axis=-1)


def decode_boxes_faster_rcnn(rel_codes, anchors, scale_factors=None):
    """Decode relative codes to boxes according to the
    [Faster RCNN paper](http://arxiv.org/abs/1506.01497).

    Faster RCNN box decoder follows the coding schema described below:

    ycent = t_y h_a + ycent_a   
    xcent= t_x w_a + xcent_a  
    h = exp(t_h) h_a  
    w = exp(t_w) w_a 
    
    where t_y, t_x, t_h, t_w denote the encoded box's center coordinates, width and height
    respectively. Similarly, ycent_a, xcent_a, h_a and w_a denote the anchor's center
    coordinates, width and height. ycent, xcent, h and w denote the anchor-encoded
    center, height and width respectively.

    Arguments:

    - *rel_codes*: a tensor representing N anchor-encoded boxes.
    - *anchors*: Tensor of shape [N, ..., (y_min,x_min,y_max,x_max)].
    - *scale_factors*: List of 4 positive scalars to scale ty, tx, th and tw.
        If set to None, does not perform scaling. For Faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0, 5.0].

    Returns:

    - *boxes*: A Tensor of shape [N, ..., (y_max,x_max,y2,x2)].
    """
    anchors = box_ops.convert_to_center_coordinates(anchors)
    ycenter_a, xcenter_a, ha, wa = tf.split(value=anchors, num_or_size_splits=4, axis=-1)

    ty, tx, th, tw = tf.split(value=rel_codes, num_or_size_splits=4, axis=-1)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]

    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    h = tf.exp(th) * ha
    w = tf.exp(tw) * wa

    ymin = ycenter - h / 2.
    xmin = xcenter - w / 2.
    ymax = ycenter + h / 2.
    xmax = xcenter + w / 2.
    return tf.concat([ymin, xmin, ymax, xmax], axis=-1)
