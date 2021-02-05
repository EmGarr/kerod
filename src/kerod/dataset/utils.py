from typing import Dict

import tensorflow as tf
from kerod.core.box_ops import compute_area
from kerod.core.standard_fields import BoxField


def _filter(_dict, _filter):
    keys = {BoxField.BOXES, BoxField.LABELS, BoxField.MASKS, 'is_crowd'}
    filtered_dict = {}
    for key in _dict.keys():
        if key in keys:
            filtered_dict[key] = tf.gather_nd(_dict[key], _filter)
        else:
            filtered_dict[key] = _dict[key]
    return filtered_dict


def filter_crowded_boxes(groundtruths: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Coco has boxes flagged as crowded which are not used during the training.
    This function will discard them.

    Arguments:

    - *groundtruths*: A dict with the following keys
        1. boxes: A tensor of shape [num_boxes, (y1, x1, y2, x2)]
        2. labels: A tensor of shape [num_boxes, ]
        3. crowd: Boolean tensor which indicates if the boxes is crowded are not.
            Crowded means that the boxes contains multiple entities which
            are to difficult to localize one by one. `True` is for crowded box.

    Returns:

    - *groundtruths*: Filtered groundtruths
    """
    ind_uncrowded_boxes = tf.where(tf.equal(groundtruths['is_crowd'], False))
    return _filter(groundtruths, ind_uncrowded_boxes)


def filter_bad_area(groundtruths: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Remove all the boxes that have an area less or equal to 0.

    Arguments:

    - *groundtruths*: A dict with the following keys:
        1. bbox: A tensor of shape [num_boxes, (y1, x1, y2, x2)]
        2. label: A tensor of shape [num_boxes, ]

    Returns:

    - *groundtruths*: All the groundtruths which match have not been filtered.
    """
    area = compute_area(groundtruths[BoxField.BOXES])
    filter_area = tf.where(area > 0)
    return _filter(groundtruths, filter_area)
