from typing import List
import tensorflow as tf
from kerod.utils import item_assignment, get_full_indices
from kerod.core.standard_fields import BoxField
from kerod.core.box_ops import compute_giou
from scipy.optimize import linear_sum_assignment


class Matcher:
    """This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.

    The matching is determined by the MxN match_quality_matrix, that characterizes
    how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values.

    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.

    Arguments:

    - *thresholds*: a list of thresholds used to stratify predictions
                into levels.

    - *labels*: a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.

    - *allow_low_quality_matches*: if True, produce additional matches
            for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.

    Example:

    thresholds = [0.3, 0.5]
    labels = [0, -1, 1]
    All predictions with iou < 0.3 will be marked with 0 and
    thus will be considered as false positives while training.
    All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
    thus will be ignored.
    All predictions with 0.5 <= iou will be marked with 1 and
    thus will be considered as true positives.

    """

    def __init__(self,
                 thresholds: List[float],
                 labels: List[int],
                 allow_low_quality_matches: bool = False):
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: tf.Tensor, num_valid_boxes: tf.Tensor):
        """
        Arguments:

        - *match_quality_matrix*: A tensor or shape [batch_size, M, N], containing the
        pairwise quality between M ground-truth elements and N predicted
        elements.

        - *num_valid_boxes*: A tensor of shape [batch_size, 1] indicating where is the padding on
        the ground_truth_boxes. E.g: If your quality_matrix is of shape [2, 4, 6] and `num_valid_boxes` 
        is equal to [3, 4] the boxes for the `batch=0` is padded from `pos=3`. It means,
        that `quality_matrix[0, 3:]` all the values from this pattern should not be considered because
        of the padding.

        Returns:

        - *matches*: a tensor of int32 and shape [batch_size, N], where matches[b, i] is a matched
               ground-truth index in [b, 0, M)
        - *match_labels*: a tensor of int32 and shape [batch_size, N], where match_labels[i] indicates
                whether a prediction is a true (1) or false positive (0) or ignored (-1)
        """
        assert len(match_quality_matrix.shape) == 3
        num_valid_boxes = tf.squeeze(num_valid_boxes, -1)
        # match_quality_matrix is B (batch) x M (gt) x N (predicted)
        # Max over gt elements to find best gt candidate for each prediction
        matches = tf.argmax(match_quality_matrix, axis=1, output_type=tf.int32)
        matched_vals = tf.math.reduce_max(match_quality_matrix, axis=1)
        # matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high)
            match_labels = item_assignment(match_labels, low_high, l)

        if self.allow_low_quality_matches:
            match_labels = self._set_low_quality_matches(match_labels, match_quality_matrix,
                                                         num_valid_boxes)

        # Remove all the padded groundtruths
        # e.g: matches = [1, 0, 4, 3] num_valid_boxes = [3]
        # mask_padded_boxes = [0, 0, 1, 1]
        mask_padded_boxes = matches >= num_valid_boxes[:, None]
        # Will flag to -1 all the padded boxes to avoid sampling them
        match_labels = item_assignment(match_labels, mask_padded_boxes, -1)

        return matches, match_labels

    def _set_low_quality_matches(self, match_labels, match_quality_matrix, num_valid_boxes):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.

        Arguments:

        - *match_labels*: a tensor of int8 and shape [batch_size, N], where match_labels[i] indicates
        whether a prediction is a true (1) or false positive (0) or ignored (-1)

        - *match_quality_matrix*: A tensor or shape [batch_size, M, N], containing the
        pairwise quality between M ground-truth elements and N predicted
        elements.

        - *num_valid_boxes*: A tensor of shape [batch_size] indicating where is the padding on
        the ground_truth_boxes. E.g: If your quality_matrix is of shape [2, 4, 6] and `num_valid_boxes` 
        is equal to [3, 4] the boxes for the `batch=0` is padded from `pos=3`. It means,
        that `quality_matrix[0, 3:]` all the values from this pattern should not be considered because
        of the padding.
        """
        # For each gt, find the prediction with which it has highest quality: shape [batch_size, M]
        highest_quality_gt = tf.math.reduce_max(match_quality_matrix, axis=2)

        # Find the highest quality match available, even if it is low, including ties.
        hq_foreach_gt_mask = match_quality_matrix == highest_quality_gt[..., None]
        # Create a mask for the valid_boxes, shape = [batch_size, M]
        mask_valid_boxes = tf.sequence_mask(num_valid_boxes,
                                            maxlen=tf.shape(match_quality_matrix)[1],
                                            dtype=tf.bool)
        hq_foreach_gt_mask = hq_foreach_gt_mask & mask_valid_boxes[..., None]

        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        # shape = [batch_size, N]
        masks_for_labels = tf.reduce_max(tf.cast(hq_foreach_gt_mask, tf.int8), 1)
        match_labels = item_assignment(match_labels, masks_for_labels, 1)
        return match_labels


def hungarian_matching(match_quality_matrix: tf.Tensor, num_valid_boxes: tf.Tensor):
    """Find the maximum-weight matching of the match_quality_matrix.
    A maximum-weight matching is also a perfect matching.

    Arguments:

    - *match_quality_matrix*: A tensor or shape [batch_size, M, N], containing the
    pairwise quality between M ground-truth elements and N predicted
    elements.

    - *num_valid_boxes*: A tensor of shape [batch_size, 1] indicating where is the padding on
    the ground_truth_boxes. E.g: If your quality_matrix is of shape [2, 4, 6] and `num_valid_boxes` 
    is equal to [3, 4] the boxes for the `batch=0` is padded from `pos=3`. It means,
    that `quality_matrix[0, 3:]` all the values from this pattern should not be considered because
    of the padding.

    Returns:

    - *matches*: a tensor of int32 and shape [batch_size, N], where matches[b, i] is a matched
            ground-truth index in [b, 0, M)
    - *match_labels*: a tensor of int32 and shape [batch_size, N], where match_labels[i] indicates
            whether a prediction is a true (1) or false positive (0) or ignored (-1)
    """

    def hungarian_assignment(cost_matrix):
        return tf.py_function(lambda c: linear_sum_assignment(c), [cost_matrix],
                              Tout=(tf.int32, tf.int32))

    # [batch_size, num_gt_boxes]
    indices = tf.vectorized_map(hungarian_assignment, match_quality_matrix)

    matches = tf.one_hot(indices[1], tf.shape(match_quality_matrix)[-1], dtype=tf.int32)
    match_labels = tf.cast(tf.reduce_max(matches, axis=1), tf.int32)
    matches = tf.reduce_max(matches * indices[0][..., None], axis=1)

    # Remove all the padded groundtruths
    # e.g: matches = [1, 0, 4, 3] num_valid_boxes = [3]
    # mask_padded_boxes = [0, 0, 1, 1]
    mask_padded_boxes = matches >= num_valid_boxes
    # Will flag to -1 all the padded boxes to avoid sampling them
    match_labels = item_assignment(match_labels, mask_padded_boxes, -1)
    return tf.stop_gradient(matches), tf.stop_gradient(match_labels)
