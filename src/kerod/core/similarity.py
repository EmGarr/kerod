from abc import abstractmethod
from typing import Dict

import tensorflow as tf
from kerod.core.box_ops import compute_giou, compute_iou
from kerod.core.standard_fields import BoxField
from kerod.utils import get_full_indices, item_assignment
from scipy.optimize import linear_sum_assignment


class Similarity:

    def __call__(self, inputs1: Dict[str, tf.Tensor], inputs2: Dict[str, tf.Tensor]):
        return self.call(inputs1, inputs2)

    @abstractmethod
    def call(self, inputs1, inputs2) -> tf.Tensor:
        pass


class IoUSimilarity(Similarity):

    def call(self, y_true: Dict[str, tf.Tensor], anchors: Dict[str, tf.Tensor]):
        """Computes pairwise intersection-over-union between boxes.

        Return:

        A 3-D tensor of float32 with shape [batch_size, N, M] representing
        pairwise  similarity scores defined in DeTr.
        """

        return compute_iou(y_true[BoxField.BOXES], anchors[BoxField.BOXES])


class DetrSimilarity(Similarity):

    def call(self, y_true: Dict[str, tf.Tensor], y_pred: Dict[str, tf.Tensor]) -> tf.Tensor:
        """ Compute the cost matrix according to the paper
        [End to end object detection with transformers](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers).

        Return:

        A 3-D tensor of float32 with shape [batch_size, N, M] representing
        pairwise  similarity scores defined in DeTr.
        """
        classification_logits = y_pred[BoxField.LABELS]
        localization_pred = y_pred[BoxField.BOXES]
        out_prob = tf.nn.softmax(classification_logits, -1)

        # Extract the target classes to approximate the classification cost
        # [batch_size, nb_class, num_detection]
        out_prob = tf.transpose(out_prob, [0, 2, 1])
        # [batch_size, nb_target, num_detection]
        cost = tf.gather_nd(out_prob, get_full_indices(y_true[BoxField.LABELS]))
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # [batch_size, num_detection, nb_target]
        cost_class = -cost

        gt_boxes = y_true[BoxField.BOXES]
        # Compute the L1 cost between boxes
        # [batch_size, nb_target, num_detection]
        cost_bbox = tf.norm(localization_pred[:, None] - gt_boxes[:, :, None], ord=1, axis=-1)

        # Compute the giou cost betwen boxes
        # [batch_size, nb_target, num_detection]
        # loss_giou= 1- giou but we approximate it with -giou
        cost_giou = -compute_giou(gt_boxes, localization_pred)

        # Final cost matrix
        cost_matrix = cost_bbox + cost_class + cost_giou
        # Linear sum assignment or the hungarian_assignment will look for the
        # minimum weight matching in bipartite graphs. In our case, we are
        # looking for the highest values of the cost matrix.
        return -cost_matrix
