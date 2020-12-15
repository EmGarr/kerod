import numpy as np
import tensorflow as tf

from kerod.core.standard_fields import BoxField
from kerod.core import target_assigner as targetassigner
from kerod.core.similarity import IoUSimilarity
from kerod.core.matcher import Matcher
from kerod.core.standard_fields import LossField


def encode_mean_stddev(boxes, anchors, stddev=0.1):
    return (boxes - anchors) / stddev


def test_assign_multiclass_with_groundtruth_weights():
    similarity_calc = IoUSimilarity()
    matcher = Matcher([0.5], [0, 1])
    box_coder = encode_mean_stddev
    target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

    anchor_means = np.array(
        [[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]],
         [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 0.8], [0, 0.5, .5, 1.0], [.75, 0, 1.0, .25]]],
        dtype=np.float32)
    groundtruth_boxes = np.array([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9], [.75, 0, .95, .27]],
                                  [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.9, 0.9], [.75, 0, .95, .27]]],
                                 dtype=np.float32)
    groundtruth_labels = np.array([[1, 5, 3], [1, 5, 3]], dtype=np.float32)
    groundtruth_weights = np.array([[0.3, 0., 0.5], [0.3, 0., 0.5]], dtype=np.float32)
    num_boxes = np.array([[3], [3]], np.int32)

    # background class gets weight of 1.
    exp_cls_weights = [[0.3, 0., 1, 0.5], [0.3, 0., 1, 0.5]]
    exp_reg_weights = [[0.3, 0., 0., 0.5], [0.3, 0., 0., 0.5]]  # background class gets weight of 0.

    gt_box_batch = {
        BoxField.BOXES: groundtruth_boxes,
        BoxField.LABELS: groundtruth_labels,
        BoxField.WEIGHTS: groundtruth_weights,
        BoxField.NUM_BOXES: num_boxes
    }

    targets, weights = target_assigner.assign({BoxField.BOXES: anchor_means}, gt_box_batch)
    cls_targets = targets[LossField.CLASSIFICATION]
    cls_weights = weights[LossField.CLASSIFICATION]
    reg_targets = targets[LossField.LOCALIZATION]
    reg_weights = weights[LossField.LOCALIZATION]

    np.testing.assert_array_almost_equal(cls_weights, exp_cls_weights)
    np.testing.assert_array_almost_equal(reg_weights, exp_reg_weights)


def test_batch_assign_multiclass_targets_with_padded_groundtruth():

    similarity_calc = IoUSimilarity()
    matcher = Matcher([0.5], [0, 1])
    box_coder = encode_mean_stddev
    target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

    groundtruth_boxes = np.array([
        [[0., 0., 0.2, 0.2], [0., 0., 0., 0.], [0, 0, 0, 0]],
        [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842], [0, 0, 0, 0]],
    ],
                                 dtype=np.float32)
    groundtruth_weights = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    num_boxes = np.array([[2], [3]], np.int32)
    class_targets = np.array([[1, 0, 0], [3, 2, 0]], dtype=np.float32)

    anchor_means = np.array([[[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]],
                             [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]]],
                            dtype=np.float32)

    exp_cls_targets = [[1, 0, 0, 0], [0, 3, 2, 0]]
    exp_cls_weights = [[1, 1, 1, 1], [1, 1, 1, 1]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                       [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

    gt_box_batch = {
        BoxField.BOXES: groundtruth_boxes,
        BoxField.LABELS: class_targets,
        BoxField.WEIGHTS: groundtruth_weights,
        BoxField.NUM_BOXES: num_boxes
    }

    targets, weights = target_assigner.assign({BoxField.BOXES: anchor_means}, gt_box_batch)
    cls_targets = targets[LossField.CLASSIFICATION]
    cls_weights = weights[LossField.CLASSIFICATION]
    reg_targets = targets[LossField.LOCALIZATION]
    reg_weights = weights[LossField.LOCALIZATION]

    np.testing.assert_array_almost_equal(cls_targets, exp_cls_targets)
    np.testing.assert_array_almost_equal(cls_weights, exp_cls_weights)
    np.testing.assert_array_almost_equal(reg_targets, exp_reg_targets)
    np.testing.assert_array_almost_equal(reg_weights, exp_reg_weights)


def test_target_assigner_with_padded_ground_truths():
    similarity_calc = IoUSimilarity()
    matcher = Matcher([0.5], [0, 1])
    box_coder = encode_mean_stddev
    target_assigner = targetassigner.TargetAssigner(similarity_calc, matcher, box_coder)

    groundtruth_boxes = np.array([
        [[0., 0., 0.2, 0.2], [0., 0., 0., 0.], [0, 0, 0, 0]],
        [[0, 0.25123152, 1, 1], [0.015789, 0.0985, 0.55789, 0.3842], [0, 0, 0, 0]],
    ],
                                 dtype=np.float32)
    groundtruth_weights = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.float32)
    class_targets = np.array([[1, 0, 0], [3, 2, 0]], dtype=np.float32)
    num_boxes = np.array([[1], [3]], dtype=np.int32)

    anchor_means = np.array([[[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]],
                             [[0, 0, .25, .25], [0, .25, 1, 1], [0, .1, .5, .5], [.75, .75, 1, 1]]],
                            dtype=np.float32)

    gt_box_batch = {
        BoxField.BOXES: groundtruth_boxes,
        BoxField.LABELS: class_targets,
        BoxField.WEIGHTS: groundtruth_weights,
        BoxField.NUM_BOXES: num_boxes
    }

    targets, weights = target_assigner.assign({BoxField.BOXES: anchor_means}, gt_box_batch)
    cls_targets = targets[LossField.CLASSIFICATION]
    cls_weights = weights[LossField.CLASSIFICATION]
    reg_targets = targets[LossField.LOCALIZATION]
    reg_weights = weights[LossField.LOCALIZATION]

    exp_cls_targets = np.array([[1, 0, 0, 0], [0, 3, 2, 0]])
    exp_cls_weights = [[1, 1, 1, 1], [1, 1, 1, 1]]
    exp_reg_targets = [[[0, 0, -0.5, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                       [[0, 0, 0, 0], [0, 0.01231521, 0, 0],
                        [0.15789001, -0.01500003, 0.57889998, -1.15799987], [0, 0, 0, 0]]]
    exp_reg_weights = [[1, 0, 0, 0], [0, 1, 1, 0]]

    np.testing.assert_array_almost_equal(cls_targets, exp_cls_targets)
    np.testing.assert_array_almost_equal(cls_weights, exp_cls_weights)
    np.testing.assert_array_almost_equal(reg_targets, exp_reg_targets)
    np.testing.assert_array_almost_equal(reg_weights, exp_reg_weights)

# NOTE is handling empty ground_truths useful?
# def test_batch_assign_empty_groundtruth(self):

#         def graph_fn(anchor_means, groundtruth_box_corners, gt_class_targets):
#             groundtruth_boxlist = {
#                 BoxField.BOXES: groundtruth_box_corners,
#                 BoxField.LABELS: gt_class_targets
#             }
#             gt_box_batch = [groundtruth_boxlist]
#             anchors_boxlist = {BoxField.BOXES: anchor_means}

#             multiclass_target_assigner = self._get_target_assigner()
#             num_classes = 3
#             unmatched_class_label = tf.constant([1] + num_classes * [0], tf.float32)
#             targets, weights, _ = targetassigner.batch_assign_targets(
#                 multiclass_target_assigner, anchors_boxlist, gt_box_batch, unmatched_class_label)
#             cls_targets = targets[LossField.CLASSIFICATION]
#             cls_weights = weights[LossField.CLASSIFICATION]
#             reg_targets = targets[LossField.LOCALIZATION]
#             reg_weights = weights[LossField.LOCALIZATION]
#             return (cls_targets, cls_weights, reg_targets, reg_weights)

#         groundtruth_box_corners = np.zeros((0, 4), dtype=np.float32)
#         anchor_means = np.array([[0, 0, .25, .25], [0, .25, 1, 1]], dtype=np.float32)
#         exp_cls_targets = [[[1, 0, 0, 0], [1, 0, 0, 0]]]
#         exp_cls_weights = [[1, 1]]
#         exp_reg_targets = [[[0, 0, 0, 0], [0, 0, 0, 0]]]
#         exp_reg_weights = [[0, 0]]
#         num_classes = 3
#         pad = 1
#         gt_class_targets = np.zeros((0, num_classes + pad), dtype=np.float32)

#         (cls_targets_out, cls_weights_out, reg_targets_out, reg_weights_out) = graph_fn(
#             anchor_means, groundtruth_box_corners, gt_class_targets)
#         self.assertAllClose(cls_targets_out, exp_cls_targets)
#         self.assertAllClose(cls_weights_out, exp_cls_weights)
#         self.assertAllClose(reg_targets_out, exp_reg_targets)
#         self.assertAllClose(reg_weights_out, exp_reg_weights)

if __name__ == '__main__':
    tf.test.main()
