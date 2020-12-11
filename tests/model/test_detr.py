import tensorflow as tf
import numpy as np
from kerod.model.detr import DeTr, DeTrResnet50Pytorch
from kerod.core.standard_fields import BoxField


def test_build_fpn_resnet50_faster_rcnn():
    num_classes = 2
    model = DeTrResnet50Pytorch(num_classes, num_queries=20)

    classification, bbox = model(tf.zeros((2, 200, 200, 3)))
    import pdb
    pdb.set_trace()
    # x['ground_truths'] = y
    # model(x, training=True)


def test_compute_loss_detr():
    num_classes = 2
    # shape [2, 2, 4]
    boxes_gt = tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                            [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]])

    # shape [batch_size, num_queries, 4] = [2, 3, 4]
    boxes_inf = tf.constant([[[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]],
                             [[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]]])

    ground_truths = {
        BoxField.BOXES: boxes_gt,
        BoxField.LABELS: tf.constant([[1, 0], [1, 0]], tf.int32),
        BoxField.WEIGHTS: tf.constant([[1, 0], [1, 1]], tf.float32),
        BoxField.NUM_BOXES: tf.constant([[2], [1]], tf.int32)
    }
    classification_logits = tf.constant(
        [[[0, 100, 0], [0, 100, 0], [0, 100, 0]], [[100, 0, 0], [0, -100, 0], [0, -100, 0]]],
        tf.float32)

    y_pred = {BoxField.BOXES: boxes_inf, BoxField.LABELS: classification_logits}

    detr = DeTr(num_classes, None)
    detr.compute_loss(ground_truths, y_pred)

    # Taken from test_box_ops.py::test_compute_giou_3d_tensor
    exp_iou = np.array([[[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                        [[2.0 / 16.0, 0, 6.0 / 400.0], [0, 0, 0]]])
    exp_term2 = np.array([[[4. / 20, 125 / 132, 0.], [12 / 28., 84 / 90., 0.]],
                          [[4. / 20, 125. / 132, 0.], [36. / 48, 224. / 225, 0.]]])
    exp_giou = 1 - (exp_iou - exp_term2)
    # (1 - 0.75, 1.947, 1- 0.0125), [1-0.75, 1-0.995555555, 1-0.015]

    import pdb
    pdb.set_trace()
