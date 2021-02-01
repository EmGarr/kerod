import pytest
import numpy as np
import tensorflow as tf
from kerod.core.standard_fields import BoxField, DatasetField
from kerod.model.detr import DeTr, DeTrResnet50Pytorch, compute_detr_metrics


def test_build_detr():
    num_classes = 2
    model = DeTrResnet50Pytorch(num_classes, num_queries=20)

    # classification, bbox = model(tf.zeros((2, 200, 200, 3)))

    ground_truths = {
        BoxField.BOXES:
            tf.constant([[[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                         [[4.0, 3.0, 7.0, 5.0], [0, 0, 0, 0]]]),
        BoxField.LABELS:
            tf.constant([[1, 0], [1, 0]], tf.int32),
        BoxField.WEIGHTS:
            tf.constant([[1, 0], [1, 1]], tf.float32),
        BoxField.NUM_BOXES:
            tf.constant([[2], [1]], tf.int32)
    }

    base_lr = 0.02
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr)
    model.compile(optimizer=optimizer, loss=None)
    model.train_step(({
        DatasetField.IMAGES: tf.zeros((2, 200, 200, 3)),
        DatasetField.IMAGES_PMASK: tf.ones((2, 200, 200))
    }, ground_truths))
    ground_truths = {
        'bbox': tf.constant([[[0., 0., 199, 199]]], dtype=tf.float32),
        'label': tf.constant([[1]], dtype=tf.int32),
        'num_boxes': tf.constant([[1]], dtype=tf.int32),
        'weights': tf.constant([[1.]], dtype=tf.float32)
    }

    # Bug in the implementation of GIoU from tensorflow addons
    # If batch_size is 1 and sample_weight is provided bad shapping
    with pytest.raises(Exception):
        model.train_step(({DatasetField.IMAGES: tf.zeros((1, 200, 200, 3))}, ground_truths))


def test_compute_loss_detr():
    """The compute loss of DETR has been tested as well with the on from the official implementation. The losses are exactly the same."""
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

    # The tiling operation mimic the number of layers
    num_layers = 4
    y_pred = {
        BoxField.BOXES: tf.tile(boxes_inf, [1, num_layers, 1]),
        BoxField.SCORES: tf.tile(classification_logits, [1, num_layers, 1])
    }

    detr = DeTr(num_classes, None)
    detr.transformer_num_layers = num_layers
    loss = detr.compute_loss(ground_truths, y_pred, tf.constant([2., 2.]))
    # Since y_pred has been tiled we can know
    # the losses value by multiplying by the number of layers
    expected_giou = 1.31 * num_layers
    expected_l1 = 93.333336 * num_layers
    expected_scc = 91.36464 * num_layers
    np.testing.assert_allclose(loss, expected_scc + expected_l1 + expected_giou)


def test_compute_detr_metrics():
    y_pred = tf.constant(
        [
            # TP foreground   - TP background    -  TP background
            [[-100, 100, -100], [100, -100, -100], [100, -100, -100]],
            # FP background & FN predicted - TP foreground   - FP
            [[100, -100, -100], [-100, -100, 100], [-100, -100, 100]]
        ],
        tf.float32)

    y_true = tf.constant([[1, 0, 0], [2, 2, 1]])

    recall = compute_detr_metrics(y_true, y_pred)
    assert recall == 0.5
