import tensorflow as tf
import numpy as np

from od.model.post_processing import post_process_fast_rcnn_boxes, post_process_rpn


def test_post_process_fast_rcnn_boxes():
    # The mocking allows to make the test deterministic

    num_classes = 3
    anchors1 = [
        [0, 0, 1, 1],
        [0, 0.1, 1, 1.1],
        [0, -0.1, 1, 0.9],
        [0, 10, 1, 11],
        [0, 10.1, 1, 11.1],
        [0, 100, 1, 101],
        [0, 1000, 1, 1002],
        [0, 999, 2, 1002.7],
    ]
    anchors2 = [
        [0, 0, 1, 1],
        [0, 0.1, 1, 1.1],
        [0, -0.1, 10, 1],
        [0, 10, 1, 21],
        [0, 32.1, 1, 111.1],
        [0, 100, 1, 101],
        [0, 1000, 1, 10020],
        [0, 999, 2, 1002.7],
    ]

    anchors = tf.constant([anchors1, anchors2], tf.float32)
    localization_pred = tf.zeros((2, 8, 4 * (num_classes - 1)))
    # We will get every boxes over 0.005
    classification_pred = [[0.09, .9, 0.01], [.2, .75, 0.05], [0.39, .6, 0.01], [0.05, .95, 0],
                           [0.49, .5, 0.01], [0.69, .3, 0.01], [0.14, .01, .85], [0.49, .01, .5]]
    # 95, 9,
    classification_pred = tf.constant([classification_pred, classification_pred])

    image_information = tf.constant([[200, 1001], [200, 1001]])
    exp_boxes = [[0, 10, 1, 11], [0, 0, 1, 1], [0, 1000, 1, 1001], [0, 999, 2, 1001]]
    exp_scores = [.95, .9, .85, .5]
    exp_labels = [0, 0, 1, 1]

    nmsed_boxes, nmsed_scores, nmsed_labels, valid_detections = post_process_fast_rcnn_boxes(
        classification_pred, localization_pred, anchors, image_information, num_classes)

    assert valid_detections[0] == 4

    np.testing.assert_array_equal(nmsed_boxes[0, :4], exp_boxes)
    np.testing.assert_array_almost_equal(nmsed_scores[0, :4], exp_scores)
    np.testing.assert_array_equal(nmsed_labels[0, :4], exp_labels)


def test_post_process_rpn():
    anchors1 = [
        [0, 0, 1, 1],
        [0, 0.1, 1, 1.1],
        [0, -0.1, 1, 0.9],  #3
        [0, 10, 1, 11],  #1
        [0, 10.1, 1, 11.1],
        [0, 100, 1, 101],  #5
        [0, 1000, 1, 1002],  #2
        [0, 999, 2, 1002.7],  #4
    ]
    anchors2 = [
        [0, 0, 1, 1],
        [0, 0.1, 1, 1.1],
        [0, -0.1, 10, 1],
        [0, 10, 1, 21],
        [0, 32.1, 1, 111.1],
        [0, 100, 1, 101],
        [0, 1000, 1, 10020],
        [0, 999, 2, 1002.7],
    ]

    anchors = tf.constant([anchors1, anchors2], tf.float32)
    localization_pred = tf.zeros((2, 8, 4))
    # We will get every boxes over 0.005
    classification_pred = [[.9, 0.01], [.75, 0.35], [0.39, .61], [0.05, .95], [0.5, .5],
                           [0.69, .31], [0.15, .85], [0.49, .51]]
    # 95, 9,
    classification_pred = tf.constant([classification_pred, classification_pred])

    image_information = tf.constant([[200, 1001], [200, 1001]])
    exp_boxes = [[0, 10, 1, 11], [0, 1000, 1, 1001], [0, 0, 1, 0.9], [0, 999, 2, 1001],
                 [0, 100, 1, 101]]
    exp_scores = [.95, .85, .61, 0.51, 0.31]
    nmsed_boxes, nmsed_scores = post_process_rpn(classification_pred, localization_pred, anchors,
                                                 image_information, 8, 5)

    np.testing.assert_array_almost_equal(nmsed_boxes[0], exp_boxes)
    np.testing.assert_array_almost_equal(nmsed_scores[0], exp_scores)
