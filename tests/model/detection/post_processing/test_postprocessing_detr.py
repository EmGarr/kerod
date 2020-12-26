import numpy as np
import tensorflow as tf
from kerod.core.box_ops import convert_to_center_coordinates
from kerod.model.post_processing.post_processing_detr import post_processing


def test_post_processing_batch_size2():
    logits = tf.constant([[[-100., 0, 100], [-100., 1000, -100]], [[4., 0, 3], [-100., 1000,
                                                                                -100]]])
    probs = tf.nn.softmax(logits, axis=-1)

    boxes = tf.constant([
        [[0, 0, 1, 1], [0, 0, 0.5, 0.5]],
        [[0, 0, 0.3, 0.3], [0, 0, 0.5, 0.5]],
    ])
    boxes = convert_to_center_coordinates(boxes)
    image_information = tf.constant([[200, 400], [400, 200]])
    image_padded_information = tf.constant([400, 400])
    boxes, scores, labels = post_processing(boxes, logits, image_information,
                                            image_padded_information)
    expected_labels = np.array([[1, 0], [0, 1]])
    expected_scores = np.array([
        [probs[0, 0, 2], probs[0, 1, 1]],
        [probs[1, 1, 1], probs[1, 0, 2]],
    ])
    expected_boxes = np.array([
        [[0, 0, 1, 1], [0, 0, 1., 0.5]],
        [[0, 0, 0.5, 1.], [0, 0, 0.3, 0.6]],
    ])
    np.testing.assert_array_equal(expected_labels, labels.numpy())
    np.testing.assert_almost_equal(expected_boxes, boxes.numpy())
    np.testing.assert_array_equal(expected_scores, scores.numpy())


def test_post_processing_singled_element():
    logits = tf.constant([[[4., 0, 3], [-100., 1000, -100]]])
    probs = tf.nn.softmax(logits, axis=-1)

    boxes = tf.constant([[[0, 0, 0.3, 0.3], [0, 0, 0.5, 0.5]]])
    boxes = convert_to_center_coordinates(boxes)

    image_information = tf.constant([[400, 200]])
    image_padded_information = tf.constant([400, 400])
    boxes, scores, labels = post_processing(boxes, logits, image_information,
                                            image_padded_information)
    expected_labels = np.array([[0, 1]])
    expected_scores = np.array([[probs[0, 1, 1], probs[0, 0, 2]]])
    expected_boxes = np.array([[[0, 0, 0.5, 1.], [0, 0, 0.3, 0.6]]])
    np.testing.assert_array_equal(expected_labels, labels.numpy())
    np.testing.assert_almost_equal(expected_boxes, boxes.numpy())
    np.testing.assert_array_equal(expected_scores, scores.numpy())
