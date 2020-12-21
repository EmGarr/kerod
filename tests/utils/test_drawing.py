import pytest
from unittest import mock
from PIL import Image
from kerod.utils.drawing import BoxDrawer
import tensorflow as tf
import numpy as np

RED_IMAGE = Image.new('RGB', (300, 200), 'red')

@mock.patch('kerod.utils.drawing.plt.show', lambda: None)
@pytest.mark.parametrize("input_type", ["numpy", "tensorflow", "list"])
def test_box_drawer(input_type):
    drawer = BoxDrawer(["c1", "c2", "c3"])
    images = np.array(RED_IMAGE)[None]
    # In  float test the float handling
    images_information = np.array([[300., 200]])
    boxes = np.array([[[0, 0, .5, .5]]])
    labels = np.array([[1]])
    scores = np.array([[1]])
    num_valid_detections = np.array([1])

    if input_type == 'tensorflow':
        images = tf.constant(images)
        images_information = tf.constant(images_information)
        boxes = tf.constant(boxes)
        labels = tf.constant(labels)
        scores = tf.constant(scores)
        num_valid_detections = tf.constant(num_valid_detections)
    elif input_type == "list":
        labels = labels.tolist()
        scores = scores.tolist()
        num_valid_detections = num_valid_detections.tolist()

    drawer(images, images_information, boxes, labels, scores, num_valid_detections)
