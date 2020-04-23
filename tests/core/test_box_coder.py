# Copyright 2017 The TensorFlow Authors modified by Emilien Garreau. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np

from kerod.core import box_coder


def test_get_correct_relative_codes_after_encoding():
    boxes = tf.constant([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]])
    anchors = tf.constant([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]])
    expected_rel_codes = [[-0.5, -0.416666, -0.405465, -0.182321],
                          [-0.083333, -0.222222, -0.693147, -1.098612]]
    rel_codes = box_coder.encode_boxes_faster_rcnn(boxes, anchors, scale_factors=None)
    np.testing.assert_allclose(rel_codes, expected_rel_codes, rtol=1e-5)


def test_get_correct_relative_codes_after_encoding_with_scaling():
    boxes = tf.constant([[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]])
    anchors = tf.constant([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]])
    scale_factors = [2, 3, 4, 5]
    expected_rel_codes = [[-1., -1.25, -1.62186, -0.911608],
                          [-0.166667, -0.666667, -2.772588, -5.493062]]
    rel_codes = box_coder.encode_boxes_faster_rcnn(boxes, anchors, scale_factors=scale_factors)
    np.testing.assert_allclose(rel_codes, expected_rel_codes, rtol=1e-5)


def test_get_correct_boxes_after_decoding():
    anchors = tf.constant([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]])
    rel_codes = tf.constant([[-0.5, -0.416666, -0.405465, -0.182321],
                             [-0.083333, -0.222222, -0.693147, -1.098612]])
    expected_boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    boxes = box_coder.decode_boxes_faster_rcnn(rel_codes, anchors, scale_factors=None)
    np.testing.assert_allclose(boxes, expected_boxes, rtol=1e-5)


def test_get_correct_boxes_after_decoding_with_scaling():
    anchors = tf.constant([[15.0, 12.0, 30.0, 18.0], [0.1, 0.0, 0.7, 0.9]])
    rel_codes = tf.constant([[-1., -1.25, -1.62186, -0.911608],
                             [-0.166667, -0.666667, -2.772588, -5.493062]])
    scale_factors = [2, 3, 4, 5]
    expected_boxes = [[10.0, 10.0, 20.0, 15.0], [0.2, 0.1, 0.5, 0.4]]
    boxes = box_coder.decode_boxes_faster_rcnn(rel_codes, anchors, scale_factors=scale_factors)
    np.testing.assert_allclose(boxes, expected_boxes, rtol=1e-5)


def test_very_small_Width_nan_after_encoding():
    boxes = tf.constant([[10.0, 10.0, 10.0000001, 20.0]])
    anchors = tf.constant([[15.0, 12.0, 30.0, 18.0]])
    expected_rel_codes = [[-0.833333, 0., -21.128731, 0.510826]]
    rel_codes = box_coder.encode_boxes_faster_rcnn(boxes, anchors, scale_factors=None)
    np.testing.assert_allclose(rel_codes, expected_rel_codes, rtol=1e-5)
