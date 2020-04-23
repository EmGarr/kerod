# Copyright 2017 The TensorFlow Authors modified by Emilien Garreau. All Rights Reserved.
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


import numpy as np
import tensorflow as tf

from kerod.utils import ops


def test_indices_to_dense_vector():
    size = 10000
    num_indices = np.random.randint(size)
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    np.testing.assert_array_equal(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype

def test_indices_to_dense_vector_size_at_inference():
    size = 5000
    num_indices = 250
    all_indices = np.arange(size)
    rand_indices = np.random.permutation(all_indices)[0:num_indices]

    expected_output = np.zeros(size, dtype=np.float32)
    expected_output[rand_indices] = 1.

    indicator = ops.indices_to_dense_vector(rand_indices,
                                            tf.shape(all_indices)[0])

    np.testing.assert_array_equal(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype

def test_indices_to_dense_vector_int():
    size = 500
    num_indices = 25
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.zeros(size, dtype=np.int64)
    expected_output[rand_indices] = 1

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(
        tf_rand_indices, size, 1, dtype=tf.int64)

    np.testing.assert_array_equal(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype

def test_indices_to_dense_vector_custom_values():
    size = 100
    num_indices = 10
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]
    indices_value = np.random.rand(1)
    default_value = np.random.rand(1)

    expected_output = np.float32(np.ones(size) * default_value)
    expected_output[rand_indices] = indices_value

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(
        tf_rand_indices,
        size,
        indices_value=indices_value,
        default_value=default_value)

    np.testing.assert_allclose(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype

def test_indices_to_dense_vector_all_indices_as_input():
    size = 500
    num_indices = 500
    rand_indices = np.random.permutation(np.arange(size))[0:num_indices]

    expected_output = np.ones(size, dtype=np.float32)

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    np.testing.assert_array_equal(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype

def test_indices_to_dense_vector_empty_indices_as_input():
    size = 500
    rand_indices = []

    expected_output = np.zeros(size, dtype=np.float32)

    tf_rand_indices = tf.constant(rand_indices)
    indicator = ops.indices_to_dense_vector(tf_rand_indices, size)

    np.testing.assert_array_equal(indicator, expected_output)
    assert indicator.dtype == expected_output.dtype




