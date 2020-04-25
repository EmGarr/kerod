# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

def indices_to_dense_vector(indices, size, indices_value=1., default_value=0, dtype=tf.float32):
    """Creates dense vector with indices set to specific value and rest to zeros.

    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])

    Arguments:

    - *indices*: 1d Tensor with integer indices which are to be set to
            indices_values.
    - *size*: scalar with size (integer) of output Tensor.
    - *indices_value*: values of elements specified by indices in the output vector
    - *default_value*: values of other elements in the output vector.
    - *dtype*: data type.

    Returns:

    A dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    size = tf.cast(size, dtype=tf.int32)
    zeros = tf.ones([size], dtype=dtype) * default_value
    values = tf.ones_like(indices, dtype=dtype) * indices_value

    return tf.dynamic_stitch([tf.range(size), tf.cast(indices, dtype=tf.int32)], [zeros, values])


def item_assignment(tensor: tf.Tensor, indicator: tf.Tensor, val):
    """Set the indicated fields of tensor to val.

    ```python
    tensor = tf.constant([1, 2, 3, 4])
    # won't work in tensorflow
    tensor[tensor == 2] = 1

    tensor = item_assignment(tensor, tensor == 2, 1)
    ```

    Arguments:

    - *tensor*: A tensor without shape constraint.
    - *indicator*: boolean tensor with same shape as `tensor`.
    - *val*: scalar with value to set.
    """
    indicator = tf.cast(indicator, tensor.dtype)
    return tensor * (1 - indicator) + val * indicator


def get_full_indices(indices):
    """ This operation allows to extract full indices from indices.
    These full-indices have the proper format for gather_nd operations.

    Example:

    ```python
    indices = [[0, 1], [2, 3]]
    full_indices = [[[0, 0], [0, 1]], [[1, 2], [1, 3]]]
    ```

    Arguments:

    - *indices*: Indices without their assorciated batch format [batch_size, k].

    Returns:

    Full-indices tensor [batch_size, k, 2]
    """
    batch_size = tf.shape(indices)[0]
    num_elements = tf.shape(indices)[1]
    # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    my_range = tf.expand_dims(tf.range(0, batch_size), 1)  # will be [[0], [1]]
    my_range_repeated = tf.tile(my_range, [1, num_elements])  # will be [[0, 0], [1, 1]]
    # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    full_indices = tf.concat(
        [tf.expand_dims(my_range_repeated, 2),
         tf.expand_dims(indices, 2)], axis=2
    )
    return full_indices
