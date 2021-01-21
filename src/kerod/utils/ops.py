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
    batch_ids = tf.range(0, batch_size)
    # Repeat the batch indice for every indices per batch
    # [0, 1, ..., n] => [[0, ..., 0], [1, ..., 1], ..., [n, ..., n]]
    batch_ids = tf.tile(batch_ids[:, None], [1, num_elements])
    # [[a1, ..., au], ...,[n1, ..., nu]] =>
    # [[[0, a1], ..., [0, au]], ..., [[n, n1], ..., [n, nu]]]
    full_indices = tf.concat(
        [batch_ids[..., None], indices[..., None]],
         axis=-1
    )
    return full_indices
