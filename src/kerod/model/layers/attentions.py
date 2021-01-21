import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    """Allows the model to jointly attend to information from different representation subspaces.
    See reference: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

    Arguments:

    - *d_model*: The number of expected features in the decoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.

    Inputs:

    - *value*: A 3-D tensor of shape [batch_size, seq_len, depth_v]
    - *key*: A 3-D tensor of shape [batch_size, seq_len, depth]
    - *query*: A 3-D tensor of shape [batch_size, seq_len_q, depth]
    - *key_padding_mask*: A 2-D bool Tensor of shape [batch_size, seq_len] where
    False means padding and True means pixel from the original image.

    Output:

    A 3-D tensor of shape [batch_size, seq_len_q, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query = tf.keras.layers.Dense(d_model)
        self.key = tf.keras.layers.Dense(d_model)
        self.value = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, tgt: tf.Tensor, batch_size: int):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)
        """
        tgt = tf.reshape(tgt, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(tgt, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        """Arguments (inputs):

        - *value*: A 3-D tensor of shape [batch_size, seq_len, depth_v]
        - *key*: A 3-D tensor of shape [batch_size, seq_len, depth]
        - *query*: A 3-D tensor of shape [batch_size, seq_len_q, depth]
        - *key_padding_mask*: A 2-D bool Tensor of shape [batch_size, seq_len]. the positions with the
        value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

        Return:

        A 3-D tensor of shape [batch_size, seq_len_q, d_model]
        """

        v, k, q, key_padding_mask = inputs
        batch_size = tf.shape(q)[0]

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(self.query(q), batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(self.key(k), batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(self.value(v), batch_size)

        # scaled dot product attention
        # (batch_size, nh, seq_len_q, depth) x (batch_size, nh, depth, seq_len_k)
        # = (batch_size, nh, seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # Here we normalize by depth_k suppose K and Q are two matrices
        # with mean=0 and var=1. After QK^T will have a matrix with
        # mean=0 and var= 1 * depth_k. QK^T/sqrt(depth_k) => mean=0 and var=1
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(self.depth, self.compute_dtype))

        if key_padding_mask is not None:
            # Apply -inf if the pixels is a padding
            # False means padded so we take: not key_padding_mask
            scaled_attention_logits = tf.where(
                ~key_padding_mask[:, None, None],
                tf.zeros_like(scaled_attention_logits) + float('-inf'), scaled_attention_logits)
            # softmax is normalized on the last axis (seq_len_k) so that the scores
            # add up to 1.
            # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        scaled_attention = tf.matmul(attention_weights, v)

        # (batch_size, seq_len_q, nh, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        return self.dense(concat_attention)
