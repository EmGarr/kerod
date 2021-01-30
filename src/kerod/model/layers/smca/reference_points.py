import tensorflow as tf


class SMCAReferencePoints(tf.keras.layers.Layer):
    """Multi head reference points from the paper [Fast Convergence of DETR with Spatially Modulated Co-Attention](https://arxiv.org/pdf/2101.07448.pdf).
    Based on the object queries will create a set of reference points which will
    allow to create a [spatial dynamical weight maps](./weight_map.py) in order to modulate
    the co-attention inside the transformer

    Arguments:
        hidden_dim: Positive integer, dimensionality of the hidden space.
        num_heads: Positive integer, each head starts from a head-shared center
            and then predicts a head-specific center offset
            and head specific scales.
    Call arguments:
        object_queries: A 3-D float32 Tensor of shape
            [batch_size, num_object_queries, d_model] small fixed number of
            learned positional embeddings input of the decoder.

    Call returns:
        reference_points: A float tensor of shape [batch_size, num_object_queries, num_heads, (y, x, w, h)].
        embedding_reference_points: A tensor of shape [batch_size, num_object_queries, num_heads, 2].
            The embedding of y and x without the sigmoid applied.
    """

    def __init__(self, hidden_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.xy_embed = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(2)  # (y_cent, x_cent)
        ])
        # Each head will have its proper focus weight and width
        self.yx_offset_hw_embed = tf.keras.layers.Dense(4 * num_heads)

    def call(self, object_queries):
        """
        Args:
            object_queries: A 3-D float32 Tensor of shape
                [batch_size, num_object_queries, d_model] small fixed number of
                learned positional embeddings input of the decoder.
        Returns:
            reference_points: A float tensor of shape [batch_size, num_object_queries, num_heads, (y, x, w, h)].
            embedding_reference_points: A tensor of shape [batch_size, num_object_queries, num_heads, 2].
                The embedding of y and x without the sigmoid applied.
        """

        yx_pre_sigmoid = self.xy_embed(object_queries)  # [bs, num_queries, 2]
        yx = tf.nn.sigmoid(yx_pre_sigmoid)

        # Where y and x are offsets predictions for 'yx' (above)
        # h and w are the scales predictions
        yx_offset_hw = self.yx_offset_hw_embed(object_queries)  # [bs, num_queries, head * 4]

        batch_size = tf.shape(object_queries)[0]
        num_queries = tf.shape(object_queries)[1]

        # Add offset coordinates to yx and concatenate with scale predictions
        # yx => [bs, num_queries, head, 2]
        yx = tf.tile(yx[:, :, None], (1, 1, self.num_heads, 1))
        # yx_offset_hw => [bs, num_queries, head, 4]
        yx_offset_hw = tf.reshape(yx_offset_hw, (batch_size, -1, self.num_heads, 4))

        yxhw = tf.concat([yx, tf.zeros((batch_size, num_queries, self.num_heads, 2))], axis=-1)
        return yxhw + yx_offset_hw, yx_pre_sigmoid
