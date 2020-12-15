import tensorflow as tf


class PositionEmbeddingLearned(tf.keras.layers.Layer):
    """Absolute pos embedding, learned.

    Argument:

    - *output_dim*: Dimension of the dense embedding.
    """

    def __init__(self, output_dim=512):
        super().__init__()
        if output_dim % 2 != 0:
            raise ValueError("x an y embedding will be concatened to form a single vector "
                             f"of shape output_dim. Please use a multiple of 2 (e.g {output_dim})")
        dim = int(output_dim / 2)
        self.row_embed = tf.keras.layers.Embedding(50, dim)
        self.col_embed = tf.keras.layers.Embedding(50, dim)

    def call(self, inputs):
        """Based on the shape of the input tensor return a positional embedding.

        Argument:

        - *inputs*: A 4-D Tensor of shape [batch_size, h, w, channel]

        Return:

        The positional embedding a 4-D Tensor of shape [batch_size, h, w, output_dim]
        """
        batch_size, h, w, _ = tf.shape(inputs)
        i = tf.range(w)
        j = tf.range(h)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        single_img_emb = tf.concat([
            tf.tile(x_emb[None], (h, 1, 1)),
            tf.tile(y_emb[:, None], (1, w, 1)),
        ],
                                   axis=-1)

        batch_emb = tf.tile(single_img_emb[None], (batch_size, 1, 1, 1))
        return batch_emb
