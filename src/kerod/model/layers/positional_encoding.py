import math
import tensorflow as tf


class PositionEmbeddingLearned(tf.keras.layers.Layer):
    """Absolute pos embedding, learned.

    Argument:

    - *output_dim*: Dimension of the dense embedding.
    """

    def __init__(self, output_dim=512, **kwargs):
        super().__init__(**kwargs)
        if output_dim % 2 != 0:
            raise ValueError("x an y embedding will be concatened to form a single vector "
                             f"of shape output_dim. Please use a multiple of 2 (e.g {output_dim})")
        self.dim = int(output_dim / 2)
        self.row_embed = tf.keras.layers.Embedding(50, self.dim)
        self.col_embed = tf.keras.layers.Embedding(50, self.dim)

    def call(self, inputs):
        """Based on the shape of the input tensor return a positional embedding.

        Argument:

        - *inputs*: A 4-D Tensor of shape [batch_size, h, w, channel]

        Return:

        The positional embedding a 4-D Tensor of shape [batch_size, h, w, output_dim]
        """
        batch_size, h, w = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
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


class PositionEmbeddingSine(tf.keras.layers.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    ```python
    import matplotlib.pyplot as plt
    from kerod.model.layers.positional_encoding import PositionEmbeddingSine

    dim = 128
    embedding =  PositionEmbeddingSine(dim)
    pos_encoding = embedding(tf.ones((1, 10, 10)))

    plt.pcolormesh(tf.reshape(pos_encoding, (1, -1, dim))[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, dim))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    ```

    ![Visualization Positional Encoding](https://raw.githubusercontent.com/EmGarr/cv/master/ressources/2d_pos_encoding.png)
    """

    def __init__(self, output_dim=64, temperature=10000):
        super().__init__()
        self.temperature = temperature
        self.scale = 2 * math.pi
        if output_dim % 2 != 0:
            raise ValueError("x an y embedding will be concatened to form a single vector "
                             f"of shape output_dim. Please use a multiple of 2 (e.g {output_dim})")
        self.dim = int(output_dim / 2)
        dim_t = tf.range(self.dim, dtype=tf.float32)
        self.dim_t = self.temperature**(2 * (dim_t // 2) / self.dim)

    def call(self, masks):
        """From a masks tensor compute the positional encoding

        Arguments:

        - *masks*: A tensor of bool and shape [batch_size, w, h] where False means
        padding and True pixel from the image

        Return:

        *encoding*: A tensor of float and shape [batch_size, w, h, output_dim]
        """
        masks = tf.cast(masks, self.compute_dtype)
        y_embed = tf.math.cumsum(masks, axis=1)
        x_embed = tf.math.cumsum(masks, axis=2)
        # Normalize x_embed and y_embed by the maximum values of the cumsum
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[..., None] / self.dim_t
        pos_y = y_embed[..., None] / self.dim_t
        pos_x = tf.stack([
            tf.math.sin(pos_x[..., 0::2]),
            tf.math.cos(pos_x[..., 1::2]),
        ], axis=4)

        pos_y = tf.stack([
            tf.math.sin(pos_y[..., 0::2]),
            tf.math.cos(pos_y[..., 1::2]),
        ], axis=4)

        batch_size, w, h = tf.shape(masks)[0], tf.shape(masks)[1], tf.shape(masks)[2]
        pos_x = tf.reshape(pos_x, (batch_size, w, h, -1))
        pos_y = tf.reshape(pos_y, (batch_size, w, h, -1))

        pos_emb = tf.concat([pos_y, pos_x], axis=-1)
        return pos_emb
