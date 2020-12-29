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
    - *key_padding_mask*: A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
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
        - *key_padding_mask*: A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
        False means padding and True means pixel from the original image.

        Return:

        A 3-D tensor of shape [batch_size, seq_len_q, d_model]
        """

        v, k, q, key_padding_mask = inputs
        seq_len_q = tf.shape(q)[1]
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
        depth_k = tf.cast(tf.shape(k)[-1], self.compute_dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(depth_k)

        if key_padding_mask is not None:
            # Tile the tensor from [batch_size, seq_len] to [batch_size, nh, seq_len_q, seq_len]
            key_padding_mask = tf.tile(key_padding_mask[:, None, None],
                                       [1, self.num_heads, seq_len_q, 1])

            # Apply -inf if the pixels is a padding
            # False means padded so we take: not key_padding_mask
            scaled_attention_logits = tf.where(
                ~key_padding_mask,
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


class EncoderLayer(tf.keras.layers.Layer):
    """Will build a TransformerEncoderLayer according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Arguments:

    - *d_model*: The number of expected features in the encoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dim_feedforward*: The dim of the feedforward neuralnetworks in the EncoderLayer.
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.

    Inputs:

    - *src*: A 3-D Tensor of float32 and shape [batch_size, M, dim]
    the sequence to the encoder layer
    - *pos_embed*: A 3-D Tensor of float32 and shape [batch_size, N, dim]
    positional encoding of the encoder
    - *mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
    False means padding and True means pixel from the original image.

    Output:

    A 3-D Tensor of float32 and shape [batch_size, M, d_model]
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """ Forward of the EncoderLayer

        Arguments (inputs):

        - *src*: A 3-D Tensor of float32 and shape [batch_size, M, dim]
        the sequence to the encoder layer
        - *pos_embed*: A 3-D Tensor of float32 and shape [batch_size, N, dim]
        positional encoding of the encoder
        - *mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
        False means padding and True means pixel from the original image.

        Return:
        A 3-D Tensor of float32 and shape [batch_size, M, d_model]
        """

        src, pos_emb, mask = inputs
        x_pos_emb = src + pos_emb
        attn_output = self.mha((src, x_pos_emb, x_pos_emb, mask),
                               training=training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(src + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """Will build a TransformerDecoderLayer according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Arguments:

    - *d_model*: The number of expected features in the encoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dim_feedforward*: The dim of the feedforward neuralnetworks in the EncoderLayer.
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.


    Inputs:

    - *tgt*: A 3-D Tensor of float32 and shape [batch_size, M, d_model] the sequence of the decoder
    - *memory*: A 3-D Tensor of float32 and shape [batch_size, N, d_model] the sequence
    from the last layer of the encoder (memory)
    - *pos_embed*: A 3-D Tensor of float32 and shape [batch_size, N, d_model] positional encoding
    of the encoder
    - *object_queries*: A 3-D Tensor of float32 and shape [batch_size, M, d_model]

    Output:

    A 3-D Tensor of float32 and shape [batch_size, M, d_model]
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """ Forward of the DecoderLayer

        Arguments (inputs):

        - *tgt*: A 3-D Tensor of float32 and shape [batch_size, M, d_model] the sequence of the decoder
        - *memory*: A 3-D Tensor of float32 and shape [batch_size, N, d_model] the sequence
        from the last layer of the encoder (memory)
        - *pos_embed*: A 3-D Tensor of float32 and shape [batch_size, N, d_model] positional encoding
        of the encoder
        - *object_queries*: A 3-D Tensor of float32 and shape [batch_size, M, d_model]

        Return:
        A 3-D Tensor of float32 and shape [batch_size, M, d_model]
        """
        tgt, memory, pos_embed, object_queries, mask = inputs

        tgt_object_queries = tgt + object_queries
        # (batch_size, M, d_model)
        attn1 = self.mha1((tgt, tgt_object_queries, tgt_object_queries, None), training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + tgt)

        # (batch_size, M, d_model)
        attn2 = self.mha2((memory, memory + pos_embed, out1 + object_queries, mask),
                          training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, M, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, M, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, M, d_model)
        return out3


class Encoder(tf.keras.layers.Layer):
    """Will build a TransformerEncoder according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    TransformerEncoder is a stack of N encoder layers.

    Arguments:

    - *num_layers*: the number of sub-layers in the encoder.
    - *d_model*: The number of expected features in the encoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dim_feedforward*: The dim of the feedforward neuralnetworks in the EncoderLayer.
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.

    Inputs:

    - *src*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model]
    the sequence to the encoder.
    - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model].
    Positional spatial positional encoding matching the flatten_tensor.
    - *mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
    False means padding and True means pixel from the original image.

    Output:

    A 3-D Tensor of float32 and shape [batch_size, seq_len_enc, d_model]
    """

    def __init__(self,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training=None):
        """
        Arguments (inputs):

        - *src*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model]
        the sequence to the encoder.
        - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model].
        Positional spatial positional encoding matching the flatten_tensor.
        - *mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
        False means padding and True means pixel from the original image.

        Return:

        A 3-D Tensor of float32 and shape [batch_size, seq_len_enc, d_model]
        """
        src, pos_embed, mask = inputs

        for enc in self.enc_layers:
            src = enc((src, pos_embed, mask), training=training)

        return src  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """Will build a TransformerDecoder according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
    TransformerDecoder is a stack of N decoder layers.

    Arguments:

    - *num_layers*: the number of sub-layers in the decoder.
    - *d_model*: The number of expected features in the decoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dim_feedforward*: The dim of the feedforward neuralnetworks in the DecoderLayer
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.

    Inputs:

    - *memory*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model].
    the sequence from the last layer of the encoder.
    - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model]. Positional spatial positional encoding matching the flatten_tensor.
    - *object_queries*: A 3-D float32 Tensor of shape [batch_size, num_queries, d_model] small fixed number of learned positional embeddings input of the decoder.
    - *memory_padding_mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where False means padding and True means pixel from the original image.

    Output:

    A 3-D Tensor of float32 and shape [batch_size, h , d_model]
    where h is num_queries * num_layers if training is true and num_queries
    if training is set to False.
    """

    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout_rate=0.1, **kwargs):

        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training=None):
        """
        Arguments (inputs):

        - *memory*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model].
        the sequence from the last layer of the encoder.
        - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, seq_len_enc, d_model]. Positional spatial positional encoding matching the flatten_tensor.
        - *object_queries*: A 3-D float32 Tensor of shape [batch_size, num_queries, d_model] small fixed number of learned positional embeddings input of the decoder.
        - *memory_padding_mask*:  A 2-D bool Tensor of shape [batch_size, seq_len_enc] where False means padding and True means pixel from the original image.

        Return:

        A 3-D Tensor of float32 and shape [batch_size, h , d_model]
        where h is num_queries * num_layers if training is true and num_queries
        if training is set to False.
        """
        memory, pos_embed, object_queries, memory_padding_mask = inputs

        # At the beginning we set target to 0
        # In the first decoder layer Q and K will be equal
        # to tgt + object_queries=object_queries
        tgt = tf.zeros_like(object_queries)
        layers_output = []
        for layer in self.dec_layers:
            tgt = layer((tgt, memory, pos_embed, object_queries, memory_padding_mask),
                        training=training)
            tgt = self.layer_norm(tgt)
            if training:
                layers_output.append(tgt)
        if training:
            return tf.concat(layers_output, axis=1)
        return tgt


class Transformer(tf.keras.Model):
    """Will build a Transformer according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Arguments:

    - *num_layers*: the number of sub-layers in the decoder and the encoder.
    - *d_model*: The number of expected features in the encoder/decoder inputs
    - *num_heads*: The number of heads in the multiheadattention models.
    - *dim_feedforward*: The dim of the feedforward neuralnetworks in the EncoderLayer and DecoderLayer
    - *dropout_rate*: Float between 0 and 1. Fraction of the input units to drop.
    The same rate is shared in all the layers using dropout in the transformer.

    Inputs:

    - *flatten_tensor*: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
    It represents the flatten output tensor of the backbone.
    - *mask*:  A 2-D bool Tensor of shape [batch_size, H * W] where False means
    padding and True means pixel from the original image.
    - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
    Positional spatial positional encoding matching the flatten_tensor.
    - *object_queries*: A 3-D float32 Tensor of shape [batch_size, num_object_queries, d_model]
    small fixed number of learned positional embeddings input of the decoder.

    Outputs:

    - *decoder_output*: 3-D float32 Tensor of shape [batch_size, num_object_queries, d_model]
    Due to the multi-head attention architecture in the transformer model,
    the output sequence length of a transformer is same as the input
    sequence (i.e. target) length of the decoder (num_object_queries).
    - *encoder_output*: A 3-D Tensor of float32 and shape [batch_size, h , d_model]
    where h is num_queries * num_layers if training is true and num_queries
    if training is set to False.
    """

    def __init__(self,
                 num_layers=6,
                 d_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout_rate=0.1,
                 **kwargs):

        super().__init__(**kwargs)

        self.encoder = Encoder(num_layers, d_model, num_heads, dim_feedforward, dropout_rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dim_feedforward, dropout_rate)

    def call(self, inputs, training=None):
        """
        Arguments (inputs):

        - *flatten_tensor*: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
        It represents the flatten output tensor of the backbone.
        - *mask*:  A 2-D bool Tensor of shape [batch_size, H * W] where False means
        padding and True means pixel from the original image.
        - *pos_embed*: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
        Positional spatial positional encoding matching the flatten_tensor.
        - *object_queries*: A 3-D float32 Tensor of shape [batch_size, num_object_queries, d_model]
        small fixed number of learned positional embeddings input of the decoder.

        Returns:

        - *decoder_output*: 3-D float32 Tensor of shape [batch_size, num_object_queries, d_model]
        Due to the multi-head attention architecture in the transformer model,
        the output sequence length of a transformer is same as the input
        sequence (i.e. target) length of the decoder (num_object_queries).
        - *encoder_output*: 3-D float32 Tensor of shape [batch_size, batch_size, d_model]
        """
        flatten_tensor, mask, pos_embed, object_queries = inputs
        # (batch_size, inp_seq_len, d_model)
        memory = self.encoder((flatten_tensor, pos_embed, mask), training=training)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output = self.decoder((memory, pos_embed, object_queries, mask), training=training)

        return dec_output, memory
