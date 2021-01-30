import tensorflow as tf
from kerod.model.layers import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    """Will build a TransformerEncoderLayer according to the paper
    [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

    Arguments:
        d_model: The number of expected features in the encoder inputs
        num_heads: The number of heads in the multiheadattention models.
            dim_feedforward: The dim of the feedforward neuralnetworks in the EncoderLayer.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            The same rate is shared in all the layers using dropout in the transformer.

    Inputs:
        src: A 3-D Tensor of float32 and shape [batch_size, M, dim]
            the sequence to the encoder layer
        pos_embed: A 3-D Tensor of float32 and shape [batch_size, N, dim]
            positional encoding of the encoder
        key_padding_mask:  [Optional] A 2-D bool Tensor of shape [batch_size, seq_len_enc] where
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

    def call(self, src, pos_emb, key_padding_mask=None, training=None):
        """Forward of the EncoderLayer

        Args:
            src: A 3-D Tensor of float32 and shape [batch_size, M, dim]
                the sequence to the encoder layer
            pos_embed: A 3-D Tensor of float32 and shape [batch_size, N, dim]
                positional encoding of the encoder
            key_padding_mask: [Optional] A 2-D bool Tensor of shape [batch_size, seq_len_enc]
                where False means padding and True means pixel from the original image.

        Returns:
            A 3-D Tensor of float32 and shape [batch_size, M, d_model]
        """
        x_pos_emb = src + pos_emb
        attn_output = self.mha(value=src,
                               key=x_pos_emb,
                               query=x_pos_emb,
                               key_padding_mask=key_padding_mask,
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
        d_model: The number of expected features in the encoder inputs
        num_heads: The number of heads in the multiheadattention models.
        dim_feedforward: The dim of the feedforward neuralnetworks in the DecoderLayer.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            The same rate is shared in all the layers using dropout in the transformer.

    Inputs:
        dec_out: A 3-D Tensor of float32 and shape [batch_size, M, d_model]
            the sequence of the decoder
        memory: A 3-D Tensor of float32 and shape [batch_size, N, d_model]
            the sequence from the last layer of the encoder (memory)
        pos_embed: A 3-D Tensor of float32 and shape [batch_size, N, d_model]
            positional encoding of the encoder
        object_queries: A 3-D Tensor of float32 and shape [batch_size, M, d_model]
        key_padding_mask: A 2-D bool Tensor of shape [batch_size, seq_len].
            The positions with the value of ``True`` will be ignored while
            the position with the value of ``False`` will be unchanged.
        coattn_mask:  A 4-D float tensor of shape [batch_size, num_heads, M, N, seq_len].
            If provided, it will be added to the attention weight at
            the coattention step (memory x self_attn).

    Outputs:
        A 3-D Tensor of float32 and shape [batch_size, M, d_model]
    """

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate=dropout_rate)

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

    def call(self,
             dec_out,
             memory,
             pos_embed,
             object_queries,
             key_padding_mask=None,
             coattn_mask=None,
             training=None):
        """ Forward of the DecoderLayer

        Arguments:
            dec_out: A 3-D Tensor of float32 and shape [batch_size, M, d_model] the sequence of the decoder
            memory: A 3-D Tensor of float32 and shape [batch_size, N, d_model] the sequence
                from the last layer of the encoder (memory)
            pos_embed: A 3-D Tensor of float32 and shape [batch_size, N, d_model] positional encoding
                of the encoder
            object_queries: A 3-D Tensor of float32 and shape [batch_size, M, d_model]
            key_padding_mask: A 2-D bool Tensor of shape [batch_size, seq_len]. the positions with the
                value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            coattn_mask:  A 4-D float tensor of shape [batch_size, num_heads, M, N, seq_len].
                If provided, it will be added to the attention weight at the coattention step (memory x self_attn)

        Return:
            A 3-D Tensor of float32 and shape [batch_size, M, d_model]
        """
        tgt_object_queries = dec_out + object_queries
        # (batch_size, M, d_model)
        self_attn = self.mha1(value=dec_out,
                              key=tgt_object_queries,
                              query=tgt_object_queries,
                              training=training)
        self_attn = self.dropout1(self_attn, training=training)
        self_attn = self.layernorm1(self_attn + dec_out)

        # (batch_size, M, d_model)
        co_attn = self.mha2(value=memory,
                            key=memory + pos_embed,
                            query=self_attn + object_queries,
                            key_padding_mask=key_padding_mask,
                            attn_mask=coattn_mask,
                            training=training)
        co_attn = self.dropout2(co_attn, training=training)
        co_attn = self.layernorm2(co_attn + self_attn)  # (batch_size, M, d_model)

        ffn_output = self.ffn(co_attn)  # (batch_size, M, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + co_attn)  # (batch_size, M, d_model)
        return out3


class Transformer(tf.keras.Model):
    """Will build a Transformer according to the paper
      [Fast Convergence of DETR with Spatially Modulated Co-Attention](https://arxiv.org/pdf/2101.07448.pdf).

    Args:
        num_layers: the number of sub-layers in the decoder and the encoder.
        d_model: The number of expected features in the encoder/decoder inputs
        num_heads: The number of heads in the multiheadattention models.
        dim_feedforward: The dim of the feedforward neuralnetworks in
            the EncoderLayer and DecoderLayer
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            The same rate is shared in all the layers using dropout in the transformer.

    Call arguments:
        flatten_tensor: A 3-D float32 Tensor of shape
            [batch_size, H * W, d_model].
            It represents the flatten output tensor of the backbone.
        pos_embed: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
            Positional spatial positional encoding matching the flatten_tensor.
        object_queries: A 3-D float32 Tensor of shape
            [batch_size, num_object_queries, d_model] small fixed number of
            learned positional embeddings input of the decoder.
        key_padding_mask:  A 2-D bool Tensor of shape [batch_size, H * W]
            where False means padding and True means pixels
            from the original image.
        coattn_mask:  A 4-D float tensor of shape
            [batch_size, num_heads, H*W, num_object_queries, seq_len].
            If provided, it will be added to the attention weight
            at the coattention step (memory x self_attn)

    Call Returns:
        decoder_output: 3-D float32 Tensor of shape [batch_size, h, d_model]
            where h is num_object_queries * num_layers if training is true and
            num_queries if training is set to False.
        encoder_output: 3-D float32 Tensor of shape [batch_size, batch_size, d_model]

    """

    def __init__(self,
                 num_layers=6,
                 d_model=256,
                 num_heads=8,
                 dim_feedforward=2048,
                 dropout_rate=0.1,
                 **kwargs):

        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout_rate)
            for _ in range(num_layers)
        ]

    def call(self,
             flatten_tensor,
             pos_embed,
             object_queries,
             key_padding_mask=None,
             coattn_mask=None,
             training=None):
        """
        Args:
            flatten_tensor: A 3-D float32 Tensor of shape
                [batch_size, H * W, d_model].
                It represents the flatten output tensor of the backbone.
            pos_embed: A 3-D float32 Tensor of shape [batch_size, H * W, d_model].
                Positional spatial positional encoding matching the flatten_tensor.
            object_queries: A 3-D float32 Tensor of shape
                [batch_size, num_object_queries, d_model] small fixed number of
                learned positional embeddings input of the decoder.
            key_padding_mask:  A 2-D bool Tensor of shape [batch_size, H * W]
                where False means padding and True means pixels
                from the original image.
            coattn_mask:  A 4-D float tensor of shape
                [batch_size, num_heads, H*W, num_object_queries, seq_len].
                If provided, it will be added to the attention weight
                at the coattention step (memory x self_attn)
        Returns:
            decoder_output: 3-D float32 Tensor of shape [batch_size, h, d_model]
                where h is num_object_queries * num_layers if training is true and
                num_queries if training is set to False.
            encoder_output: 3-D float32 Tensor of shape [batch_size, batch_size, d_model]
        """
        memory = flatten_tensor
        for enc in self.enc_layers:
            # (batch_size, seq_len, d_model)
            memory = enc(memory, pos_embed, key_padding_mask=key_padding_mask, training=training)

        # At the beginning we set target to 0
        # In the first decoder layer Q and K will be equal
        # to dec_out + object_queries=object_queries
        dec_out = tf.zeros_like(object_queries)
        layers_output = []
        for layer in self.dec_layers:
            dec_out = layer(dec_out,
                            memory,
                            pos_embed,
                            object_queries,
                            key_padding_mask=key_padding_mask,
                            coattn_mask=coattn_mask,
                            training=training)
            dec_out = self.layer_norm(dec_out)
            if training:
                layers_output.append(dec_out)

        if training:
            return tf.concat(layers_output, axis=1), memory

        return dec_out, memory
