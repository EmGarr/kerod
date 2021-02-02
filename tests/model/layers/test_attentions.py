import tensorflow as tf

from kerod.model.layers.attentions import MultiHeadAttention


def test_multihead_attention_no_mask():
    d_model = 16
    value, key, query = [
        tf.random.uniform(shape) for shape in ((2, 4, 10), (2, 4, 10), (2, 10, 10))
    ]
    mha = MultiHeadAttention(d_model, 4)
    assert mha(value, key, query).shape == (2, 10, 16)


def test_multihead_attention_with_mask():
    d_model = 16
    value, key, query = [
        tf.random.uniform(shape) for shape in ((2, 4, 10), (2, 4, 10), (2, 10, 10))
    ]

    key_padding_mask = tf.ones((2, 4), tf.bool)
    mha = MultiHeadAttention(d_model, 4)
    assert mha(value, key, query, key_padding_mask=key_padding_mask).shape == (2, 10, 16)
