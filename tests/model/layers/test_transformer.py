import tensorflow as tf

from kerod.model.layers.transformer import Transformer, MultiHeadAttention


def test_multihead_attention_no_mask():
    d_model = 16
    value, key, query = [
        tf.random.uniform(shape) for shape in ((2, 4, 10), (2, 4, 10), (2, 10, 10))
    ]
    mha = MultiHeadAttention(d_model, 4)
    assert mha((value, key, query, None)).shape == (2, 10, 16)


def test_multihead_attention_with_mask():
    d_model = 16
    value, key, query = [
        tf.random.uniform(shape) for shape in ((2, 4, 10), (2, 4, 10), (2, 10, 10))
    ]

    key_padding_mask = tf.ones((2, 4), tf.bool)
    mha = MultiHeadAttention(d_model, 4)
    assert mha((value, key, query, key_padding_mask)).shape == (2, 10, 16)


def test_transformer():
    d_model = 8
    transformer = Transformer(dim_feedforward=32, num_layers=2, d_model=d_model)

    flatten_tensor = tf.random.uniform((2, 100, d_model))
    pos_embed = tf.random.uniform((2, 100, d_model))
    object_queries = tf.random.uniform((2, 20, d_model))
    out_decoder, out_encoder = transformer.call((flatten_tensor, None, pos_embed, object_queries))

    # 20 matches the number of object queries
    assert out_decoder.shape == (2, 20, d_model)

    # 100 matches the number of flatten_tensor
    assert out_encoder.shape == (2, 100, d_model)
