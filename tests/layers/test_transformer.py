import tensorflow as tf

from kerod.layers.transformer import Transformer


def test_transformer():
    d_model = 8
    transformer = Transformer(dim_feedforward=32, num_layers=2, d_model=d_model)

    flatten_tensor = tf.random.uniform((2, 100, d_model))
    pos_embed = tf.random.uniform((2, 100, d_model))
    object_queries = tf.random.uniform((2, 20, d_model))
    out_decoder, out_encoder = transformer.call(flatten_tensor, pos_embed, object_queries)

    # 20 matches the number of object queries
    assert out_decoder.shape == (2, 20, d_model)

    # 100 matches the number of flatten_tensor
    assert out_encoder.shape == (2, 100, d_model)
