import pytest
import tensorflow as tf
from kerod.model.layers.positional_encoding import PositionEmbeddingLearned

def test_positional_embedding_learned():
    pos_embed = PositionEmbeddingLearned(128)
    out = tf.random.uniform((3, 45, 50, 4))
    batch_emb = pos_embed(out)
    assert batch_emb.shape == (3, 45, 50, 128)

def test_catch_value_error():
    with pytest.raises(ValueError):
        pos_embed = PositionEmbeddingLearned(3)
